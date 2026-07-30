"""통합 학습/시연 엔트리포인트.

  python -m src.agent.train                          # 헤드리스, 300 에피소드, 바닥부터
  python -m src.agent.train --mode live              # 브라우저로 보면서 학습(무한)
  python -m src.agent.train --mode window            # matplotlib 창으로 보면서
  python -m src.agent.train --load best              # 최고 기록 체크포인트에서 이어서
  python -m src.agent.train --load latest            # 마지막 체크포인트에서 이어서
  python -m src.agent.train --load results/x.pt      # 특정 파일에서
  python -m src.agent.train --mode live --load best --episodes 0   # 조합

기본값은 '바닥부터 학습'이다 — 실수로 이전 결과에 이어붙는 사고를 막기 위해
이어서 하려면 --load 를 명시해야 한다.

체크포인트 (results/):
  ckpt_latest.pt  매 --save-every 에피소드마다 갱신
  ckpt_best.pt    최근 10 에피소드 평균 정보이득이 최고를 갱신할 때만 저장
                  (단일 에피소드는 시작 위치가 무작위라 편차가 커서 평균으로 판단)
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# results/ 를 용도별로 분리 (한 폴더에 다 쌓이면 찾기 어려움)
CKPT_DIR = RESULTS / "checkpoints"   # *.pt, gains.npy — 학습 상태
FIG_DIR = RESULTS / "figures"        # *.png — 학습 곡선, 불확실성, baseline
MAP_DIR = RESULTS / "maps"           # *.html — folium 지도


def ensure_dirs():
    for d in (CKPT_DIR, FIG_DIR, MAP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="무인기 대기질 강화학습 — 학습 및 실시간 시연",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("실행 방식")
    g.add_argument("--mode", choices=["headless", "live", "window"],
                   default="headless",
                   help="headless=화면 없이 빠르게, live=브라우저, window=matplotlib 창")
    g.add_argument("--episodes", type=int, default=None,
                   help="에피소드 수. 0이면 무한. 기본: headless=300, live/window=0")
    g.add_argument("--no-train", action="store_true",
                   help="학습 없이 추론만 (시연 전용)")
    g.add_argument("--port", type=int, default=8000, help="live 모드 웹 포트")
    g.add_argument("--public", action="store_true",
                   help="읽기 전용으로 공개 (관람객이 일시정지 등 제어 불가)")
    g.add_argument("--figures", action="store_true",
                   help="종료 시 figure + 랜덤 baseline 비교 생성 (headless 기본 on)")
    g.add_argument("--no-figures", dest="figures", action="store_false")
    p.set_defaults(figures=None)

    g = p.add_argument_group("체크포인트")
    g.add_argument("--load", default=None, metavar="WHAT",
                   help="'best' | 'latest' | 파일경로. 생략하면 바닥부터 학습")
    g.add_argument("--save-every", type=int, default=10,
                   help="이 에피소드마다 ckpt_latest.pt 저장")

    g = p.add_argument_group("문제 설정")
    g.add_argument("--region", default=None,
                   help="'서울특별시' 등. 생략하면 전국")
    g.add_argument("--resolution", type=int, default=60, help="Kriging 격자 해상도")
    g.add_argument("--snapshot", default="2024-07-01 13:00")
    g.add_argument("--pollutant", default="PM10")
    g.add_argument("--visible-ratio", type=float, default=0.4,
                   help="초기 가시 측정소 비율")
    g.add_argument("--bbox-pad", type=float, default=None,
                   help="캔버스 주변 이 비율만큼만 측정소 사용. 생략하면 전부")
    g.add_argument("--ttl", type=int, default=None,
                   help="무인기 이동 예산. 기본: headless=40, live/window=12")
    g.add_argument("--step-frac", type=float, default=0.06,
                   help="스텝당 최대 이동거리 (캔버스 대각선 대비)")
    g.add_argument("--mask", action="store_true", default=True,
                   help="바다 비행·경계 이탈을 수읽기 단계에서 차단")
    g.add_argument("--no-mask", dest="mask", action="store_false",
                   help="마스킹 끔 (기존 동작 — 드론이 바다 위를 난다)")
    g.add_argument("--mask-res", type=int, default=512,
                   help="경로 검사용 고해상 마스크 해상도. 캔버스 해상도와 별개")
    g.add_argument("--mainland-only", action="store_true", default=True,
                   help="시작 위치를 본토(최대 연결 성분)로 제한")
    g.add_argument("--allow-islands", dest="mainland_only", action="store_false",
                   help="섬에서도 시작 허용 (바다 횡단 금지라 갇힐 수 있음)")

    g = p.add_argument_group("보상 (복합 목적함수)")
    g.add_argument("--reward", default="info=1",
                   help="항별 가중치. 예: 'info=1,risk=2,hazard=0.5'. "
                        "항: info(정보이득) risk(위험가중) conc(농도가중) "
                        "hazard(기준치초과 판정 불확실성)")
    g.add_argument("--risk-csv", default=None, metavar="PATH",
                   help="지역별 호흡기 위험도 통계 CSV. risk 항을 켜려면 필수")
    g.add_argument("--risk-col", default=None, metavar="COL",
                   help="--risk-csv 에서 위험도로 쓸 값 컬럼명")
    g.add_argument("--risk-sido-col", default="시도")
    g.add_argument("--risk-sigungu-col", default="시군구",
                   help="시도 단위 통계면 'none' 을 주세요")
    g.add_argument("--risk-power", type=float, default=2.0,
                   help="위험도장 IDW 거리 감쇠 지수")
    g.add_argument("--hazard-threshold", type=float, default=None,
                   help="hazard 항 임계 농도. 생략하면 오염물질별 '나쁨' 기준")
    g.add_argument("--conc-floor", type=float, default=0.0,
                   help="conc 항에서 이 농도 미만은 가중치 0")

    g = p.add_argument_group("모델 / 탐색")
    g.add_argument("--sims", type=int, default=30, help="MCTS 시뮬레이션 횟수")
    g.add_argument("--sampled", type=int, default=None,
                   help="노드당 후보 행동 수 K. 기본: headless=8, live/window=4"
                        " (작을수록 트리가 깊어져 수읽기가 잘 보임)")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--seed", type=int, default=0)

    a = p.parse_args(argv)
    watching = a.mode in ("live", "window")
    if a.episodes is None:
        a.episodes = 0 if watching else 300
    if a.ttl is None:
        a.ttl = 12 if watching else 40
    if a.sampled is None:
        a.sampled = 4 if watching else 8
    if a.figures is None:
        a.figures = not watching
    return a


def resolve_ckpt(spec: str | None) -> Path | None:
    """--load 값을 실제 파일 경로로. 없으면 None(바닥부터)."""
    if spec is None:
        return None
    if spec in ("best", "latest"):
        name = f"ckpt_{spec}.pt"
        p = CKPT_DIR / name
        if not p.exists() and (RESULTS / name).exists():
            p = RESULTS / name           # 폴더 분리 이전에 저장된 것
    else:
        p = Path(spec)
        if not p.is_absolute() and not p.exists():
            p = ROOT / spec
    if not p.exists():
        avail = sorted(str(f.relative_to(RESULTS))
                       for f in RESULTS.rglob("*.pt")) if RESULTS.exists() else []
        msg = f"체크포인트 없음: {p}"
        if avail:
            msg += f"\n  results/ 에 있는 것: {', '.join(avail)}"
        else:
            msg += "\n  results/ 에 체크포인트가 아직 없습니다."
        msg += "\n  --load 를 빼면 바닥부터 학습합니다."
        raise SystemExit(msg)
    return p


def save_ckpt(path, net, opt, episode, gains, best_ma):
    """가중치만이 아니라 학습 상태 전체를 저장.

    옵티마이저 상태(Adam 모멘텀)와 best 기준값을 빼먹으면:
      - 재시작 때 Adam 이 0부터 시작해 학습이 잠시 흔들리고
      - best_ma 가 -inf 로 리셋돼 다음 몇 판이 무조건 '최고'로 판정,
        기존 ckpt_best.pt 를 더 나쁜 모델로 덮어쓴다.
    """
    import torch
    torch.save({"model": net.state_dict(),
                "optim": opt.state_dict() if opt is not None else None,
                "episode": episode,
                "gains": list(gains),
                "best_ma": best_ma}, path)


def load_ckpt(path, net, opt):
    """(episode, gains, best_ma) 반환. 옛 형식(가중치만)도 읽는다."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        net.load_state_dict(blob)          # 옛 형식: state_dict 그대로
        return 0, [], -float("inf")
    net.load_state_dict(blob["model"])
    if opt is not None and blob.get("optim") is not None:
        try:
            opt.load_state_dict(blob["optim"])
        except Exception as e:
            print(f"[모델] 옵티마이저 상태 복원 실패(무시): {e}")
    return (blob.get("episode", 0), list(blob.get("gains", [])),
            blob.get("best_ma", -float("inf")))


def build_risk_field(args, canvas, RiskField):
    """--risk-csv 가 있으면 지역 통계 → 측정소 앵커 → 연속 위험도장.

    없으면 None 을 돌려주고, risk 항이 꺼져 있는지 검사는 build_reward 가 한다
    (여기서 조용히 균등장으로 떨어뜨리면 '위험도를 켰다고 착각한 실험'이 나온다).
    """
    if args.risk_csv is None:
        return None
    if args.risk_col is None:
        raise SystemExit("--risk-csv 를 줬으면 --risk-col 도 필요합니다 (값 컬럼명).")
    sigungu = None if str(args.risk_sigungu_col).lower() == "none" \
        else args.risk_sigungu_col
    risk = RiskField.from_csv(args.risk_csv, canvas, value_col=args.risk_col,
                              sido_col=args.risk_sido_col, sigungu_col=sigungu,
                              power=args.risk_power)
    risk.summary()
    risk.plot(FIG_DIR / "risk_field.png")
    return risk


def main(argv=None):
    args = parse_args(argv)

    # 시각화 없는 모드는 창을 띄우지 않는다 (pyplot import 전에 지정)
    import matplotlib
    if args.mode != "window":
        matplotlib.use("Agg")

    import numpy as np
    import torch

    from src.agent.config import ModelConfig
    from src.agent.muzero import MuZeroNet
    from src.agent.mcts import SampledMuZeroMCTS
    from src.agent.losses import compute_losses
    from src.env.air_quality_env import AirQualityEnv
    from src.env.canvas import Canvas
    from src.env.motion import FlightMask
    from src.env.rewards import build_reward
    from src.env.risk_field import RiskField
    from src.data.preprocess import AirQualityPreprocessor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ensure_dirs()

    cfg = ModelConfig(grid_size=16, num_simulations=args.sims,
                      num_sampled_actions=args.sampled,
                      max_battery=args.ttl, max_step_frac=args.step_frac)

    print(f"[데이터] Canvas({args.region or '전국'}, res={args.resolution}) 로드 중...")
    canvas = Canvas(region=args.region, resolution=args.resolution)
    pre = AirQualityPreprocessor(year=2024)
    snap = pre.get_snapshot(args.snapshot, args.pollutant)

    risk = build_risk_field(args, canvas, RiskField)
    # 이름을 reward 로 두면 루프의 `obs, reward, done = env.step(...)` 이
    # 객체를 스칼라로 덮어쓴다 — reward_fn 으로 분리해야 한다.
    reward_fn = build_reward(args.reward, risk_field=risk, pollutant=args.pollutant,
                             hazard_threshold=args.hazard_threshold,
                             conc_floor=args.conc_floor)
    print(f"[보상] {reward_fn.describe()}")

    # 비행 마스크 — 바다 위 기능 고장 시 회수가 불가능해서 안전상 막는다.
    # 경로까지 검사하므로 canvas.land_mask(기본 60, 한 칸 8km)로는 부족하다.
    fmask = None
    if args.mask:
        fmask = FlightMask(canvas, resolution=args.mask_res)
        fmask.summary()

    env = AirQualityEnv(canvas, snap, cfg, pollutant=args.pollutant,
                        visible_ratio=args.visible_ratio,
                        bbox_pad=args.bbox_pad, seed=args.seed,
                        reward_fn=reward_fn,
                        flight_mask=fmask if args.mainland_only else None)
    print(f"         측정소 {env.n_stations}개, 육지칸 {int(canvas.land_mask.sum())}, "
          f"TTL={args.ttl}")

    net = MuZeroNet(cfg)
    train = not args.no_train
    opt = torch.optim.Adam(net.parameters(), lr=args.lr) if train else None

    resume_ep, resume_gains, resume_best = 0, [], -float("inf")
    ckpt = resolve_ckpt(args.load)
    if ckpt is not None:
        resume_ep, resume_gains, resume_best = load_ckpt(ckpt, net, opt)
        # relative_to 는 레포 밖 경로에서 예외를 던진다 — --load 는 임의 경로를
        # 허용하므로(절대경로 포함) 실패해도 죽지 않게 한다.
        try:
            shown = ckpt.relative_to(ROOT)
        except ValueError:
            shown = ckpt
        print(f"[모델] 이어서 학습: {shown}")
        if resume_ep:
            ma = float(np.mean(resume_gains[-10:])) if resume_gains else 0.0
            print(f"       누적 {resume_ep} 에피소드까지 진행됨 "
                  f"(최근10 평균 {ma:.3f}, best {resume_best:.3f})")
        else:
            print("       (옛 형식 — 가중치만 복원, 에피소드 기록은 없음)")

    print(f"[학습] {'on' if train else 'off'}"
          + (f", lr={args.lr}" if train else ""))

    # --- 시각화 준비 ---
    view = bus = ctl = None
    if args.mode in ("live", "window"):
        from src.agent.visualize_search import LiveSearchView
        if args.mode == "live":
            from src.agent.live_server import FrameBus, Controller, serve_in_background
            bus, ctl = FrameBus(), Controller()
            url = serve_in_background(bus, ctl, port=args.port,
                                      readonly=args.public)
            print(f"\n  ┌─ 브라우저에서 열기 ──────────────────────")
            print(f"  │  이 컴퓨터  : http://localhost:{args.port}")
            print(f"  │  같은 네트워크: {url}")
            if args.public:
                print(f"  │  (읽기 전용 — 관람객은 제어할 수 없습니다)")
            print(f"  └──────────────────────────────────────────\n")
        extent = [canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat]
        view = LiveSearchView(
            extent, canvas.land_mask, cfg.max_step_frac, cfg.max_battery,
            interactive=(args.mode == "window"),
            on_frame=(bus.publish if bus else None),
            gate=(ctl.gate if ctl else None),
            # 브라우저 속도 슬라이더를 실제로 연결한다. 안 넘기면 view.speed 가
            # None 이라 _pause 의 배율이 항상 1.0 이고, 애니메이션 대기 때문에
            # 무한 학습이 기어간다 (조절 UI 는 있는데 아무 효과가 없었다).
            speed=((lambda: ctl.speed) if ctl else None),
            descend_pause=0.07, backup_pause=0.035, commit_pause=0.5,
            title="MCTS 수읽기 — 파랑=확정, 주황=탐색, 초록=선택")

    extent = (canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat)
    mcts = SampledMuZeroMCTS(net, cfg, mask=fmask, extent=extent,
                             stations=(env.st_lon, env.st_lat))
    if fmask is not None:
        print("[마스킹] 바다 경로·경계 이탈 차단 on "
              "(폴백: 최근접 측정소 방향 → 제자리)")

    def status(**kw):
        if bus:
            bus.set_status(**kw)

    gains, best_ma, last_loss = resume_gains, resume_best, None
    episode = resume_ep
    endless = args.episodes == 0

    try:
        target = resume_ep + args.episodes   # 이번 실행에서 돌릴 만큼 더
        while endless or episode < target:
            episode += 1
            obs = env.reset()
            if view:
                view.set_belief(env.var); view.set_confirmed(env.traj)
            traj, prev_reward = [], 0.0
            step, total, done = 0, 0.0, False
            # 마스킹 진단 — 기각률이 높으면 후보 K개를 못 채워 트리가 좁아진다
            ep_reject, ep_fallback, ep_narrow = [], 0, 0

            while not done:
                pos = (env.drones[0]["lon"], env.drones[0]["lat"])
                bat = env.battery
                if view:
                    status(episode=episode, step=step, battery=f"{bat:.0f}/{args.ttl}",
                           info_gain=total, best=max(gains, default=0.0),
                           loss=last_loss, sims_total=args.sims, sim=0,
                           phase="수읽기 중")
                    view.set_info(
                        f"ep {episode}  |  step {step}  |  배터리 {bat:.0f}/{args.ttl}"
                        f"\n누적 정보이득 {total:.3f}")

                    def cb(info):
                        status(sim=info["sim"] + 1)
                        view.on_simulation(info, pos, bat)
                    on_sim = cb
                else:
                    on_sim = None

                actions, visits, beta_lp, _ = mcts.run(
                    obs, bat, on_simulation=on_sim, pos=pos)
                chosen = actions[int(visits.argmax())]
                ms = mcts.last_stats
                ep_reject.append(ms.get("reject_rate", 0.0))
                ep_fallback += int(ms.get("root_fallback", False))
                ep_narrow += int(len(actions) < args.sampled)

                traj.append(dict(obs=obs, action=chosen, sampled_actions=actions,
                                 visit_counts=visits, beta_logprob=beta_lp,
                                 reward=prev_reward))
                if view:
                    status(phase="수 선택")
                    view.commit(chosen, pos, bat)
                    status(phase="이동 + Kriging 재계산")

                obs, reward, done = env.step(chosen)
                prev_reward = reward
                total += reward
                step += 1
                if view:
                    view.set_belief(env.var); view.set_confirmed(env.traj)
                    status(step=step, info_gain=total)
                    view._pause(0.05)

            gains.append(total)

            # --- 학습 ---
            if train:
                if view:
                    status(phase="학습 중")
                traj.append(dict(obs=obs, action=torch.zeros(cfg.action_dim),
                                 sampled_actions=None, visit_counts=None,
                                 beta_logprob=None, reward=prev_reward))
                G = 0.0
                for t in reversed(range(len(traj))):
                    traj[t]["value_target"] = G
                    G = traj[t]["reward"] + cfg.discount * G

                loss, _ = compute_losses(net, cfg, traj)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                last_loss = float(loss.detach())

                # 최근 10 에피소드 평균으로 best 판단 (단일 에피소드는 편차가 큼)
                ma = float(np.mean(gains[-10:]))
                if len(gains) >= 3 and ma > best_ma:
                    best_ma = ma
                    save_ckpt(CKPT_DIR / "ckpt_best.pt", net, opt,
                              episode, gains, best_ma)
                if episode % args.save_every == 0:
                    save_ckpt(CKPT_DIR / "ckpt_latest.pt", net, opt,
                              episode, gains, best_ma)
                    np.save(CKPT_DIR / "gains.npy", np.array(gains))

            ma10 = float(np.mean(gains[-10:]))
            line = (f"  ep {episode:4d} | reward={total:.3f} "
                    f"(최근10 {ma10:.3f}) ")
            if last_loss is not None:
                line += f"| loss={last_loss:.3f}"
            # 항별 기여를 같이 찍는다 — λ 를 손으로 잡으려면 이게 없으면 못 한다
            bd = reward_fn.breakdown_str()
            if bd:
                line += f" | {bd}"
            if fmask is not None and ep_reject:
                line += f" | 기각 {np.mean(ep_reject)*100:.0f}%"
                if ep_narrow:
                    line += f" 좁음{ep_narrow}"
                if ep_fallback:
                    line += f" 폴백{ep_fallback}"
            print(line)

            if view:
                status(episode=episode, info_gain=total,
                       best=max(gains), loss=last_loss, phase="에피소드 종료")
                msg = f"ep {episode} 종료  |  정보이득 {total:.3f}  (최근10 {ma10:.3f})"
                if last_loss is not None:
                    msg += f"   loss {last_loss:.3f}"
                view.set_info(msg)
                view._pause(1.0)

    except KeyboardInterrupt:
        print("\n중단됨")

    if train:
        save_ckpt(CKPT_DIR / "ckpt_latest.pt", net, opt, episode, gains, best_ma)
        np.save(CKPT_DIR / "gains.npy", np.array(gains))
        print(f"[저장] results/checkpoints/ckpt_latest.pt"
              + (", ckpt_best.pt" if best_ma > -float("inf") else ""))

    # --- 종료 시 figure + baseline ---
    if args.figures and gains:
        make_figures(args, cfg, env, net, canvas, gains, mcts)

    if view:
        view.finish(out_path=FIG_DIR / "live_final.png")
    if args.mode == "window":
        input("창을 닫으려면 Enter...")


def make_figures(args, cfg, env, net, canvas, gains, mcts):
    """학습 곡선 + 불확실성 before/after + 랜덤 baseline 비교 + folium 지도."""
    import numpy as np
    import torch
    from src.agent import visualize as viz
    from src.agent import visualize_map as vmap

    def random_action(env):
        """랜덤 baseline도 **같은 제약**을 받아야 비교가 공정하다.

        마스킹을 켜면 학습 정책은 바다로 못 가는데, 분산이 제일 큰 곳이 바다다.
        랜덤에게만 그 자유를 주면 랜덤이 실제보다 좋아 보인다.
        """
        from src.env.motion import step_position, STAY
        d = env.drones[0]
        for _ in range(64):
            a = np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1)])
            if mcts.mask is None:
                return torch.tensor(a, dtype=torch.float32)
            lon1, lat1 = step_position(d["lon"], d["lat"], a, env.battery,
                                       cfg.max_battery, cfg.max_step_frac,
                                       mcts.extent, clip=False)
            if bool(mcts.mask.valid_step(d["lon"], d["lat"], lon1, lat1)[0]):
                return torch.tensor(a, dtype=torch.float32)
        return torch.tensor(STAY, dtype=torch.float32)      # 사방이 막힘

    def episode(random_policy=False):
        obs = env.reset(); var0 = env.var.copy()
        total, done = 0.0, False
        while not done:
            if random_policy:
                a = random_action(env)
            else:
                d = env.drones[0]
                acts, vis, _, _ = mcts.run(obs, env.battery,
                                           pos=(d["lon"], d["lat"]))
                a = acts[int(vis.argmax())]
            obs, r, done = env.step(a)
            total += r
        return total, list(env.traj), var0, env.var.copy()

    N = 10
    learned, random_ = [], []
    demo = None
    for _ in range(N):
        g, tr, v0, v1 = episode()
        learned.append(g)
        if demo is None or g > demo[0]:
            demo = (g, tr, v0, v1)
        random_.append(episode(random_policy=True)[0])

    lm, ls = float(np.mean(learned)), float(np.std(learned))
    rm, rs = float(np.mean(random_)), float(np.std(random_))
    print(f"[평가] {N}회 평균 — 학습 {lm:.3f}±{ls:.3f} | 랜덤 {rm:.3f}±{rs:.3f}")

    _, traj, var0, var1 = demo
    extent = [canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat]
    viz.uncertainty_before_after(var0, var1, traj, canvas.land_mask,
                                 FIG_DIR / "uncertainty.png", extent=extent)
    viz.learning_curve(gains, FIG_DIR / "learning_curve.png")
    viz.baseline_compare(lm, rm, FIG_DIR / "baseline.png",
                         learned_std=ls, random_std=rs, n=N)
    vi = env.visible_idx
    vmap.trajectory_map(env.st_lon[vi], env.st_lat[vi], env.st_val[vi],
                        traj, var1, canvas.land_mask,
                        (canvas.min_lon, canvas.min_lat,
                         canvas.max_lon, canvas.max_lat),
                        MAP_DIR / "trajectory.html",
                        pollutant=args.pollutant)
    print("[저장] results/maps/trajectory.html, "
          "results/figures/{uncertainty,learning_curve,baseline}.png")


if __name__ == "__main__":
    main()