"""출격 환경 학습 엔트리포인트 — 국지 고농도 사건 탐지.

  python -m src.agent.train_sortie --episodes 0        # 무한, 이어서(기본)
  python -m src.agent.train_sortie --episodes 0 --fresh # 바닥부터
  python -m src.agent.train_sortie --baseline greedy --episodes 40

체크포인트는 results/sortie/ 에 쌓인다:
  ckpt_latest.pt      --save-every 마다 갱신. **기본으로 여기서 이어서 학습한다**
  ckpt_ep000100.pt    --snapshot-every 마다 남는 되돌림 지점
  incompatible_*.pt   구조가 바뀌어 못 읽은 옛 체크포인트 (덮어쓰지 않고 보관)

기존 train.py 는 연속 행동 + 정보이득 경로다. 비교용으로 남겨두고 여기서
새 경로(격자 이산 행동 + 사건 탐지)를 돌린다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "sortie"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="출격 강화학습 — 국지 고농도 사건 탐지",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("실행")
    g.add_argument("--episodes", type=int, default=200,
                   help="0 이면 무한 (Ctrl+C 로 중단, 중단해도 저장됨)")
    g.add_argument("--no-train", action="store_true")
    g.add_argument("--baseline", choices=["none", "random", "greedy"], default="none",
                   help="학습 정책 대신 baseline 으로 굴린다 (비교용)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--save-every", type=int, default=20,
                   help="이 에피소드마다 ckpt_latest.pt 갱신")
    g.add_argument("--snapshot-every", type=int, default=100,
                   help="이 에피소드마다 번호 붙은 사본을 따로 남긴다 (되돌릴 지점). "
                        "0 이면 안 남김. 파일당 약 19MB")
    g.add_argument("--load", default="latest",
                   help="'latest' | 파일경로. **기본이 이어서 학습**이다 — "
                        "끊었다 다시 켜는 일이 잦으므로 진행분을 보존하는 쪽을 기본으로 둔다")
    g.add_argument("--fresh", action="store_true",
                   help="체크포인트를 무시하고 바닥부터. 곡선을 깨끗하게 보고 싶을 때")

    g = p.add_argument_group("문제 설정")
    g.add_argument("--resolution", type=int, default=36)
    g.add_argument("--mask-res", type=int, default=384)
    g.add_argument("--pollutant", default="PM10")
    g.add_argument("--threshold", type=float, default=80.0)
    g.add_argument("--local-only", action="store_true",
                   help="사건을 국지(전국평균 30~70분위)로 한정. 314건으로 줄지만 "
                        "초과 커버리지가 1.1%%라 학습 신호가 극히 희소하다. "
                        "기본은 광역 포함(1,565건, 커버리지 96.6%%)")
    g.add_argument("--hidden-ratio", type=float, default=0.35)
    g.add_argument("--battery-h", type=float, default=12.0,
                   help="드론 1대의 비행 가능 시간. 한 수가 최소 1시간이므로 "
                        "대략 이 값만큼의 스텝을 쓴다")
    g.add_argument("--speed-kmh", type=float, default=70.0)
    g.add_argument("--drones", type=int, default=3)
    g.add_argument("--w-event", type=float, default=1.0,
                   help="등록된 사건 최초 탐지 보상. 지표를 올리는 유일한 항")
    g.add_argument("--w-repeat", type=float, default=0.1,
                   help="같은 사건 재확인. 지표는 안 오르지만 근처라는 신호")
    g.add_argument("--w-dense", type=float, default=0.0,
                   help="측정값 크기 보상. 켜면 보상-지표 상관이 0.99 -> 0.47 로 "
                        "떨어진다(측정). 기본 0")
    g.add_argument("--dense-floor", type=float, default=0.5,
                   help="조밀 항이 켜지는 하한 (임계 대비 비율)")
    g.add_argument("--event-prob", type=float, default=0.7,
                   help="에피소드를 사건 시각에서 시작할 확률. 1.0 이면 항상 사건이 "
                        "있다고 학습해 과잉탐지가 된다")

    g = p.add_argument_group("탐색 / 학습")
    g.add_argument("--sims", type=int, default=200)
    g.add_argument("--temperature", type=float, default=1.0,
                   help="방문수 -> 행동 선택 온도. 후반에 0 으로 낮추면 탐욕적")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--replay", type=int, default=200)
    g.add_argument("--updates", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--reward-scale", default="auto")
    g.add_argument("--scale-warmup", type=int, default=5)
    g.add_argument("--channels", type=int, default=48)
    return p.parse_args(argv)


# ---------------------------------------------------------------- baselines
def greedy_action(env, mask):
    """즉시 기대이득 최대 — 진짜 비교 대상.

    belief 의 (평균, 분산) 으로 각 칸의 '초과확률 × 은닉측정소 존재' 를 계산해
    가장 높은 칸을 고른다. 미래를 보지 않으므로 배터리를 다 써버리고 일찍
    귀환하는 실패가 나타나야 정상이다.
    """
    from scipy.special import ndtr
    thr = env.spec.threshold
    sd = np.sqrt(np.clip(env.var, 1e-9, None))
    p_over = ndtr((env.mean - thr) / sd)               # [res,res]
    # 아직 안 가본 은닉 측정소가 반경 안에 있는 칸만 값이 있다
    gain = np.zeros_like(p_over)
    for h in env.hidden:
        if int(h) in env.visited_hidden:
            continue
        i, j = env._si[h], env._sj[h]
        gain[i, j] = max(gain[i, j], p_over[i, j])
    sc = np.where(mask, gain, -np.inf)
    if not np.isfinite(sc).any() or sc.max() <= 0:
        ii, jj = np.where(mask)
        k = np.random.randint(len(ii))
        return int(ii[k] * env.res + jj[k])
    return int(np.argmax(sc))


def random_action(env, mask):
    ii, jj = np.where(mask)
    k = np.random.randint(len(ii))
    return int(ii[k] * env.res + jj[k])


# ---------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    from src.agent.config import ModelConfig
    from src.agent.grid_networks import GridMuZeroNet
    from src.agent.grid_mcts import GridMCTS, policy_target
    from src.agent.grid_losses import compute_losses
    from src.agent.replay import ReplayBuffer
    from src.env.canvas import Canvas
    from src.env.motion import FlightMask
    from src.env.events import load_or_build, EventSpec
    from src.env.sortie_env import SortieEnv, SortieSpec, make_table

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    # 합성곱 정책 헤드는 '칸마다 로짓 하나'를 뱉는다. 그러려면 신경망 격자와
    # 환경의 행동 격자가 **같아야** 한다. 다르면 행동 공간과 정책 출력이 어긋난다.
    cfg = ModelConfig(grid_size=a.resolution)
    canvas = Canvas(region=None, resolution=a.resolution)
    fmask = FlightMask(canvas, resolution=a.mask_res)
    table, coord = make_table(pollutant=a.pollutant)
    espec = EventSpec(pollutant=a.pollutant, threshold=a.threshold,
                      local_only=a.local_only)
    events = load_or_build(espec)
    n_loc = int(events.is_local.sum())
    print(f"[사건] {len(events):,}건 (국지 {n_loc}, 광역·기타 {len(events)-n_loc})"
          + ("  — 국지만" if a.local_only else ""))
    spec = SortieSpec(pollutant=a.pollutant, threshold=a.threshold,
                      hidden_ratio=a.hidden_ratio, battery_h=a.battery_h,
                      speed_kmh=a.speed_kmh, n_drones=a.drones,
                      event_prob=a.event_prob, w_event=a.w_event,
                      w_event_repeat=a.w_repeat, w_dense=a.w_dense,
                      dense_floor=a.dense_floor)
    env = SortieEnv(canvas, table, coord, cfg, fmask, events, spec, seed=a.seed)
    print(f"[환경] 측정소 {table.shape[1]}개(은닉 {a.hidden_ratio:.0%}), "
          f"사건 {len(env.events)}건, 배터리 {a.battery_h:.0f}시간, 드론 {a.drones}대")

    net = GridMuZeroNet(cfg, channels=a.channels)
    print(f"[모델] 파라미터 {sum(p.numel() for p in net.parameters()):,}")
    train = not a.no_train and a.baseline == "none"
    opt = torch.optim.Adam(net.parameters(), lr=a.lr) if train else None
    mcts = GridMCTS(net, cfg, sims=a.sims)

    reward_sig = (a.w_event, a.w_repeat, a.w_dense, a.dense_floor, a.threshold)
    ep0 = 0
    ep0_scale_stale = False
    if a.load and not a.fresh:
        fp = OUT / "ckpt_latest.pt" if a.load == "latest" else Path(a.load)
        if not fp.exists():
            print(f"[모델] 체크포인트 없음 ({fp.name}) — 바닥부터 시작")
        else:
            # 구조가 바뀌면(예: 관측 평면 9->15) 옛 체크포인트를 못 읽는다.
            # 기본이 자동 로드이므로, 조용히 깨지지 않게 크게 알리고 새로 시작한다.
            blob = torch.load(fp, map_location="cpu", weights_only=False)
            try:
                net.load_state_dict(blob["model"])
                if opt is not None and blob.get("optim"):
                    opt.load_state_dict(blob["optim"])
                ep0 = blob.get("episode", 0)
                cfg.reward_scale = blob.get("reward_scale", 1.0)
                print(f"[모델] 이어서: {fp.name} (누적 {ep0} 에피소드, "
                      f"reward_scale={cfg.reward_scale:.2f})")
                # 보상 정의가 바뀌면 저장된 스케일이 맞지 않는다. 가중치는
                # 살리되 스케일만 다시 잡는다 — 안 그러면 가치 타깃이 통째로
                # 어긋난다(예: 보상 평균 5.21 -> 0.21 로 바뀐 적이 있다).
                if blob.get("reward_sig") != reward_sig:
                    print(f"       [주의] 보상 정의가 바뀌었습니다 "
                          f"— reward_scale 을 다시 측정합니다")
                    ep0_scale_stale = True
            except Exception as e:
                bak = fp.with_name(f"incompatible_ep{blob.get('episode', 0)}.pt")
                fp.rename(bak)
                print("=" * 66)
                print(f"[경고] 체크포인트가 지금 모델 구조와 안 맞습니다 — 바닥부터 시작합니다.")
                print(f"       {type(e).__name__}: {str(e)[:160]}")
                print(f"       옛 파일은 {bak.name} 로 옮겨 두었습니다 (덮어쓰지 않음).")
                print("=" * 66)

    if ep0_scale_stale:
        ep0 = ep0                      # 가중치·에피소드는 유지, 스케일만 재측정
    buf = ReplayBuffer(capacity=a.replay, seed=a.seed)
    auto = str(a.reward_scale).lower() == "auto"
    if not auto:
        cfg.reward_scale = float(a.reward_scale)
    scale_fixed = (not auto) or (ep0 > 0 and not ep0_scale_stale)
    warm, hist, n_upd = [], [], 0

    def save(episode):
        """ckpt_latest.pt 를 갱신하고, --snapshot-every 마다 번호 붙은 사본을 남긴다.

        되돌릴 지점이 없으면 '어느 시점이 제일 좋았나' 를 나중에 확인할 수 없다.
        용량이 파일당 약 19MB 라 100 에피소드마다면 감당할 만하다.
        """
        if not train:
            return
        blob = {"model": net.state_dict(), "optim": opt.state_dict(),
                "episode": episode, "reward_scale": cfg.reward_scale,
                "reward_sig": reward_sig, "hist": hist}
        torch.save(blob, OUT / "ckpt_latest.pt")
        if a.snapshot_every > 0 and episode % a.snapshot_every == 0:
            torch.save(blob, OUT / f"ckpt_ep{episode:06d}.pt")

    def episode_ids():
        """--episodes 0 이면 무한. 본문 들여쓰기를 건드리지 않으려고 생성자로 둔다."""
        e = ep0
        while a.episodes == 0 or e < ep0 + a.episodes:
            e += 1
            yield e

    ep = ep0
    last = None
    try:
        for ep in episode_ids():
            obs = env.reset()
            traj, prev_r, total, done = [], 0.0, 0.0, False
            while not done:
                mask = env.action_mask()
                if not mask.any():
                    break
                flat = mask.reshape(-1)
                if a.baseline == "random":
                    act = random_action(env, mask); pol = None
                elif a.baseline == "greedy":
                    act = greedy_action(env, mask); pol = None
                else:
                    counts, _ = mcts.run(obs, mask)
                    pol = policy_target(counts, a.temperature)
                    act = int(np.random.choice(len(pol), p=pol) if a.temperature > 0
                              else pol.argmax())
                if pol is not None:
                    traj.append(dict(obs=obs, action=act, mask=flat.copy(),
                                     policy=pol.copy(), reward=prev_r))
                obs, r, done = env.step(act)
                prev_r = r; total += r

            n_ev = len(env.found)
            n_site = len({g for g, _, _ in env.found})    # 서로 다른 지점 수
            # 보상은 (지점,시각) 단위인데 평가지표는 사건 단위다. 한 사건에
            # 눌러앉으면 n_ev 만 늘고 n_site 는 안 는다 — 그 차이를 보이게 둔다.
            n_evt = len(env.found_events)      # 서로 다른 '사건' 수 = 평가지표
            n_loc_hit = sum(1 for k in env.found_events if env.ev_is_local.get(k))
            hist.append((total, n_ev, len(env.visited_hidden), env.event_id,
                         n_site, n_evt, n_loc_hit))

            if train and traj:
                traj.append(dict(obs=obs, action=0, mask=traj[-1]["mask"],
                                 policy=traj[-1]["policy"], reward=prev_r))
                G = 0.0
                for t in reversed(range(len(traj))):
                    traj[t]["value_target"] = G
                    G = traj[t]["reward"] + cfg.discount * G
                buf.add(traj)
                if not scale_fixed:
                    warm.append(total)
                    if len(warm) < a.scale_warmup:
                        print(f"  ep {ep:4d} | 보상 {total:.3f} 탐지 {n_ev} "
                              f"[워밍업 {len(warm)}/{a.scale_warmup}]", flush=True)
                        continue
                    # 보상이 희소해지면서 워밍업 대부분이 0 이 된다. 중앙값을
                    # 쓰면 0 이 나와 스케일이 1.0 으로 굳는다 — 가치 타깃이
                    # 너무 작아져 가치 헤드에 그래디언트가 안 간다(예전에 겪은 문제).
                    # 0 이 아닌 리턴들의 평균을 쓰고, 그것도 없으면 계속 모은다.
                    nz = [x for x in warm if abs(x) > 1e-9]
                    if not nz:
                        if len(warm) < a.scale_warmup * 8:
                            print(f"  ep {ep:4d} | 보상 0 — 워밍업 연장 "
                                  f"({len(warm)}판째, 0 아닌 리턴을 기다림)",
                                  flush=True)
                            continue
                        med = float(np.mean(np.abs(warm))) or 1.0
                    else:
                        med = float(np.mean(np.abs(nz)))
                    cfg.reward_scale = 1.0 / med if med > 1e-9 else 1.0
                    scale_fixed = True
                    print(f"[스케일] 리턴 중앙값 {med:.4f} -> reward_scale="
                          f"{cfg.reward_scale:.2f}", flush=True)
                for _ in range(a.updates):
                    opt.zero_grad(); got = 0
                    for tr, t0 in buf.sample_batch(a.batch_size, cfg.unroll_steps):
                        l, parts = compute_losses(net, cfg, tr, t0=t0)
                        if parts["steps"] == 0:
                            continue
                        (l / a.batch_size).backward(); got += 1; last = parts
                    if got:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                        opt.step(); n_upd += 1
                if ep % a.save_every == 0:
                    save(ep)

            # 최근 50 이동평균으로 본다. 20 은 사건 발생이 성긴 탓에 너무 흔들린다.
            w = hist[-50:]
            rec = np.mean([h[5] for h in w])
            rloc = np.mean([h[6] for h in w])
            line = (f"  ep {ep:4d} | 사건 {rec:.2f} (국지 {rloc:.2f}) "
                    f"| 이번 {n_evt}/{n_loc_hit} | 탐지 {n_ev}건 {n_site}지점 "
                    f"| 방문 {len(env.visited_hidden)}곳")
            if train and last:
                line += (f" | loss v {last['value']:.3f} r {last['reward']:.3f} "
                         f"p {last['policy']:.3f} | upd {n_upd}")
            print(line, flush=True)
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지 상태를 저장합니다")

    det = np.array([h[1] for h in hist], dtype=float)
    site = np.array([h[4] for h in hist], dtype=float)
    evt = np.array([h[5] for h in hist], dtype=float)
    n = max(len(evt), 1)
    with_ev = [h for h in hist if h[3] >= 0]
    # **사건**이 평가지표다. 탐지 건수는 (지점,시각) 단위라 한 곳에 눌러앉으면
    # 부풀려진다 — 둘을 같이 찍어 그 차이가 보이게 한다.
    print(f"\n[결과] {len(hist)} 에피소드")
    loc = np.array([h[6] for h in hist], dtype=float)
    print(f"  사건   {evt.mean():.2f} ± {evt.std()/np.sqrt(n):.2f} 건/에피소드  <- 평가지표")
    print(f"         그중 국지 {loc.mean():.2f}건 "
          f"({loc.mean()/max(evt.mean(),1e-9)*100:.0f}%)  — 광역은 쉬운 문제라 "
          f"국지 성능이 실질이다")
    print(f"  탐지   {det.mean():.2f}건 ({site.mean():.2f}지점)/에피소드")
    if with_ev:
        hit = sum(1 for h in with_ev if h[5] > 0) / len(with_ev)
        print(f"  사건 포함 에피소드 {len(with_ev)}개 중 {hit*100:.0f}% 에서 그 사건을 탐지")
    if train:
        save(ep)
        print(f"[저장] {OUT / 'ckpt_latest.pt'}")
    return hist


if __name__ == "__main__":
    main()
