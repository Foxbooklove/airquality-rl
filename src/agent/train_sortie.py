"""출격 환경 학습 엔트리포인트 — 국지 고농도 사건 탐지.

  python -m src.agent.train_sortie --episodes 200
  python -m src.agent.train_sortie --baseline greedy --episodes 50 --no-train
  python -m src.agent.train_sortie --load latest --episodes 100

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
    g.add_argument("--save-every", type=int, default=20)
    g.add_argument("--load", default=None, help="'latest' | 파일경로")

    g = p.add_argument_group("문제 설정")
    g.add_argument("--resolution", type=int, default=36)
    g.add_argument("--mask-res", type=int, default=384)
    g.add_argument("--pollutant", default="PM10")
    g.add_argument("--threshold", type=float, default=80.0)
    g.add_argument("--hidden-ratio", type=float, default=0.35)
    g.add_argument("--battery-h", type=float, default=12.0,
                   help="드론 1대의 비행 가능 시간. 한 수가 최소 1시간이므로 "
                        "대략 이 값만큼의 스텝을 쓴다")
    g.add_argument("--speed-kmh", type=float, default=70.0)
    g.add_argument("--drones", type=int, default=3)
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
    events = load_or_build(EventSpec(pollutant=a.pollutant, threshold=a.threshold))
    spec = SortieSpec(pollutant=a.pollutant, threshold=a.threshold,
                      hidden_ratio=a.hidden_ratio, battery_h=a.battery_h,
                      speed_kmh=a.speed_kmh, n_drones=a.drones,
                      event_prob=a.event_prob)
    env = SortieEnv(canvas, table, coord, cfg, fmask, events, spec, seed=a.seed)
    print(f"[환경] 측정소 {table.shape[1]}개(은닉 {a.hidden_ratio:.0%}), "
          f"사건 {len(env.events)}건, 배터리 {a.battery_h:.0f}시간, 드론 {a.drones}대")

    net = GridMuZeroNet(cfg, channels=a.channels)
    print(f"[모델] 파라미터 {sum(p.numel() for p in net.parameters()):,}")
    train = not a.no_train and a.baseline == "none"
    opt = torch.optim.Adam(net.parameters(), lr=a.lr) if train else None
    mcts = GridMCTS(net, cfg, sims=a.sims)

    ep0 = 0
    if a.load:
        fp = OUT / "ckpt_latest.pt" if a.load == "latest" else Path(a.load)
        blob = torch.load(fp, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["model"])
        if opt is not None and blob.get("optim"):
            opt.load_state_dict(blob["optim"])
        ep0 = blob.get("episode", 0)
        cfg.reward_scale = blob.get("reward_scale", 1.0)
        print(f"[모델] 이어서: {fp.name} (누적 {ep0} 에피소드, "
              f"reward_scale={cfg.reward_scale:.1f})")

    buf = ReplayBuffer(capacity=a.replay, seed=a.seed)
    auto = str(a.reward_scale).lower() == "auto"
    if not auto:
        cfg.reward_scale = float(a.reward_scale)
    scale_fixed = (not auto) or ep0 > 0
    warm, hist, n_upd = [], [], 0

    def save(episode):
        if not train:
            return
        torch.save({"model": net.state_dict(), "optim": opt.state_dict(),
                    "episode": episode, "reward_scale": cfg.reward_scale,
                    "hist": hist}, OUT / "ckpt_latest.pt")

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
            hist.append((total, n_ev, len(env.visited_hidden), env.event_id,
                         n_site, n_evt))

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
                    med = float(np.median(np.abs(warm))) or 1.0
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

            rec = np.array([h[5] for h in hist[-20:]], dtype=float)   # 사건 기준
            line = (f"  ep {ep:4d} | 보상 {total:6.3f} | 사건 {n_evt} "
                    f"(최근20 평균 {rec.mean():.2f}) | 탐지 {n_ev}건 "
                    f"{n_site}지점 | 방문 {len(env.visited_hidden)}곳")
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
    print(f"  사건   {evt.mean():.2f} ± {evt.std()/np.sqrt(n):.2f} 건/에피소드  <- 평가지표")
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
