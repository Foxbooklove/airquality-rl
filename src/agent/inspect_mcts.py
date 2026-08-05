"""MCTS 탐색 트리 뜯어보기 (디버깅 전용).

실행:  python -m src.agent.inspect_mcts

학습된 체크포인트(있으면)를 불러와 한 스텝의 탐색 트리를 그려줌.
results/ 에 저장:
  mcts_candidates_step{t}.png  — 불확실성 지도 위 후보 행동 부채꼴
  mcts_tree_step{t}.png        — 트리 구조

봐야 할 것:
  1. 화살표가 사방으로 뻗나, 한쪽으로만 몰리나  (정책 붕괴 여부)
  2. 방문수가 후보마다 다른가, 다 똑같은가      (PUCT가 구분하나)
  3. 방문 많은 화살표가 밝은(불확실성 높은) 쪽을 향하나  (가치가 맞나)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.agent.config import ModelConfig
from src.agent.muzero import MuZeroNet
from src.agent.mcts import SampledMuZeroMCTS
from src.agent import visualize_mcts as vm
from src.env.air_quality_env import AirQualityEnv
from src.env.canvas import Canvas
from src.data.preprocess import AirQualityPreprocessor

ROOT = Path(__file__).resolve().parents[2]

# run_real.py 의 SETTINGS 와 맞춰야 의미 있음
REGION = None
RESOLUTION = 60
SNAPSHOT_DT = "2024-07-01 13:00"
POLLUTANT = "PM10"
VISIBLE_RATIO = 0.4
BBOX_PAD = None
TTL = 40
MAX_STEP_FRAC = 0.06

INSPECT_STEPS = [0, 5, 15]     # 이 스텝들의 트리를 저장
CKPT = "ckpt_latest.pt"  # 없으면 학습 안 된 상태로 봄


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="MCTS 탐색 트리 뜯어보기")
    ap.add_argument("--resolution", type=int, default=RESOLUTION,
                    help="Kriging 격자 해상도 (낮추면 빠름)")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--ttl", type=int, default=TTL)
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--sampled", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=INSPECT_STEPS,
                    help="트리를 저장할 스텝 번호들")
    ap.add_argument("--no-mask", action="store_true",
                    help="바다·경계 마스킹 끄고 관찰 (기존 동작 비교용)")
    ap.add_argument("--mask-res", type=int, default=512)
    a = ap.parse_args(argv)

    cfg = ModelConfig(grid_size=16, num_simulations=a.sims,
                      num_sampled_actions=a.sampled, max_battery=a.ttl,
                      max_step_frac=MAX_STEP_FRAC)
    canvas = Canvas(region=a.region, resolution=a.resolution)
    pre = AirQualityPreprocessor(year=2024)
    snap = pre.get_snapshot(SNAPSHOT_DT, POLLUTANT)
    # 디버깅 도구는 train.py 와 같은 조건을 봐야 의미가 있다 — 마스킹을 끄면
    # 화살표가 바다로 뻗는 걸 보게 되는데 그건 학습 때 실제 동작이 아니다.
    from src.env.motion import FlightMask
    fmask = None if a.no_mask else FlightMask(canvas, resolution=a.mask_res)
    env = AirQualityEnv(canvas, snap, cfg, pollutant=POLLUTANT,
                        visible_ratio=VISIBLE_RATIO, bbox_pad=BBOX_PAD, seed=0,
                        flight_mask=fmask)
    inspect_steps = list(a.steps)

    net = MuZeroNet(cfg)
    ckpt = ROOT / "results" / "checkpoints" / CKPT
    if not ckpt.exists():
        ckpt = ROOT / "results" / CKPT      # 폴더 분리 이전 위치
    if ckpt.exists():
        # train.py 와 같은 로더 사용 — 체크포인트가 가중치만이 아니라
        # {model, optim, episode, gains, best_ma} 딕셔너리라서 직접 읽으면 깨진다.
        from src.agent.train import load_ckpt
        ep, _, _, _ = load_ckpt(ckpt, net, None)
        print(f"[모델] 체크포인트 로드: {CKPT}"
              + (f" (누적 {ep} 에피소드)" if ep else ""))
    else:
        print("[모델] 체크포인트 없음 — 학습 전 상태로 관찰")

    out = ROOT / "results" / "debug"; out.mkdir(parents=True, exist_ok=True)
    extent = [canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat]
    mcts = SampledMuZeroMCTS(net, cfg, mask=fmask, extent=tuple(extent),
                             stations=(env.st_lon, env.st_lat))

    obs = env.reset()
    for t in range(max(inspect_steps) + 1):
        d = env.drones[0]
        actions, visits, _, _, root = mcts.run(
            obs, env.battery, return_root=True, pos=(d["lon"], d["lat"]))
        if fmask is not None:
            s = mcts.last_stats
            print(f"[마스킹] 기각 {s['reject_rate']*100:.1f}% "
                  f"(후보 {len(actions)}개"
                  + (", 폴백" if s["root_fallback"] else "") + ")")
        if t in inspect_steps:
            print(f"\n===== step {t} | 드론 위치 "
                  f"({env.drones[0]['lon']:.3f}, {env.drones[0]['lat']:.3f}) =====")
            print(vm.summarize_root(root))
            vm.plot_root_candidates(
                root, (env.drones[0]["lon"], env.drones[0]["lat"]),
                env.var, canvas.land_mask, extent, cfg.max_step_frac,
                out / f"mcts_candidates_step{t}.png",
                title_extra=f"step {t}, battery {env.battery:.0f}")
            vm.plot_tree(root, out / f"mcts_tree_step{t}.png",
                         max_depth=2, top_k=4)
        a = actions[int(visits.argmax())]
        obs, r, done = env.step(a)
        if done:
            break

    print(f"\n[저장] results/debug/mcts_candidates_step*.png, mcts_tree_step*.png")


if __name__ == "__main__":
    main()