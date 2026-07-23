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
CKPT = "proto_muzero_ckpt.pt"  # 없으면 학습 안 된 상태로 봄


def main():
    cfg = ModelConfig(grid_size=16, num_simulations=30, max_battery=TTL,
                      max_step_frac=MAX_STEP_FRAC)
    canvas = Canvas(region=REGION, resolution=RESOLUTION)
    pre = AirQualityPreprocessor(year=2024)
    snap = pre.get_snapshot(SNAPSHOT_DT, POLLUTANT)
    env = AirQualityEnv(canvas, snap, cfg, pollutant=POLLUTANT,
                        visible_ratio=VISIBLE_RATIO, bbox_pad=BBOX_PAD, seed=0)

    net = MuZeroNet(cfg)
    ckpt = ROOT / "results" / CKPT
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[모델] 체크포인트 로드: {CKPT}")
    else:
        print("[모델] 체크포인트 없음 — 학습 전 상태로 관찰")

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    extent = [canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat]
    mcts = SampledMuZeroMCTS(net, cfg)

    obs = env.reset()
    for t in range(max(INSPECT_STEPS) + 1):
        actions, visits, _, _, root = mcts.run(obs, env.battery, return_root=True)
        if t in INSPECT_STEPS:
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

    print(f"\n[저장] results/mcts_candidates_step*.png, mcts_tree_step*.png")


if __name__ == "__main__":
    main()
