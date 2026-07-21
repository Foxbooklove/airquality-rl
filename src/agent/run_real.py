"""실제 서울 데이터로 Sampled MuZero 프로토타입 학습 + PPT figure 생성.

실행 (레포 루트):  python -m src.agent.run_real

흐름:
  1) Canvas(서울) + 스냅샷 로드 -> AirQualityEnv (step마다 Kriging)
  2) N 에피소드 self-play + 학습 (에피소드별 정보이득 기록)
  3) 데모 에피소드로 belief before/after + 궤적 캡처
  4) 랜덤 baseline 과 비교
  5) results/ 에 figure 3장 + 모델 저장

느리지만(스텝마다 Kriging) 프로토타입/설득용이라 OK.
파라미터는 아래 SETTINGS 에서 조절.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.agent.config import ModelConfig, TrainConfig
from src.agent.muzero import MuZeroNet
from src.agent.losses import compute_losses
from src.agent.run_skeleton import self_play
from src.agent import visualize as viz
from src.agent import visualize_map as vmap
from src.env.air_quality_env import AirQualityEnv

# 실제 데이터 모듈 (레포에 이미 있음)
from src.env.canvas import Canvas
from src.data.preprocess import AirQualityPreprocessor

ROOT = Path(__file__).resolve().parents[2]

# ---------------- SETTINGS ----------------
REGION = "서울특별시"
RESOLUTION = 60            # Kriging 격자 (크면 정확·느림)
SNAPSHOT_DT = "2024-07-01 13:00"
POLLUTANT = "PM10"
VISIBLE_RATIO = 0.4       # A(초기 가시 측정소) 비율 — 낮을수록 드론 여지 큼
EPISODES = 40


def run_episode(net, cfg, env, greedy=True, random_policy=False):
    """한 에피소드 굴리고 (총 정보이득, 궤적, 초기/최종 분산) 반환."""
    obs = env.reset()
    var0 = env.var.copy()
    from src.agent.mcts import SampledMuZeroMCTS
    mcts = SampledMuZeroMCTS(net, cfg)
    total = 0.0
    done = False
    while not done:
        if random_policy:
            action = torch.tensor([np.random.uniform(-1, 1),
                                   np.random.uniform(0, 1)], dtype=torch.float32)
        else:
            actions, visits, _, _ = mcts.run(obs, env.battery)
            action = actions[int(visits.argmax())]
        obs, reward, done = env.step(action)
        total += reward
    return total, list(env.traj), var0, env.var.copy()


def main():
    mcfg = ModelConfig(grid_size=16, num_simulations=30, max_battery=12)
    tcfg = TrainConfig(lr=1e-3, seed=0)
    torch.manual_seed(tcfg.seed); np.random.seed(tcfg.seed)

    print(f"[데이터] Canvas({REGION}, res={RESOLUTION}) + 스냅샷 {SNAPSHOT_DT} 로드")
    canvas = Canvas(region=REGION, resolution=RESOLUTION)
    pre = AirQualityPreprocessor(year=2024)
    snap = pre.get_snapshot(SNAPSHOT_DT, POLLUTANT)
    print(f"         측정소 {len(snap)}개, 육지칸 {int(canvas.land_mask.sum())}")

    env = AirQualityEnv(canvas, snap, mcfg, pollutant=POLLUTANT,
                        visible_ratio=VISIBLE_RATIO, seed=tcfg.seed)
    net = MuZeroNet(mcfg)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr)

    # --- 학습 ---
    gains = []
    for ep in range(EPISODES):
        traj = self_play(net, mcfg, env)
        total, parts = compute_losses(net, mcfg, traj)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        g = sum(s["reward"] for s in traj)
        gains.append(g)
        if ep % 5 == 0 or ep == EPISODES - 1:
            print(f"  ep {ep:3d} | info_gain={g:.3f} | loss={float(total.detach()):.3f}")

    # --- 데모 + baseline ---
    learned_gain, traj, var0, var1 = run_episode(net, mcfg, env)
    random_gain, *_ = run_episode(net, mcfg, env, random_policy=True)
    print(f"[결과] 학습 정책 info_gain={learned_gain:.3f} | 랜덤={random_gain:.3f}")

    # --- figure 저장 ---
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    extent = [canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat]
    viz.uncertainty_before_after(var0, var1, traj, canvas.land_mask,
                                 out / "proto_uncertainty.png", extent=extent)
    viz.learning_curve(gains, out / "proto_learning_curve.png")
    viz.baseline_compare(learned_gain, random_gain, out / "proto_baseline.png")

    # folium HTML 지도 (실제 지도 위 무작위 시작 -> 드론 경로 + 불확실성 오버레이)
    vis_idx = env.visible_idx
    vmap.trajectory_map(
        env.st_lon[vis_idx], env.st_lat[vis_idx], env.st_val[vis_idx],
        traj, var1, canvas.land_mask,
        (canvas.min_lon, canvas.min_lat, canvas.max_lon, canvas.max_lat),
        out / "proto_trajectory_map.html", pollutant=POLLUTANT)

    torch.save(net.state_dict(), out / "proto_muzero.pt")
    print(f"[저장] results/proto_trajectory_map.html (folium 지도), "
          f"proto_uncertainty.png, proto_learning_curve.png, "
          f"proto_baseline.png, proto_muzero.pt")


if __name__ == "__main__":
    main()
