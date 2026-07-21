"""스켈레톤 실행: 더미 환경에서 self-play 1 에피소드 -> 학습 1 스텝.

목적은 'h/g/f + 샘플링 + MCTS + 손실' 배선이 end-to-end로 도는지 확인.
성능이 아니라 파이프가 이어지는지가 포인트.

실행:  (레포 루트에서)  python -m src.agent.run_skeleton
"""
from __future__ import annotations

import torch

from src.agent.config import ModelConfig, TrainConfig
from src.agent.muzero import MuZeroNet
from src.agent.mcts import SampledMuZeroMCTS
from src.agent.losses import compute_losses
from src.env.dummy_env import DummyDroneEnv, InfoGainReward


def self_play(net, cfg: ModelConfig, env: DummyDroneEnv):
    """MCTS로 한 에피소드 굴리며 학습용 궤적 수집."""
    mcts = SampledMuZeroMCTS(net, cfg)
    obs = env.reset()
    traj: list[dict] = []
    prev_reward = 0.0
    done = False
    while not done:
        actions, visits, beta_lp, root_value = mcts.run(obs, env.battery)
        idx = int(visits.argmax())                 # 방문 최다 행동 선택
        chosen = actions[idx]
        traj.append(dict(obs=obs, action=chosen, sampled_actions=actions,
                         visit_counts=visits, beta_logprob=beta_lp,
                         reward=prev_reward))
        obs, reward, done = env.step(chosen)
        prev_reward = reward
    # 종료 상태 (손실은 이 지점의 action은 안 씀; reward만 사용)
    traj.append(dict(obs=obs, action=torch.zeros(cfg.action_dim),
                     sampled_actions=None, visit_counts=None,
                     beta_logprob=None, reward=prev_reward))

    # value 타깃 = 몬테카를로 리턴 (뒤에서부터 누적)
    G = 0.0
    for t in reversed(range(len(traj))):
        traj[t]["value_target"] = G
        G = traj[t]["reward"] + cfg.discount * G
    return traj


def main():
    mcfg, tcfg = ModelConfig(), TrainConfig()
    torch.manual_seed(tcfg.seed)

    net = MuZeroNet(mcfg)
    env = DummyDroneEnv(mcfg, reward_fn=InfoGainReward())
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[모델] MuZeroNet params = {n_params:,}")

    # --- self-play ---
    traj = self_play(net, mcfg, env)
    total_reward = sum(s["reward"] for s in traj)
    print(f"[self-play] 스텝 {len(traj)-1}, 총 정보이득 = {total_reward:.3f}")
    print(f"           루트 샘플 행동 예시 = {traj[0]['sampled_actions'][:2].tolist()}")
    print(f"           루트 방문수 = {traj[0]['visit_counts'].tolist()}")

    # --- 학습 1 스텝 ---
    total, parts = compute_losses(net, mcfg, traj)
    opt.zero_grad()
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()

    print(f"[학습] total_loss = {float(total.detach()):.4f} | "
          f"value={parts['value']:.4f} reward={parts['reward']:.4f} "
          f"policy={parts['policy']:.4f}")
    print(f"       grad_norm = {float(grad_norm):.4f}")
    print("[OK] 배선 통과 — h/g/f + 샘플링 + MCTS + 손실 end-to-end 동작")


if __name__ == "__main__":
    main()
