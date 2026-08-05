"""격자 이산 행동용 손실 — 가치 + 보상 + 정책(교차엔트로피).

기존 losses.py 와의 차이:
  정책 타깃  IS 보정 pi_hat  ->  **방문수 정규화** (전수 확장이라 보정 불필요)
  정책 손실  샘플집합 위 log_softmax  ->  **유효칸 위 masked log_softmax**

reward_scale 은 여기서만 곱한다 — 보고되는 탐지 수·보상은 원단위 유지.
왜 필요한지는 config.ModelConfig.reward_scale 주석 참고.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_losses(net, cfg, traj: list[dict], t0: int = 0):
    """traj[t] = {obs, action(int), mask[G*G] bool, policy[G*G], reward, value_target}

    t0 부터 cfg.unroll_steps 만큼 펼친다. ReplayBuffer 가 무작위 t0 를 주므로
    궤적 전 구간이 학습에 들어온다.
    """
    t0 = max(0, min(int(t0), len(traj) - 1))
    steps = min(cfg.unroll_steps, len(traj) - 1 - t0)
    z = torch.zeros((), dtype=torch.float32)
    if steps <= 0:
        return z, {"value": 0.0, "reward": 0.0, "policy": 0.0, "steps": 0}

    s, logits, v = net.initial_inference(traj[t0]["obs"])
    v_loss = r_loss = p_loss = z

    for k in range(steps):
        node = traj[t0 + k]
        # --- 정책: 방문수 분포 vs 신경망 (유효칸 위에서만 정규화) ---
        m = torch.as_tensor(node["mask"], dtype=torch.bool).view(1, -1)
        tgt = torch.as_tensor(node["policy"], dtype=torch.float32).view(1, -1)
        neg = torch.finfo(logits.dtype).min
        logq = torch.log_softmax(logits.view(1, -1).masked_fill(~m, neg), dim=-1)
        p_loss = p_loss - (tgt * logq.clamp(min=-30.0)).sum()

        # --- 가치 ---
        vt = torch.tensor(node["value_target"] * cfg.reward_scale, dtype=torch.float32)
        v_loss = v_loss + F.mse_loss(v.view(()), vt)

        # --- 한 칸 펼치기 ---
        a = torch.tensor([int(node["action"])])
        s, r, logits, v = net.recurrent_inference(s, a)

        # --- 보상 ---
        rt = torch.tensor(traj[t0 + k + 1]["reward"] * cfg.reward_scale,
                          dtype=torch.float32)
        r_loss = r_loss + F.mse_loss(r.view(()), rt)

    total = (cfg.value_loss_coef * v_loss + cfg.reward_loss_coef * r_loss
             + cfg.policy_loss_coef * p_loss) / steps
    return total, {"value": float(v_loss.detach()) / steps,
                   "reward": float(r_loss.detach()) / steps,
                   "policy": float(p_loss.detach()) / steps,
                   "steps": steps}
