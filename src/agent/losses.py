"""학습 손실 = 가치 + 보상 + 정책.

'학습 목표 함수'는 두 군데서 정해짐:
  - 보상의 '의미'는 env의 RewardFn 슬롯 (지금은 InfoGainReward)
  - 손실의 '조성/가중치'는 여기 + config
정책 타깃은 Sampled MuZero 방식: 방문수가 아니라 IS 보정 pi_hat.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .sampling import (ContinuousActionSampler, is_corrected_policy_target)


def compute_losses(net, cfg: ModelConfig, trajectory: list[dict], t0: int = 0):
    """궤적의 t0 지점부터 unroll 하며 손실 계산.

    trajectory[t] = {
      obs, action[action_dim], reward(float),
      sampled_actions[K,action_dim], visit_counts[K], beta_logprob[K],
      value_target(float)
    }

    t0 를 받는 이유:
      예전엔 항상 0 에서만 unroll 해서 TTL=40 궤적의 앞 3스텝만 학습에 쓰였다
      (92% 폐기). ReplayBuffer.sample() 이 궤적과 함께 무작위 시작 지점을
      돌려주므로, 그걸 여기 넘기면 궤적 전 구간이 학습에 들어온다.

    반환: (총손실, 항별 dict). steps==0 이면 손실은 0 텐서 (호출자가 걸러도 되고
    그냥 더해도 무해하다).
    """
    sampler = ContinuousActionSampler(cfg)
    t0 = max(0, min(int(t0), len(trajectory) - 1))
    steps = min(cfg.unroll_steps, len(trajectory) - 1 - t0)

    v_loss = r_loss = p_loss = torch.zeros((), dtype=torch.float32)
    if steps <= 0:                       # 마지막 원소에서 시작하면 펼칠 게 없다
        return v_loss, {"value": 0.0, "reward": 0.0, "policy": 0.0, "steps": 0}

    # 루트: h + f
    s, policy_params, v = net.initial_inference(trajectory[t0]["obs"])

    for k in range(steps):
        t = t0 + k
        node = trajectory[t]
        # --- 정책 손실: pi_hat vs 현재 네트워크가 샘플행동에 준 확률 ---
        pi_hat = is_corrected_policy_target(node["visit_counts"],
                                            node["beta_logprob"])       # [K]
        logp = sampler.network_logprob(_unbatch(policy_params),
                                       node["sampled_actions"])         # [K]
        q = F.log_softmax(logp, dim=0)                                  # 샘플집합 위 정규화
        p_loss = p_loss - (pi_hat * q).sum()

        # --- 가치 손실 ---
        # reward_scale 은 여기서만 곱한다 — 보고되는 정보이득은 원단위 유지.
        v_target = torch.tensor(node["value_target"] * cfg.reward_scale,
                                dtype=torch.float32)
        v_loss = v_loss + F.mse_loss(v.squeeze(0), v_target)

        # --- 한 칸 펼치기: g(s, 실제 취한 행동) ---
        a = node["action"].unsqueeze(0)
        s, r, policy_params, v = net.recurrent_inference(s, a)

        # --- 보상 손실 ---
        r_target = torch.tensor(trajectory[t + 1]["reward"] * cfg.reward_scale,
                                dtype=torch.float32)
        r_loss = r_loss + F.mse_loss(r.squeeze(0), r_target)

    total = (cfg.value_loss_coef * v_loss
             + cfg.reward_loss_coef * r_loss
             + cfg.policy_loss_coef * p_loss) / max(steps, 1)
    return total, {"value": float(v_loss.detach()) / steps,
                   "reward": float(r_loss.detach()) / steps,
                   "policy": float(p_loss.detach()) / steps,
                   "steps": steps}


def _unbatch(policy_params):
    mean, log_std = policy_params
    return (mean.squeeze(0), log_std.squeeze(0))
