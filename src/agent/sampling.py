"""Sampled MuZero의 핵심 추가분:
  (1) 정책 분포 beta 에서 행동 K개 샘플링 + 배터리/비행금지 마스킹
  (2) 방문수 대신 IS 보정된 정책 타깃 pi_hat

이산 waypoint로 가려면 이 파일의 샘플러만 '그리드 칸 카테고리컬'로
교체하면 돼 — mcts/networks/muzero 는 그대로.
"""
from __future__ import annotations

import torch
from torch.distributions import Normal

from .config import ModelConfig


class ContinuousActionSampler:
    """정책 파라미터 (mean, log_std) 로 정의된 대각 가우시안에서
    행동 (theta_norm, dist_norm) 을 뽑는다."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def _dist(self, policy_params):
        mean, log_std = policy_params
        return Normal(mean, log_std.exp())

    def sample(self, policy_params, battery: float):
        """K개 행동 + 각 행동의 제안분포 log-prob(β) 반환.
        battery 로 이동거리 상한을 마스킹."""
        K = self.cfg.num_sampled_actions
        dist = self._dist(policy_params)                 # 배치 없는 단일 상태
        actions = dist.sample((K,))                      # [K, action_dim]
        beta_logprob = dist.log_prob(actions).sum(-1)    # [K]

        # --- 마스킹(하드 제약): 배터리로 못 가는 거리는 잘라냄 ---
        # dist_norm(=actions[:,1]) 은 [0,1]. 최대 이동거리 = max_step_frac.
        # 남은 배터리로 갈 수 있는 최대 비율로 clamp.
        max_reach = min(1.0, battery / max(self.cfg.max_battery, 1e-6))
        theta = actions[:, 0].clamp(-1.0, 1.0)
        dist_c = actions[:, 1].clamp(0.0, 1.0) * max_reach
        actions = torch.stack([theta, dist_c], dim=-1)
        # TODO: 비행금지 구역 마스킹은 env의 land_mask/no-fly 로 여기서 추가.
        return actions, beta_logprob

    def network_logprob(self, policy_params, actions):
        """학습 시: 현재 네트워크가 (저장된) 샘플 행동들에 부여하는 log-prob."""
        dist = self._dist(policy_params)
        return dist.log_prob(actions).sum(-1)            # [K]


def is_corrected_policy_target(visit_counts: torch.Tensor,
                               beta_logprob: torch.Tensor) -> torch.Tensor:
    """방문수 -> IS 보정된 정책 타깃 pi_hat.

    beta 에서 뽑았기 때문에 자주 뽑힌 행동이 과대표현됨.
    pi_hat(a_i) ∝ N(a_i) / beta(a_i)  로 편향 보정.
    (Sampled MuZero, Hubert et al. 2021 의 핵심 한 곳)
    """
    beta = beta_logprob.exp().clamp(min=1e-8)
    w = visit_counts / beta
    if w.sum() <= 0:
        # 방문이 하나도 안 붙은 초기엔 균등으로 폴백
        return torch.full_like(w, 1.0 / len(w))
    return w / w.sum()
