"""Sampled MuZero의 핵심 추가분:
  (1) 정책 분포 beta 에서 행동 K개 샘플링 (tanh squashing)
  (2) 방문수 대신 IS 보정된 정책 타깃 pi_hat

행동 파라미터화 (중요):
  신경망은 무한 범위 실수 u 를 뱉음 -> tanh 로 (-1,1) 에 부드럽게 매핑.
  예전엔 clamp 를 썼는데, mean 이 범위 밖으로 밀리면 샘플이 전부 경계값
  (theta=+-1, dist=0/1)으로 잘려서 후보 다양성이 사라졌음(정책 붕괴).
  tanh 는 경계에 '붙지' 않아서 서로 다른 출력이 서로 다른 행동으로 유지됨.

  변수변환이므로 log-prob 에 야코비안 보정 필요:
      log q(a) = log N(u) - log|da/du|
  이걸 빠뜨리면 IS 보정(pi_hat)이 틀려짐.

행동은 '정규 공간'(theta in (-1,1), dist in (0,1))으로 반환하고,
배터리에 따른 이동거리 제한은 env.step 에서 적용 (역변환 가능하게 하려고).

이산 waypoint로 가려면 이 파일의 샘플러만 '그리드 칸 카테고리컬'로
교체하면 돼 — mcts/networks/muzero 는 그대로.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from .config import ModelConfig

_EPS = 1e-6


def _squash(u: torch.Tensor) -> torch.Tensor:
    """u -> 행동.  theta: tanh -> (-1,1) / dist: (tanh+1)/2 -> (0,1)."""
    t = torch.tanh(u)
    theta = t[..., 0]
    dist = (t[..., 1] + 1.0) * 0.5
    return torch.stack([theta, dist], dim=-1)


def _unsquash(a: torch.Tensor) -> torch.Tensor:
    """행동 -> u (역변환). 학습 때 저장된 행동의 log-prob 계산용."""
    theta = a[..., 0].clamp(-1.0 + _EPS, 1.0 - _EPS)
    dist = (a[..., 1].clamp(_EPS, 1.0 - _EPS) * 2.0 - 1.0)
    return torch.stack([torch.atanh(theta), torch.atanh(dist)], dim=-1)


def _squashed_logprob(dist_n: Normal, u: torch.Tensor) -> torch.Tensor:
    """log q(a) = log N(u) - sum log|da/du|.
    da/du: theta 는 1-tanh^2, dist 는 (1-tanh^2)/2."""
    logp = dist_n.log_prob(u)                              # [..., 2]
    t = torch.tanh(u)
    log_det = torch.log(1.0 - t.pow(2) + _EPS)             # [..., 2]
    log_det = log_det.clone()
    log_det[..., 1] = log_det[..., 1] - float(np.log(2.0))  # dist 의 1/2 배
    return (logp - log_det).sum(-1)


class ContinuousActionSampler:
    """정책 파라미터 (mean, log_std) 로 정의된 대각 가우시안 + tanh squashing."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def _dist(self, policy_params, explore_std: float = 0.0) -> Normal:
        mean, log_std = policy_params
        if explore_std > 0.0:
            # 분포를 넓혀 탐색 촉진 (정책이 한 방향으로 굳는 것 방지)
            log_std = torch.logaddexp(
                log_std, torch.full_like(log_std, float(np.log(explore_std))))
        return Normal(mean, log_std.exp())

    def sample(self, policy_params, battery: float = None,
               explore_std: float = 0.0):
        """K개 행동(정규 공간) + 각 행동의 제안분포 log-prob(beta) 반환.
        battery 인자는 하위호환용으로 받되 사용 안 함 — 이동거리 제한은
        env.step 에서 적용(역변환 가능하게)."""
        K = self.cfg.num_sampled_actions
        dist_n = self._dist(policy_params, explore_std)
        u = dist_n.sample((K,))                      # [K, action_dim]
        actions = _squash(u)                         # [K, 2] in (-1,1)x(0,1)
        beta_logprob = _squashed_logprob(dist_n, u)  # [K]
        # TODO: 비행금지 구역 마스킹은 env 의 land_mask/no-fly 로 여기서 추가.
        return actions, beta_logprob

    def network_logprob(self, policy_params, actions: torch.Tensor) -> torch.Tensor:
        """학습 시: 현재 네트워크가 (저장된) 샘플 행동들에 부여하는 log-prob."""
        dist_n = self._dist(policy_params)
        u = _unsquash(actions)
        return _squashed_logprob(dist_n, u)          # [K]


def is_corrected_policy_target(visit_counts: torch.Tensor,
                               beta_logprob: torch.Tensor) -> torch.Tensor:
    """방문수 -> IS 보정된 정책 타깃 pi_hat.

    beta 에서 뽑았기 때문에 자주 뽑힌 행동이 과대표현됨.
    pi_hat(a_i) ∝ N(a_i) / beta(a_i)  로 편향 보정.
    (Sampled MuZero, Hubert et al. 2021 의 핵심 한 곳)

    log 공간에서 계산 — beta 가 아주 작으면 exp 가 0으로 죽어서
    가중치가 발산하기 때문.
    """
    log_w = torch.log(visit_counts.clamp(min=1e-8)) - beta_logprob
    if float(visit_counts.sum()) <= 0:
        return torch.full_like(log_w, 1.0 / len(log_w))
    return torch.softmax(log_w, dim=0)
