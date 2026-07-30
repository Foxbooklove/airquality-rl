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
               explore_std: float = 0.0, validate=None, fallback=None,
               max_rounds: int = 4):
        """K개 행동(정규 공간) + 각 행동의 제안분포 log-prob(beta) + 통계 반환.

        battery 인자는 하위호환용으로 받되 사용 안 함 — 이동거리 제한은
        env.step 에서 적용(역변환 가능하게).

        마스킹 (바다 비행 / 경계 이탈 금지):
          validate : callable(actions[N,2]) -> bool mask[N].
                     기하 판정은 호출자(mcts)가 한다 — 이 모듈은 분포만 안다.
          fallback : callable() -> action[2]. 유효 후보가 0개일 때 쓸 보증된 한 수.

        연속 가우시안이라 로짓을 -inf 로 미는 방식이 불가하므로 **기각 샘플링**을
        쓴다. 결과는 정책분포를 유효 영역으로 절단한 것과 동등하다.

        IS 보정은 깨지지 않는다: 절단분포는 q(a)/Z 이고 Z 는 이 노드의 모든
        샘플에 동일한 상수라, is_corrected_policy_target 의 log 공간 softmax 와
        losses 의 log_softmax 에서 모두 상쇄된다.

        fallback 을 유효 후보와 **섞지 않는** 것이 중요하다. 폴백 행동(제자리)은
        제안분포의 극단 꼬리라 beta_logprob 이 -12 수준이고, 섞으면
        log(N) - beta 가 폭발해 IS 보정 질량을 100% 독식한다(측정 확인).
        여기서는 유효 후보가 0개일 때만 쓰므로 후보 집합 크기가 1이 되고,
        그러면 losses 의 log_softmax 가 정확히 0이라 **정책 손실이 0**이 된다
        — 폴백 스텝은 정책 그래디언트를 아예 만들지 않아 자동으로 안전하다.
        (가치·보상 손실은 정상 적용된다. 그 전이는 실제로 일어났으므로 맞다.)
        """
        K = self.cfg.num_sampled_actions
        dist_n = self._dist(policy_params, explore_std)

        if validate is None:                          # 마스킹 없음 = 기존 동작
            u = dist_n.sample((K,))                   # [K, action_dim]
            return (_squash(u), _squashed_logprob(dist_n, u),
                    {"draws": K, "rejected": 0, "fallback": False, "n_valid": K})

        kept, n_drawn, n_rej = [], 0, 0
        for _ in range(max_rounds):
            u = dist_n.sample((4 * K,))               # 넉넉히 뽑아 기각
            ok = np.asarray(validate(_squash(u)), dtype=bool)
            n_drawn += len(u)
            n_rej += int((~ok).sum())
            if ok.any():
                kept.append(u[torch.as_tensor(ok)])
            if sum(len(t) for t in kept) >= K:
                break

        used_fallback = False
        if kept:
            u = torch.cat(kept, dim=0)[:K]             # 앞에서 K개 (iid 이므로 무편향)
        else:
            # 사방이 막힘 — 호출자가 준 보증된 한 수로 대체
            if fallback is None:
                raise RuntimeError(
                    "유효 후보가 0개인데 fallback 이 없습니다. "
                    "mcts 가 fallback 을 넘겨야 합니다.")
            used_fallback = True
            u = _unsquash(torch.as_tensor(fallback(), dtype=torch.float32)
                          ).unsqueeze(0)               # [1, action_dim]

        return (_squash(u), _squashed_logprob(dist_n, u),
                {"draws": n_drawn, "rejected": n_rej, "fallback": used_fallback,
                 "n_valid": len(u)})

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
