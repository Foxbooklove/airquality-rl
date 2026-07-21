"""Sampled MuZero의 세 신경망.

  h  표현망 (representation): 관측 -> 잠재상태 s
  g  동역학망 (dynamics):     (s, a) -> (s', 보상 r)
  f  예측망 (prediction):     s -> (정책 파라미터, 가치 V)

설계 메모:
- 측정점은 '가변 개수 집합'이라 순서무관 인코더(DeepSets)로 처리.
  (attention/Set Transformer로 교체 가능 — 여기가 '구조 결정 #3')
- belief(Kriging mean+variance) 그리드는 작은 CNN으로.
  belief를 h 입력으로 명시 제공하는 쪽을 택함 (= '구조 결정 #1'의 실용안).
  잠재가 belief를 알아서 학습하게 하려면 이 CNN 입력을 빼면 됨.
- 정책 헤드는 연속 (theta, dist)에 대한 대각 가우시안 (= '구조 결정 #2').
  theta의 순환성(각도)은 스켈레톤에선 무시; von Mises로 교체 가능.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


def mlp(sizes: list[int], act=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class SetEncoder(nn.Module):
    """DeepSets: 측정점 집합 -> 고정 임베딩 (순서/개수 무관)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.phi = mlp([cfg.point_feat_dim, cfg.hidden_dim, cfg.hidden_dim])
        self.rho = mlp([cfg.hidden_dim, cfg.hidden_dim])

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # points: [B, N, F], mask: [B, N] (1=유효, 0=패딩)
        h = self.phi(points)                       # [B, N, H]
        m = mask.unsqueeze(-1)                      # [B, N, 1]
        summed = (h * m).sum(dim=1)                 # 마스크된 합
        count = m.sum(dim=1).clamp(min=1.0)
        pooled = summed / count                    # 평균 풀링
        return self.rho(pooled)                    # [B, H]


class GridEncoder(nn.Module):
    """belief 그리드(mean, variance) -> 임베딩."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cfg.belief_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, cfg.hidden_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)                      # [B, H]


class RepresentationNetwork(nn.Module):
    """h: 관측 -> 잠재상태 s0."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.set_enc = SetEncoder(cfg)
        self.grid_enc = GridEncoder(cfg)
        self.drone_enc = mlp([cfg.drone_feat_dim, cfg.hidden_dim])
        self.head = mlp([3 * cfg.hidden_dim, cfg.hidden_dim, cfg.latent_dim])

    def forward(self, obs) -> torch.Tensor:
        e_pts = self.set_enc(obs.points, obs.mask)
        e_grid = self.grid_enc(obs.belief_grid)
        e_drone = self.drone_enc(obs.drone)
        s = self.head(torch.cat([e_pts, e_grid, e_drone], dim=-1))
        return normalize_latent(s)


class DynamicsNetwork(nn.Module):
    """g: (s, a) -> (s', r).  드론 이동은 알지만, 여기선 belief 갱신/미측정값을
    잠재공간에서 학습하는 자리 (최종 reward가 확률적일 때 값을 함)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.trunk = mlp([cfg.latent_dim + cfg.action_dim,
                          cfg.hidden_dim, cfg.latent_dim])
        self.reward = mlp([cfg.latent_dim, cfg.hidden_dim, 1])

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        s_next = normalize_latent(self.trunk(torch.cat([s, a], dim=-1)))
        r = self.reward(s_next).squeeze(-1)
        return s_next, r


class PredictionNetwork(nn.Module):
    """f: s -> (정책 파라미터 [mean, log_std], 가치 V)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.trunk = mlp([cfg.latent_dim, cfg.hidden_dim])
        self.policy_mean = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.policy_logstd = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.value = mlp([cfg.hidden_dim, cfg.hidden_dim, 1])
        # TODO: 가치는 논문처럼 categorical support로 바꾸면 안정적. 스켈레톤은 scalar MSE.

    def forward(self, s: torch.Tensor):
        h = torch.relu(self.trunk(s))
        mean = self.policy_mean(h)
        log_std = self.policy_logstd(h).clamp(-5.0, 2.0)
        v = self.value(h).squeeze(-1)
        return (mean, log_std), v


def normalize_latent(s: torch.Tensor) -> torch.Tensor:
    """MuZero 안정화 트릭: 잠재를 [0,1] 범위로 min-max 정규화."""
    s_min = s.min(dim=-1, keepdim=True).values
    s_max = s.max(dim=-1, keepdim=True).values
    return (s - s_min) / (s_max - s_min + 1e-5)
