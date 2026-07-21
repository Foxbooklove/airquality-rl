"""프로토타입용 더미 드론 환경.

목적: RL 환경 클래스(3-way 분할 + Kriging)가 아직 없으니, 그 '인터페이스'만
흉내내서 모델 스켈레톤을 end-to-end로 돌려보기 위함.
진짜 환경이 준비되면 이 파일을 그대로 대체하면 됨 (reset/step 시그니처 동일).

핵심: reward 는 RewardFn 슬롯으로 분리. 지금은 InfoGainReward(더미)만 꽂음.
나중에 대기환경 공부하고 ReconstructionReward 등으로 교체 -> 환경/모델 안 건드림.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.agent.config import ModelConfig


@dataclass
class Observation:
    points: torch.Tensor       # [1, N, point_feat_dim] 측정점 (위치+값)
    mask: torch.Tensor         # [1, N] 유효 마스크
    belief_grid: torch.Tensor  # [1, C, G, G] Kriging (mean, variance)
    drone: torch.Tensor        # [1, drone_feat_dim] (x, y, battery)


class RewardFn(ABC):
    """학습 목표 함수의 핵심 조각 — 슬롯."""
    @abstractmethod
    def __call__(self, env: "DummyDroneEnv", action: torch.Tensor) -> float: ...


class InfoGainReward(RewardFn):
    """더미 정보이득: '분산 그리드에서 방문 지점 주변 불확실성 감소량'을 흉내.
    Kriging 분산 감소는 측정값과 무관(위치만 의존)하다는 성질을 반영해,
    실제 값 없이 위치만으로 보상을 계산.
    """
    def __call__(self, env: "DummyDroneEnv", action: torch.Tensor) -> float:
        x, y = env.drone_xy
        # 방문 셀 주변 분산 총합을 정보이득으로 보고, 감소분을 보상으로.
        before = float(env.variance.sum())
        gx = min(env.G - 1, max(0, int(x * env.G)))
        gy = min(env.G - 1, max(0, int(y * env.G)))
        env.variance[gy, gx] *= 0.3            # 측정 -> 그 셀 불확실성 급감
        after = float(env.variance.sum())
        return before - after


class DummyDroneEnv:
    def __init__(self, cfg: ModelConfig, reward_fn: RewardFn | None = None):
        self.cfg = cfg
        self.G = cfg.grid_size
        self.reward_fn = reward_fn or InfoGainReward()

    def reset(self) -> Observation:
        self.battery = self.cfg.max_battery
        self.drone_xy = (0.5, 0.5)             # 캔버스 중앙에서 시작
        self.mean = torch.rand(self.G, self.G)         # 더미 Kriging 농도장
        self.variance = torch.rand(self.G, self.G) + 0.5  # 더미 불확실성
        self.visited = [(0.5, 0.5, float(self.mean[self.G // 2, self.G // 2]))]
        return self._obs()

    def step(self, action: torch.Tensor):
        theta = float(action[0]) * math.pi     # [-1,1] -> [-pi,pi]
        dist = float(action[1]) * self.cfg.max_step_frac
        x, y = self.drone_xy
        x = min(1.0, max(0.0, x + dist * math.cos(theta)))
        y = min(1.0, max(0.0, y + dist * math.sin(theta)))
        self.drone_xy = (x, y)
        self.battery -= 1.0

        reward = self.reward_fn(self, action)
        self.visited.append((x, y, float(self.mean[
            min(self.G - 1, int(y * self.G)), min(self.G - 1, int(x * self.G))])))
        done = self.battery <= 0
        return self._obs(), reward, done

    def _obs(self) -> Observation:
        pts = torch.tensor(self.visited, dtype=torch.float32)      # [N, 3]
        grid = torch.stack([self.mean, self.variance], dim=0)      # [C, G, G]
        drone = torch.tensor([*self.drone_xy,
                              self.battery / self.cfg.max_battery], dtype=torch.float32)
        return Observation(
            points=pts.unsqueeze(0),
            mask=torch.ones(1, pts.shape[0]),
            belief_grid=grid.unsqueeze(0),
            drone=drone.unsqueeze(0),
        )
