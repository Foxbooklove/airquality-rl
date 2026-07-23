"""Replay buffer — 수집한 궤적을 재사용.

왜 필요한가:
  Kriging 때문에 데이터 수집이 매우 비쌈(에피소드당 수 초~수십 초).
  그런데 예전 구조는 궤적 1개를 딱 1번 학습에 쓰고 버렸음 -> gradient step
  총 개수가 에피소드 수와 같아짐(300 에피소드 = 300 업데이트).
  MuZero 는 보통 수만~수백만 업데이트가 필요.

  게다가 항상 trajectory[0] 에서만 unroll 해서, TTL=40 궤적 중
  앞 3스텝만 학습에 쓰이고 92% 는 버려졌음.

이 버퍼는 두 문제를 같이 해결:
  - 과거 궤적을 보관해 여러 번 재사용
  - 궤적 안의 '무작위 시작 지점'에서 unroll 윈도우를 뽑아 전 구간 학습
"""
from __future__ import annotations

import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 200, seed: int = 0):
        self.buf: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, trajectory: list[dict]):
        self.buf.append(trajectory)

    def __len__(self):
        return len(self.buf)

    def sample(self, unroll_steps: int):
        """(궤적, 시작 인덱스) 하나를 무작위로.
        시작 인덱스는 unroll 이 가능한 범위에서만 고름."""
        traj = self.rng.choice(self.buf)
        # 마지막 원소는 종료 상태(sampled_actions=None)라 시작점이 될 수 없음
        last_valid = len(traj) - 2
        if last_valid < 0:
            return traj, 0
        t0 = self.rng.randint(0, last_valid)
        return traj, t0

    def sample_batch(self, batch_size: int, unroll_steps: int):
        return [self.sample(unroll_steps) for _ in range(batch_size)]
