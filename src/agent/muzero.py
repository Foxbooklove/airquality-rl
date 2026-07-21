"""세 망을 묶은 최상위 모델 + MuZero 표준 추론 인터페이스."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .networks import (RepresentationNetwork, DynamicsNetwork,
                       PredictionNetwork)


class MuZeroNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.h = RepresentationNetwork(cfg)
        self.g = DynamicsNetwork(cfg)
        self.f = PredictionNetwork(cfg)

    def initial_inference(self, obs):
        """관측 -> (잠재 s0, 정책파라미터, 가치). 루트에서 1번."""
        s = self.h(obs)
        policy_params, v = self.f(s)
        return s, policy_params, v

    def recurrent_inference(self, s: torch.Tensor, a: torch.Tensor):
        """(잠재 s, 행동 a) -> (다음 잠재 s', 보상 r, 정책파라미터, 가치).
        트리를 한 칸 내려갈 때마다 호출."""
        s_next, r = self.g(s, a)
        policy_params, v = self.f(s_next)
        return s_next, r, policy_params, v
