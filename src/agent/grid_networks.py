"""격자 위 이산 행동용 신경망 — 공간 정보를 보존하는 h/g/f.

왜 새로 만드는가 (networks.py 의 실측 문제):
  기존 GridEncoder 는 마지막에 AdaptiveAvgPool2d(1) 로 16×16 을 1×1 로 뭉갠다.
  SetEncoder 도 평균 풀링이다. 그래서 **잠재상태가 '불확실성이 어디에 있는지'를
  모른다.** 측정:
    - 불확실성 덩어리를 좌하 -> 우상으로 옮겨도 잠재 거리 0.0015
    - 드론 위치만 바꾸면 0.79  (500배 차이)
    - 전체 크기만 바꾸면 0.020 (위치보다 13배 민감)
  그 결과 g 가 행동과 무관한 상수를 뱉었고(예측 σ=0.008 vs 실제 0.078, r=0.02),
  MCTS backup 이 후보를 구분하지 못해 방문수가 균등해지고 정책이 못 배웠다.

해법: 공간 축을 끝까지 들고 간다.
  잠재상태를 벡터가 아니라 **[C, G, G] 특징맵**으로 둔다 (AlphaZero 방식).
  정책은 1×1 합성곱으로 칸마다 로짓 하나를 뱉는다 — '어디가 좋은가'가
  공간적으로 표현되므로 전역 풀링이 필요 없다.

드론 위치는 브로드캐스트 평면으로 넣는다:
  현재 위치·기지 위치를 좌표 채널로 깔면, CNN 이 '내 기준 어느 쪽이 불확실한가'를
  국소 합성곱만으로 읽을 수 있다. 스칼라(x, y)를 MLP 로 섞는 것보다 직접적이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _coord_planes(G: int, device=None):
    """[2, G, G] 절대 좌표 평면 (0~1). CoordConv 의 위치 인식용."""
    ys = torch.linspace(0.0, 1.0, G, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    return torch.stack([xx, yy])


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(4, c)
        self.n2 = nn.GroupNorm(4, c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class GridRepresentation(nn.Module):
    """h: 관측 -> 잠재 특징맵 [B, C, G, G].

    입력 평면 (채널 순서):
      0,1   belief (Kriging mean, variance)
      2     측정점 밀도 — 각 칸에 관측점이 몇 개 있나 (DeepSets 평균풀링 대체)
      3     측정점 값 평균
      4     현재 드론 위치 (원-핫에 가까운 가우시안 범프)
      5     기지 위치
      6     남은 배터리 (상수 평면)
      7,8   절대 좌표 x, y
      9     **방문 최근성** — 이 칸을 얼마나 최근에 쟀나 (exp 감쇠)
      10    **마지막 측정값** — 거기서 뭘 읽었나
      11    **직전 위치** — 두 칸 왕복을 스스로 알아채게
      12,13 **시각** (하루 주기 sin/cos, 상수 평면)
      14    **에피소드 경과** (상수 평면)

    9~14 를 넣는 이유:
      보상이 (측정소, 시각) 단위라 같은 지점을 다시 재는 게 유효한 전략이다
      (값이 오를 때 잡는다). 그런데 언제 무엇을 쟀는지, 지금 몇 시인지를 모르면
      그 판단이 불가능하다 — 문제가 마르코프가 아니게 된다. 실제로 이 평면들이
      없을 때 학습된 정책은 두 칸을 9시간 왕복하며 탐지 0 을 기록했다.
      MuZero 원논문도 표현망에 최근 관측 + **최근 행동**을 함께 넣는다.
    """

    N_PLANES = 15

    def __init__(self, cfg: ModelConfig, channels: int = 48, blocks: int = 3):
        super().__init__()
        self.cfg = cfg
        self.G = cfg.grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(self.N_PLANES, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.out_channels = channels

    def build_planes(self, obs) -> torch.Tensor:
        G = self.G
        dev = obs.belief_grid.device
        B = obs.belief_grid.shape[0]
        planes = [obs.belief_grid]                                    # [B,2,G,G]

        # --- 측정점을 격자에 뿌린다 (밀도 + 값 평균) ---
        pts, m = obs.points, obs.mask                                 # [B,N,3], [B,N]
        gx = (pts[..., 0].clamp(0, 1) * (G - 1)).round().long()
        gy = (pts[..., 1].clamp(0, 1) * (G - 1)).round().long()
        idx = (gy * G + gx)                                           # [B,N]
        dens = torch.zeros(B, G * G, device=dev)
        vals = torch.zeros(B, G * G, device=dev)
        dens.scatter_add_(1, idx, m)
        vals.scatter_add_(1, idx, pts[..., 2] * m)
        vals = vals / dens.clamp(min=1.0)
        planes.append((dens / dens.amax(dim=1, keepdim=True).clamp(min=1.0)
                       ).view(B, 1, G, G))
        planes.append(vals.view(B, 1, G, G))

        # --- 드론/기지 위치, 배터리 ---
        drone = obs.drone                                             # [B,3]
        coord = _coord_planes(G, dev)[None]                           # [1,2,G,G]
        def bump(xy):
            dx = coord[:, 0] - xy[:, 0, None, None]
            dy = coord[:, 1] - xy[:, 1, None, None]
            return torch.exp(-(dx * dx + dy * dy) * (G * G) / 8.0)[:, None]
        planes.append(bump(drone[:, :2]))
        home = getattr(obs, "home", None)
        planes.append(bump(home[:, :2]) if home is not None else torch.zeros_like(planes[-1]))
        planes.append(drone[:, 2, None, None, None].expand(B, 1, G, G))
        planes.append(coord.expand(B, 2, G, G))

        # --- 기억: 방문 최근성 + 마지막 측정값 (환경이 격자로 만들어 준다) ---
        mem = getattr(obs, "memory", None)
        planes.append(mem if mem is not None
                      else torch.zeros(B, 2, G, G, device=dev))
        # --- 직전 위치 ---
        prev = getattr(obs, "prev", None)
        planes.append(bump(prev[:, :2]) if prev is not None
                      else torch.zeros(B, 1, G, G, device=dev))
        # --- 시각 (하루 주기 2 + 경과 1) ---
        tt = getattr(obs, "time", None)
        if tt is None:
            planes.append(torch.zeros(B, 3, G, G, device=dev))
        else:
            planes.append(tt[:, :, None, None].expand(B, 3, G, G))
        return torch.cat(planes, dim=1)

    def forward(self, obs) -> torch.Tensor:
        return self.blocks(self.stem(self.build_planes(obs)))


class GridDynamics(nn.Module):
    """g: (잠재맵, 행동칸) -> (다음 잠재맵, 보상).

    행동을 **원-핫 평면**으로 넣는다. 스칼라 2차원을 64차원 잠재에 이어붙이면
    묻히지만, 평면으로 넣으면 '어느 칸을 골랐나'가 공간적으로 정렬돼 들어간다.
    """

    def __init__(self, cfg: ModelConfig, channels: int = 48, blocks: int = 2):
        super().__init__()
        self.G = cfg.grid_size
        self.merge = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.reward = nn.Sequential(
            nn.Conv2d(channels, 8, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * self.G * self.G, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s: torch.Tensor, action_idx: torch.Tensor):
        B = s.shape[0]
        a = torch.zeros(B, 1, self.G * self.G, device=s.device)
        a.scatter_(2, action_idx.view(B, 1, 1), 1.0)
        h = self.blocks(self.merge(torch.cat([s, a.view(B, 1, self.G, self.G)], 1)))
        return h, self.reward(h).squeeze(-1)


class GridPrediction(nn.Module):
    """f: 잠재맵 -> (칸별 정책 로짓 [B,G*G], 가치 [B]).

    정책이 1×1 합성곱이라 **칸마다 로짓이 하나씩** 나온다. 전역 풀링이 없으므로
    '어디가 좋은가'가 공간적으로 보존된다. 마스킹은 로짓에 -inf 를 더하면 끝이다.
    """

    def __init__(self, cfg: ModelConfig, channels: int = 48):
        super().__init__()
        self.G = cfg.grid_size
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 16, 1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 1, 1))
        self.value = nn.Sequential(
            nn.Conv2d(channels, 8, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * self.G * self.G, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s: torch.Tensor):
        B = s.shape[0]
        return self.policy(s).view(B, -1), self.value(s).squeeze(-1)


class GridMuZeroNet(nn.Module):
    """세 망을 묶은 최상위 모델. 인터페이스는 MuZeroNet 과 같다."""

    def __init__(self, cfg: ModelConfig, channels: int = 48):
        super().__init__()
        self.cfg = cfg
        self.h = GridRepresentation(cfg, channels)
        self.g = GridDynamics(cfg, channels)
        self.f = GridPrediction(cfg, channels)

    def initial_inference(self, obs):
        s = self.h(obs)
        logits, v = self.f(s)
        return s, logits, v

    def recurrent_inference(self, s, action_idx):
        s_next, r = self.g(s, action_idx)
        logits, v = self.f(s_next)
        return s_next, r, logits, v


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor):
    """유효 칸만 남기고 정규화. mask: [B, G*G] bool.

    연속 가우시안에서는 로짓 마스킹이 불가해 기각 샘플링을 썼지만, 카테고리컬로
    바꾸면 이렇게 한 줄로 끝난다 — 원래 의도했던 방식이다.
    """
    neg = torch.finfo(logits.dtype).min
    return torch.log_softmax(logits.masked_fill(~mask, neg), dim=-1)
