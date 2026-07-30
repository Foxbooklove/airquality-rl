"""Sampled MuZero MCTS (최소 골격).

노드마다 정책 beta 에서 K개 행동만 샘플해 자식으로 두고, PUCT로 선택,
g로 잠재공간에서 한 칸 내려가며 시뮬레이션. 리프는 f의 가치로 평가.

원본 MuZero와 다른 점은 딱: '모든 행동' 대신 '샘플된 K개'만 확장.
PUCT prior 는 샘플 기반이라 균등(1/K)을 씀.

--- 노드에 지리 좌표를 태우는 이유 ---
노드는 잠재상태라 원래 위경도가 없다. 그런데 **드론 운동학은 우리가 정확히 안다**
(모르는 건 농도장이다). 그래서 부모 위치 + 행동 벡터로 자식 위치를 정확히 계산할 수
있고, 그래야 '바다 위로 날지 말 것' 같은 물리 제약을 수읽기 단계에서 걸 수 있다.
규칙은 env.step 과 공유한다 (src/env/motion.py).

배터리도 같이 태운다. 이동거리가 남은 배터리에 비례하므로
(motion.reach_frac), 깊이 d 노드는 배터리가 d 만큼 줄어 있어야 한다. 이전에는
모든 깊이에 루트 배터리를 넘겨서 수읽기가 '갈 수 없는 곳'을 계획에 넣었다.
"""
from __future__ import annotations

import math
import torch

from .config import ModelConfig
from .sampling import ContinuousActionSampler


class Node:
    def __init__(self, prior: float, pos=None, battery: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent = None                 # 확장될 때 g가 채움
        self.actions = None                # [K, action_dim] 이 노드에서 뽑은 행동
        self.beta_logprob = None           # [K] 제안분포 log-prob
        self.children: list["Node"] = []   # actions[i] 로 가는 자식
        self.pos = pos                     # (lon, lat) — 마스킹용. None 이면 미사용
        self.battery = battery             # 이 노드에서의 남은 배터리
        self.fallback = False              # 후보가 전부 막혀 폴백을 썼는가

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0


class SampledMuZeroMCTS:
    """mask 를 주면 바다 비행·경계 이탈을 수읽기 단계에서 차단한다.

    mask     : src.env.motion.FlightMask (None 이면 마스킹 없음 = 기존 동작)
    extent   : (min_lon, max_lon, min_lat, max_lat)
    stations : (lon[], lat[]) — 폴백 방향용. 없으면 제자리 대기만 폴백.
    """

    def __init__(self, net, cfg: ModelConfig, mask=None, extent=None,
                 stations=None):
        self.net = net
        self.cfg = cfg
        self.sampler = ContinuousActionSampler(cfg)
        self.mask = mask
        self.extent = extent
        self.stations = stations
        if mask is not None and extent is None:
            raise ValueError("mask 를 주면 extent 도 필요합니다")
        self.last_stats: dict = {}

    @torch.no_grad()
    def run(self, obs, battery: float, return_root: bool = False,
            on_simulation=None, pos=None):
        """루트 관측에서 탐색 -> (샘플 행동들, 방문수, beta_logprob, 루트가치).
        방문수/beta 로 sampling.is_corrected_policy_target 을 부르면 학습 타깃.
        return_root=True 면 루트 Node 도 함께 반환(트리 시각화/디버깅용).

        pos: 드론의 현재 (lon, lat). mask 를 쓸 때는 필수.

        on_simulation: 시뮬레이션 1회가 끝날 때마다 호출되는 콜백(실시간 시각화용).
          info = {sim, nodes, action_idx, leaf_value, leaf_reward}
          - nodes:      루트부터 리프까지의 Node 리스트
          - action_idx: 각 단계에서 고른 자식 인덱스 (nodes 보다 1개 짧음)
          이걸로 '뻗어나갔다 되돌아오는' 수읽기 과정을 그릴 수 있음.

        탐색이 끝나면 self.last_stats 에 기각률·폴백 여부가 담긴다.
        루트가 폴백을 썼으면 그 스텝은 정책 손실에서 빼야 한다 (sampling.sample 주석).
        """
        if self.mask is not None and pos is None:
            raise ValueError("mask 를 쓰면 run(pos=...) 로 드론 위치를 줘야 합니다")

        self._stats = {"draws": 0, "rejected": 0, "fallback_nodes": 0,
                       "expansions": 0}
        s0, policy_params, _ = self.net.initial_inference(obs)
        root = Node(prior=1.0, pos=pos, battery=float(battery))
        root.latent = s0
        self._expand(root, policy_params, explore=True)

        for sim in range(self.cfg.num_simulations):
            node = root
            path = [root]
            action_idx = []
            while node.expanded():
                idx = self._select_child(node)
                action_idx.append(idx)
                node = node.children[idx]
                path.append(node)

            # node = 리프. 부모 잠재 + 그 행동으로 g 한 칸 내려감.
            parent = path[-2]
            a = parent.actions[action_idx[-1]].unsqueeze(0)
            s_next, r, pol, v = self.net.recurrent_inference(parent.latent, a)
            node.latent = s_next
            node.reward = float(r.item())
            self._expand(node, pol)
            value = float(v.item())

            if on_simulation is not None:
                on_simulation({"sim": sim, "nodes": list(path),
                               "action_idx": list(action_idx),
                               "leaf_value": value, "leaf_reward": node.reward})

            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = n.reward + self.cfg.discount * value

        st = self._stats
        self.last_stats = dict(
            st, root_fallback=root.fallback,
            reject_rate=(st["rejected"] / st["draws"]) if st["draws"] else 0.0)

        visit_counts = torch.tensor(
            [c.visit_count for c in root.children], dtype=torch.float32)
        if return_root:
            return root.actions, visit_counts, root.beta_logprob, root.value(), root
        return root.actions, visit_counts, root.beta_logprob, root.value()

    # ---------------- 확장 ----------------
    def _expand(self, node: Node, policy_params, explore: bool = False):
        validate = fallback = None
        if self.mask is not None and node.pos is not None:
            validate = self._validator(node)
            fallback = lambda: self._fallback_action(node)   # noqa: E731

        actions, beta_logprob, info = self.sampler.sample(
            _unbatch(policy_params), node.battery,
            explore_std=self.cfg.root_explore_std if explore else 0.0,
            validate=validate, fallback=fallback)

        node.actions = actions
        node.beta_logprob = beta_logprob
        node.fallback = bool(info["fallback"])
        self._stats["draws"] += info["draws"]
        self._stats["rejected"] += info["rejected"]
        self._stats["expansions"] += 1
        self._stats["fallback_nodes"] += int(info["fallback"])

        # 자식의 위치·배터리를 지금 계산해 넣는다 (그 자식을 확장할 때 필요)
        nb = max(node.battery - 1.0, 0.0)
        prior = 1.0 / len(actions)
        if self.mask is not None and node.pos is not None:
            lon1, lat1 = self._next_pos(node, actions)
            node.children = [Node(prior, pos=(float(lon1[i]), float(lat1[i])),
                                  battery=nb) for i in range(len(actions))]
        else:
            node.children = [Node(prior, battery=nb) for _ in actions]

    # ---------------- 기하 (motion.py 규칙 공유) ----------------
    def _next_pos(self, node: Node, actions):
        from src.env.motion import step_position
        return step_position(node.pos[0], node.pos[1],
                             actions.detach().cpu().numpy(), node.battery,
                             self.cfg.max_battery, self.cfg.max_step_frac,
                             self.extent, clip=False)

    def _validator(self, node: Node):
        """행동 후보 [N,2] → 유효 마스크 [N]. 경계 이탈 + 바다 경로를 거른다."""
        def _v(actions):
            lon1, lat1 = self._next_pos(node, actions)
            return self.mask.valid_step(node.pos[0], node.pos[1], lon1, lat1)
        return _v

    def _fallback_action(self, node: Node):
        """2단 폴백: ① 가장 가까운 측정소 방향으로 (경로가 육지면)
                     ② 제자리 대기 — 현재 위치는 항상 육지라 무조건 유효."""
        from src.env import motion
        if self.stations is not None:
            slon, slat = self.stations
            d2 = (slon - node.pos[0]) ** 2 + (slat - node.pos[1]) ** 2
            k = int(d2.argmin())
            for frac in (1.0, 0.5, 0.25):
                a = motion.action_toward(
                    node.pos[0], node.pos[1], slon[k], slat[k], node.battery,
                    self.cfg.max_battery, self.cfg.max_step_frac, self.extent,
                    frac=frac)
                lon1, lat1 = motion.step_position(
                    node.pos[0], node.pos[1], a, node.battery,
                    self.cfg.max_battery, self.cfg.max_step_frac, self.extent)
                if bool(self.mask.valid_step(node.pos[0], node.pos[1],
                                             lon1, lat1)[0]):
                    return a
        return motion.STAY

    def _select_child(self, node: Node) -> int:
        best, best_idx = -1e9, 0
        for i, child in enumerate(node.children):
            pb_c = (math.log((node.visit_count + self.cfg.pb_c_base + 1)
                             / self.cfg.pb_c_base) + self.cfg.pb_c_init)
            u = pb_c * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value() + u
            if score > best:
                best, best_idx = score, i
        return best_idx


def _unbatch(policy_params):
    mean, log_std = policy_params
    return (mean.squeeze(0), log_std.squeeze(0))
