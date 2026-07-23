"""Sampled MuZero MCTS (최소 골격).

노드마다 정책 beta 에서 K개 행동만 샘플해 자식으로 두고, PUCT로 선택,
g로 잠재공간에서 한 칸 내려가며 시뮬레이션. 리프는 f의 가치로 평가.

원본 MuZero와 다른 점은 딱: '모든 행동' 대신 '샘플된 K개'만 확장.
PUCT prior 는 샘플 기반이라 균등(1/K)을 씀.
"""
from __future__ import annotations

import math
import torch

from .config import ModelConfig
from .sampling import ContinuousActionSampler


class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent = None                 # 확장될 때 g가 채움
        self.actions = None                # [K, action_dim] 이 노드에서 뽑은 행동
        self.beta_logprob = None           # [K] 제안분포 log-prob
        self.children: list["Node"] = []   # actions[i] 로 가는 자식

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0


class SampledMuZeroMCTS:
    def __init__(self, net, cfg: ModelConfig):
        self.net = net
        self.cfg = cfg
        self.sampler = ContinuousActionSampler(cfg)

    @torch.no_grad()
    def run(self, obs, battery: float, return_root: bool = False):
        """루트 관측에서 탐색 -> (샘플 행동들, 방문수, beta_logprob, 루트가치).
        방문수/beta 로 sampling.is_corrected_policy_target 을 부르면 학습 타깃.
        return_root=True 면 루트 Node 도 함께 반환(트리 시각화/디버깅용)."""
        s0, policy_params, _ = self.net.initial_inference(obs)
        root = Node(prior=1.0)
        root.latent = s0
        self._expand(root, policy_params, battery, explore=True)

        for _ in range(self.cfg.num_simulations):
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
            self._expand(node, pol, battery)
            value = float(v.item())

            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = n.reward + self.cfg.discount * value

        visit_counts = torch.tensor(
            [c.visit_count for c in root.children], dtype=torch.float32)
        if return_root:
            return root.actions, visit_counts, root.beta_logprob, root.value(), root
        return root.actions, visit_counts, root.beta_logprob, root.value()

    def _expand(self, node: Node, policy_params, battery: float,
                explore: bool = False):
        actions, beta_logprob = self.sampler.sample(
            _unbatch(policy_params), battery,
            explore_std=self.cfg.root_explore_std if explore else 0.0)
        node.actions = actions
        node.beta_logprob = beta_logprob
        node.children = [Node(prior=1.0 / len(actions)) for _ in actions]

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
