"""격자 위 이산 행동 MCTS — 원본 MuZero 방식 (AlphaZero 계열).

기존 mcts.py 와의 차이:
  행동      연속 (theta, dist) K개 샘플  ->  격자 칸 카테고리컬 전체
  마스킹    기각 샘플링                  ->  **로짓에 -inf** (한 줄)
  정책타깃  IS 보정 pi_hat               ->  **방문수 그대로**
  prior     균등 1/K                     ->  신경망 정책 (마스킹 후 softmax)

왜 단순해지는가:
  Sampled MuZero 의 복잡함(tanh squashing, 야코비안 보정, 제안분포 log-prob,
  IS 보정)은 전부 '연속 행동을 K개만 뽑아 쓴다'는 제약에서 나왔다. 행동이
  이산이면 전부 필요 없다. 방문수가 곧 정책 타깃이고, 마스킹은 로짓에 -inf 다.

비용:
  시뮬레이션은 잠재공간에서만 돈다 — 환경(Kriging)을 건드리지 않는다.
  실측: recurrent_inference 0.84ms(배치1) / 0.23ms(배치64), PUCT 선택은
  자식 256개에도 0.05ms. sims=1000 이 수 초 수준이라 수천도 가능하다.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "reward", "latent",
                 "children", "mask", "expanded_")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent = None
        self.children: dict[int, "Node"] = {}
        self.mask = None
        self.expanded_ = False

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class GridMCTS:
    """mask 는 [G*G] bool. 루트는 환경이 주고, 깊은 노드는 루트 마스크를 물려받는다.

    깊은 노드에서 마스크를 정확히 계산하려면 배터리·위치를 잠재상태 밖에서
    추적해야 하는데, 격자 행동에서는 그게 환경 재구현이 된다. 대신 루트 마스크를
    상속한다 — 수읽기가 약간 낙관적이 되지만(갈 수 없는 칸을 상상할 수 있다)
    실제로 두는 수는 항상 루트 마스크로 걸러진다.
    """

    def __init__(self, net, cfg, sims: int = 200, c_puct_base: float = 19652.0,
                 c_puct_init: float = 1.25, discount: float = 0.997,
                 dirichlet_alpha: float = 0.3, root_noise: float = 0.25):
        self.net, self.cfg = net, cfg
        self.sims = sims
        self.pb_c_base, self.pb_c_init = c_puct_base, c_puct_init
        self.discount = discount
        self.alpha, self.root_noise = dirichlet_alpha, root_noise
        self.last_stats: dict = {}

    @torch.no_grad()
    def run(self, obs, mask: np.ndarray, add_noise: bool = True):
        """-> (방문수 [G*G] float, 루트 가치). 방문수가 곧 정책 타깃이다."""
        m = torch.as_tensor(mask.reshape(-1), dtype=torch.bool)
        if not bool(m.any()):
            raise ValueError("유효한 행동이 하나도 없습니다")

        s, logits, v = self.net.initial_inference(obs)
        if logits.numel() != m.numel():
            raise ValueError(
                f"정책 출력 {logits.numel()}칸 != 행동 마스크 {m.numel()}칸. "
                f"cfg.grid_size 와 환경 격자 해상도가 같아야 합니다.")
        root = Node()
        root.latent, root.mask = s, m
        self._expand(root, logits, m)
        if add_noise:
            self._add_dirichlet(root)

        max_depth = 0
        for _ in range(self.sims):
            node, path, actions = root, [root], []
            while node.expanded_:
                a = self._select(node)
                actions.append(a)
                node = node.children[a]
                path.append(node)
            parent = path[-2]
            a = torch.tensor([actions[-1]])
            s_next, r, logits, v_leaf = self.net.recurrent_inference(parent.latent, a)
            node.latent = s_next
            node.reward = float(r.item())
            self._expand(node, logits, parent.mask)      # 루트 마스크 상속
            value = float(v_leaf.item())
            max_depth = max(max_depth, len(path) - 1)
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = n.reward + self.discount * value

        counts = np.zeros(m.numel(), dtype=np.float64)
        for a, ch in root.children.items():
            counts[a] = ch.visit_count
        self.last_stats = {
            "n_valid": int(m.sum()), "max_depth": max_depth,
            "root_value": root.value(),
            "top_frac": float(counts.max() / max(counts.sum(), 1)),
        }
        return counts, root.value()

    # ---------------- 내부 ----------------
    def _expand(self, node: Node, logits: torch.Tensor, mask: torch.Tensor):
        """마스킹 후 softmax 를 prior 로. 유효 칸만 자식으로 만든다."""
        neg = torch.finfo(logits.dtype).min
        p = torch.softmax(logits.view(-1).masked_fill(~mask, neg), dim=-1)
        idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        node.children = {int(a): Node(float(p[a])) for a in idx}
        node.mask = mask
        node.expanded_ = len(node.children) > 0

    def _add_dirichlet(self, root: Node):
        """루트 탐색 노이즈 — 정책이 한 칸으로 굳는 것을 막는다 (AlphaZero)."""
        ks = list(root.children)
        if len(ks) < 2:
            return
        noise = np.random.dirichlet([self.alpha] * len(ks))
        for k, nz in zip(ks, noise):
            c = root.children[k]
            c.prior = (1 - self.root_noise) * c.prior + self.root_noise * float(nz)

    def _select(self, node: Node) -> int:
        """PUCT. 자식이 수백 개라 numpy 로 한 번에 계산한다."""
        ks = np.fromiter(node.children.keys(), dtype=np.int64)
        ch = [node.children[int(k)] for k in ks]
        vis = np.fromiter((c.visit_count for c in ch), float, len(ch))
        val = np.fromiter((c.value() for c in ch), float, len(ch))
        pri = np.fromiter((c.prior for c in ch), float, len(ch))
        pb_c = math.log((node.visit_count + self.pb_c_base + 1) / self.pb_c_base) \
            + self.pb_c_init
        u = pb_c * pri * math.sqrt(max(node.visit_count, 1)) / (1.0 + vis)
        return int(ks[int(np.argmax(val + u))])


def policy_target(counts: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """방문수 -> 정책 타깃. IS 보정이 필요 없다 (전수 확장이므로).

    temperature=0 이면 argmax(탐욕), 1이면 방문수 비례.
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.sum() <= 0:
        return c
    if temperature <= 1e-6:
        out = np.zeros_like(c)
        out[int(c.argmax())] = 1.0
        return out
    c = c ** (1.0 / temperature)
    return c / c.sum()
