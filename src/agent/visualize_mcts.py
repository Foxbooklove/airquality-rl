"""MCTS 탐색 트리 시각화 (디버깅용).

바둑 해설에서 후보수마다 방문수/승률 띄우는 것과 같은 개념:
  plot_root_candidates() — 불확실성 지도 위에, 드론 현재 위치에서 K개 후보
                           행동이 어디로 가는지 화살표. 굵기=방문수, 색=Q값.
  plot_tree()            — 트리 구조 자체(루트->자식->손자). 원 크기=방문수.

이걸로 보이는 것:
  - 화살표가 한쪽으로만 뻗어 있으면  -> 정책 붕괴(한 방향으로 굳음)
  - 방문수가 전부 균등하면            -> PUCT가 후보를 구분 못 함(가치 학습 안 됨)
  - 화살표가 불확실성 낮은 쪽으로 가면 -> 가치 추정이 틀림
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def action_to_lonlat(action, lon, lat, max_step_frac, span):
    """(theta_norm, dist_norm) -> 도착 (lon, lat).  env.step 과 동일 규칙."""
    theta = float(action[0]) * np.pi
    dist = float(action[1]) * max_step_frac * span
    return lon + dist * np.cos(theta), lat + dist * np.sin(theta)


def plot_root_candidates(root, drone_lonlat, var, land, extent,
                         max_step_frac, out_path: Path, title_extra: str = ""):
    """불확실성 배경 + 루트에서 뻗는 K개 후보 행동.

    root        : mcts.run(..., return_root=True) 가 준 루트 Node
    drone_lonlat: (lon, lat) 현재 드론 위치
    extent      : [min_lon, max_lon, min_lat, max_lat]
    """
    lon, lat = drone_lonlat
    min_lon, max_lon, min_lat, max_lat = extent
    span = float(np.hypot(max_lon - min_lon, max_lat - min_lat))

    bg = np.array(var, dtype=float)
    bg[~land] = np.nan

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(bg, origin="lower", cmap="viridis", extent=extent,
                   aspect="auto", alpha=0.85)
    fig.colorbar(im, ax=ax, fraction=0.046, label="uncertainty (Kriging variance)")

    visits = np.array([c.visit_count for c in root.children], dtype=float)
    qvals = np.array([c.value() for c in root.children], dtype=float)
    vmaxv = max(visits.max(), 1.0)
    qlo, qhi = float(qvals.min()), float(qvals.max())
    qrange = max(qhi - qlo, 1e-9)

    for i, (a, n, q) in enumerate(zip(root.actions, visits, qvals)):
        tx, ty = action_to_lonlat(a, lon, lat, max_step_frac, span)
        w = 0.8 + 5.0 * (n / vmaxv)                 # 굵기 = 방문수
        c = plt.cm.autumn(1.0 - (q - qlo) / qrange)  # 색 = Q값(높을수록 진함)
        ax.annotate("", xy=(tx, ty), xytext=(lon, lat),
                    arrowprops=dict(arrowstyle="-|>", lw=w, color=c, alpha=0.9))
        ax.text(tx, ty, f"n={int(n)}\nQ={q:.3f}", fontsize=7,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, lw=0))

    ax.plot(lon, lat, "o", ms=11, mfc="red", mec="white", mew=1.5, zorder=6)
    ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
    ax.set_title("MCTS root candidates — arrow width = visits, label = visits/Q"
                 + (f"\n{title_extra}" if title_extra else ""))
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_tree(root, out_path: Path, max_depth: int = 2, top_k: int | None = None):
    """트리 구조 자체. 원 크기=방문수, 세로=깊이.
    top_k 주면 각 노드에서 방문수 상위 k개만 (가지가 너무 많을 때)."""
    fig, ax = plt.subplots(figsize=(13, 6))

    def layout(node, depth, x0, x1, parent_xy=None):
        x = (x0 + x1) / 2
        y = -depth
        if parent_xy is not None:
            ax.plot([parent_xy[0], x], [parent_xy[1], y], "-",
                    color="#B4B2A9", lw=0.8, zorder=1)
        size = 40 + 260 * (node.visit_count / max(root.visit_count, 1))
        ax.scatter([x], [y], s=size, color="#1D9E75" if depth else "#C7663F",
                   zorder=3, edgecolors="white", linewidths=0.8)
        if depth <= 1:
            ax.text(x, y - 0.18, f"n={node.visit_count}\nQ={node.value():.3f}",
                    fontsize=6.5, ha="center", va="top")
        if depth >= max_depth or not node.children:
            return
        kids = list(enumerate(node.children))
        if top_k:
            kids = sorted(kids, key=lambda t: -t[1].visit_count)[:top_k]
        if not kids:
            return
        w = (x1 - x0) / len(kids)
        for j, (_, child) in enumerate(kids):
            layout(child, depth + 1, x0 + j * w, x0 + (j + 1) * w, (x, y))

    layout(root, 0, 0.0, 1.0)
    ax.set_axis_off()
    ax.set_title(f"MCTS tree (depth ≤ {max_depth}) — circle size = visit count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summarize_root(root) -> str:
    """텍스트 요약 — 로그로 빠르게 보기용."""
    visits = np.array([c.visit_count for c in root.children], dtype=float)
    qvals = np.array([c.value() for c in root.children])
    thetas = np.array([float(a[0]) * 180 for a in root.actions])   # deg
    dists = np.array([float(a[1]) for a in root.actions])
    lines = ["idx  visits      Q    theta(deg)  dist"]
    order = np.argsort(-visits)
    for i in order:
        lines.append(f"{i:3d}  {int(visits[i]):6d}  {qvals[i]:7.4f}  "
                     f"{thetas[i]:9.1f}  {dists[i]:.3f}")
    spread = float(visits.std() / (visits.mean() + 1e-9))
    lines.append(f"방문수 변동계수={spread:.3f} "
                 f"(0에 가까우면 후보 구분 못 하는 중)")
    return "\n".join(lines)
