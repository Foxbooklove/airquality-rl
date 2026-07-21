"""교수님 설득용 figure 생성.

  1) 불확실성 before/after + 드론 궤적  (핵심 슬라이드)
  2) 학습 곡선 (에피소드별 총 정보이득)
  3) 학습 정책 vs 랜덤 baseline 비교

제목은 영어로 (matplotlib 한글 폰트 깨짐 방지). 필요하면 rcParams 로 교체.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mask_nan(grid, land):
    out = np.array(grid, dtype=float)
    out[~land] = np.nan
    return out


def uncertainty_before_after(var0, var1, traj, land, out_path: Path,
                             extent=None):
    """왼쪽=초기 불확실성, 오른쪽=관측 후 불확실성, 위에 드론 궤적."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = np.nanmax(_mask_nan(var0, land))
    xs = [p[0] for p in traj]; ys = [p[1] for p in traj]

    for ax, v, title in [(axes[0], var0, "Uncertainty — before (initial belief)"),
                         (axes[1], var1, "Uncertainty — after drone survey")]:
        im = ax.imshow(_mask_nan(v, land), origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax, extent=extent, aspect="auto")
        ax.plot(xs, ys, "-o", c="white", ms=4, lw=1.5, label="drone path")
        ax.scatter(xs[0], ys[0], c="red", s=60, zorder=5, label="start")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def learning_curve(info_gains, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(info_gains, lw=1.5)
    ax.set_xlabel("episode"); ax.set_ylabel("total info gain (frac. of uncertainty removed)")
    ax.set_title("Training: uncertainty reduction per episode")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def baseline_compare(learned_gain, random_gain, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["random", "learned (MuZero)"], [random_gain, learned_gain],
           color=["#B4B2A9", "#1D9E75"])
    ax.set_ylabel("total info gain (one episode)")
    ax.set_title("Learned policy vs random baseline")
    for i, v in enumerate([random_gain, learned_gain]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
