"""Canvas + 스냅샷 + Kriging → 마스킹된 연속 농도장."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data.preprocess import AirQualityPreprocessor
from env.canvas import Canvas

REGION = "서울특별시"      # None이면 전국. reference 보고 여기만 바꾸면 됨
DT = "2024-07-01 13:00"
POLLUTANT = "PM10"

# 1. 캔버스
canvas = Canvas(region=REGION, resolution=100)
canvas.summary()

# 2. 스냅샷 → 캔버스 범위 안 측정소만
pre = AirQualityPreprocessor(year=2024)
snap = pre.get_snapshot(DT, POLLUTANT)
# 캔버스 bounding box 안에 드는 측정소만 (여유 좀 두고: 경계 근처도 Kriging에 도움)
margin = 0.1
m = ((snap["lon"] >= canvas.min_lon - margin) & (snap["lon"] <= canvas.max_lon + margin) &
     (snap["lat"] >= canvas.min_lat - margin) & (snap["lat"] <= canvas.max_lat + margin))
snap = snap[m].reset_index(drop=True)
print(f"캔버스 주변 측정소: {len(snap)}개")

# 3. Kriging
OK = OrdinaryKriging(
    snap["lon"].values, snap["lat"].values, snap[POLLUTANT].values,
    variogram_model="spherical", verbose=False,
)
# 캔버스 격자 좌표로 추정
grid_lon = np.linspace(canvas.min_lon, canvas.max_lon, canvas.resolution)
grid_lat = np.linspace(canvas.min_lat, canvas.max_lat, canvas.resolution)
z, ss = OK.execute("grid", grid_lon, grid_lat)

# 4. 육지 마스킹 — 바다 칸은 NaN 으로
z_masked = np.where(canvas.land_mask, z, np.nan)
ss_masked = np.where(canvas.land_mask, ss, np.nan)

# 5. 그리기
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

im0 = axes[0].pcolormesh(grid_lon, grid_lat, z_masked, shading="auto", cmap="RdYlBu_r")
axes[0].scatter(snap["lon"], snap["lat"], c="black", s=8)
axes[0].set_title(f"Masked Kriging ({POLLUTANT}) — {REGION}")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(grid_lon, grid_lat, ss_masked, shading="auto", cmap="viridis")
axes[1].scatter(snap["lon"], snap["lat"], c="red", s=8)
axes[1].set_title("Uncertainty (masked)")
fig.colorbar(im1, ax=axes[1])

out = ROOT / "results" / f"masked_kriging_{REGION}.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"저장: {out}")