"""스냅샷 → Ordinary Kriging → 연속 농도장 + 불확실성 히트맵."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data.preprocess import AirQualityPreprocessor

# 1. 스냅샷
pre = AirQualityPreprocessor(year=2024)
snap = pre.get_snapshot("2024-07-01 13:00", "PM10")
lons = snap["lon"].values
lats = snap["lat"].values
vals = snap["PM10"].values
print(f"측정소 {len(snap)}개")

# 2. 보간할 격자 (전국 bounding box)
grid_lon = np.linspace(lons.min(), lons.max(), 200)
grid_lat = np.linspace(lats.min(), lats.max(), 200)

# 3. Ordinary Kriging (베리오그램 모델: spherical)
OK = OrdinaryKriging(
    lons, lats, vals,
    variogram_model="spherical",   # GP 커널에 해당. 나중에 바꿔볼 수 있음
    verbose=False,
)
z, ss = OK.execute("grid", grid_lon, grid_lat)  # z=추정값, ss=분산(불확실성)

# 4. 두 개 나란히 그리기
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (좌) 추정 농도장
im0 = axes[0].pcolormesh(grid_lon, grid_lat, z, shading="auto", cmap="RdYlBu_r")
axes[0].scatter(lons, lats, c="black", s=3)   # 실제 측정소 위치
axes[0].set_title("Kriging estimate (PM10)")
fig.colorbar(im0, ax=axes[0])

# (우) 불확실성 (분산) — 측정소에서 멀수록 높음
im1 = axes[1].pcolormesh(grid_lon, grid_lat, ss, shading="auto", cmap="viridis")
axes[1].scatter(lons, lats, c="red", s=3)
axes[1].set_title("Uncertainty (variance)")
fig.colorbar(im1, ax=axes[1])

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "kriging_20240701_13.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"저장: {out}")