"""특정 시각 PM10 스냅샷을 folium 지도에 색으로 표시."""
import sys
from pathlib import Path

import pandas as pd
import folium
import branca.colormap as cm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data.preprocess import AirQualityPreprocessor

# 스냅샷 뽑기
pre = AirQualityPreprocessor(year=2024)
snap = pre.get_snapshot("2024-07-01 13:00", "PM10")
print(f"측정소 {len(snap)}개, PM10 범위 {snap['PM10'].min()}~{snap['PM10'].max()}")

# 값 → 색 매핑 (파랑=낮음, 빨강=높음)
vmin, vmax = snap["PM10"].min(), snap["PM10"].max()
colormap = cm.LinearColormap(["blue", "lime", "yellow", "red"], vmin=vmin, vmax=vmax)
colormap.caption = "PM10 (㎍/㎥)"

# 지도
center = [snap["lat"].mean(), snap["lon"].mean()]
m = folium.Map(location=center, zoom_start=7)

for _, row in snap.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=4,
        color=None,
        fill=True,
        fill_color=colormap(row["PM10"]),
        fill_opacity=0.8,
        popup=f"{row['station']}: {row['PM10']}",
    ).add_to(m)

colormap.add_to(m)   # 범례

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "concentration_map_20240701_13.html"
m.save(str(out))
print(f"저장: {out}")