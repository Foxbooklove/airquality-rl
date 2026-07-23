"""folium HTML 지도 시각화 (concentration_map 스타일 계승).

실제 지도 배경 위에:
  - 측정소: 값(PM10)에 따라 색 (파랑->빨강)
  - 드론 경로: 무작위 시작점에서 이동 궤적 (PolyLine + 스텝 마커, 시작=초록/끝=빨강)
  - 불확실성(최종 belief): 반투명 이미지 오버레이 (LayerControl 로 토글)

브라우저로 열면 확대/드래그 되는 인터랙티브 지도. PPT엔 스샷 떠서 넣으면 됨.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import folium
import branca.colormap as bcm


def _variance_overlay_png(var, land, path: Path):
    """분산 격자 -> 반투명 RGBA PNG (바다=투명). folium ImageOverlay 용."""
    v = np.array(var, dtype=float)
    vmax = float(np.nanmax(v[land])) if land.any() else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax + 1e-9)
    rgba = cm.viridis(norm(v))                 # [H, W, 4]
    rgba[~land, 3] = 0.0                        # 바다 투명
    rgba[land, 3] = 0.55                        # 육지 반투명
    rgba = rgba[::-1]                           # folium: row0 = 북쪽
    mpimg.imsave(path, rgba)


def trajectory_map(st_lon, st_lat, st_val, traj_lonlat, var, land,
                   bounds, out_html: Path, pollutant: str = "PM10",
                   overlay_dir: Path | None = None):
    """
    st_lon/lat/val : 지도에 표시할 측정소(보통 A 가시 측정소)
    traj_lonlat    : [(lon, lat), ...] 드론 궤적 (무작위 시작 포함)
    var, land      : 최종 분산 격자, 육지 마스크 (오버레이용)
    bounds         : (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    m = folium.Map(location=center, tiles="CartoDB positron")
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])   # 캔버스 크기에 자동 맞춤

    # --- 불확실성 오버레이 (토글 레이어) ---
    overlay_dir = overlay_dir or out_html.parent
    png = overlay_dir / (out_html.stem + "_unc.png")
    _variance_overlay_png(var, land, png)
    folium.raster_layers.ImageOverlay(
        image=str(png),
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.6, name="uncertainty (after survey)",
    ).add_to(m)

    # --- 측정소 (값 색상) ---
    vmin, vmax = float(np.min(st_val)), float(np.max(st_val))
    colormap = bcm.LinearColormap(["blue", "lime", "yellow", "red"],
                                  vmin=vmin, vmax=vmax)
    colormap.caption = f"{pollutant} (㎍/㎥)"
    stations = folium.FeatureGroup(name="stations (visible)")
    for lo, la, val in zip(st_lon, st_lat, st_val):
        folium.CircleMarker(
            location=[la, lo], radius=4, color=None, fill=True,
            fill_color=colormap(val), fill_opacity=0.85,
            popup=f"{pollutant}: {val:.0f}",
        ).add_to(stations)
    stations.add_to(m)

    # --- 드론 경로 ---
    path_latlon = [[la, lo] for lo, la in traj_lonlat]
    drone = folium.FeatureGroup(name="drone path")
    folium.PolyLine(path_latlon, color="black", weight=3, opacity=0.9).add_to(drone)
    for i, (la, lo) in enumerate(path_latlon):
        folium.CircleMarker(location=[la, lo], radius=3, color="white",
                            fill=True, fill_color="black", fill_opacity=1.0,
                            popup=f"step {i}").add_to(drone)
    # 시작(초록) / 끝(빨강)
    folium.Marker(path_latlon[0], icon=folium.Icon(color="green", icon="play"),
                  popup="start (random)").add_to(drone)
    folium.Marker(path_latlon[-1], icon=folium.Icon(color="red", icon="stop"),
                  popup="end").add_to(drone)
    drone.add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html
