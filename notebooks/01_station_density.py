"""측정소 위치를 folium 지도에 찍어 밀도 확인."""
from pathlib import Path

import pandas as pd
import folium

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "raw" / "stations.csv")

# 지도 중심을 측정소들의 평균 위치로 (대략 한국 중앙)
center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=center, zoom_start=7)   # zoom 7 = 전국이 한눈에

# 측정소마다 작은 원 찍기
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=2,                    # 점 크기
        color="blue",
        fill=True,
        fill_opacity=0.6,
        popup=row["stationName"],    # 클릭하면 측정소 이름 뜸
    ).add_to(m)

# HTML 로 저장
# 결과 폴더 없으면 생성
out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)

# HTML 로 저장
out = out_dir / "station_density.html"
m.save(str(out))
print(f"저장 완료: {out}")