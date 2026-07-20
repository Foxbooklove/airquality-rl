import geopandas as gpd
gdf = gpd.read_file("data/raw/korea_sido.geojson")
print(gdf.crs)          # 좌표계 — EPSG:4326(위경도)이어야 우리 측정소랑 맞음
print(gdf.columns.tolist())
print(gdf[[c for c in gdf.columns if c != 'geometry']].head())