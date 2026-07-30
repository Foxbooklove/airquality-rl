"""Canvas: 격자 생성 + 육지 마스킹.

region=None 이면 전국, region="서울특별시" 면 해당 시도만.
resolution 으로 격자 크기 조절. reference 보고 지역/해상도만 바꾸면 됨.
"""
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]

GEOJSON = ROOT / "data" / "raw" / "korea_sido.geojson"


def load_sido(region: str | None = None):
    """시도 경계를 위경도(4326) shapely 도형으로. region=None 이면 전국 합집합.

    geopandas(=GDAL) 가 있으면 그걸 쓰고, 없으면 json+shapely+pyproj 로 읽는다.
    GDAL 은 Windows 에서 설치가 자주 깨지고(애플리케이션 제어 정책이 DLL 을
    차단하는 경우도 있다) 여기선 파일 하나 읽는 용도뿐이라, 폴백 경로를 둔다.
    두 경로 모두 동일한 좌표 변환을 한다 (아래 CRS 주의사항 참고).

    ⚠️ korea_sido.geojson 은 좌표가 UTMK(미터, EPSG:5179)인데 헤더엔 4326 으로
       잘못 표기돼 있다. 5179 로 강제 지정한 뒤 위경도로 변환해야 측정소 좌표와 맞다.
    """
    try:
        import geopandas as gpd
        gdf = gpd.read_file(GEOJSON)
        gdf = gdf.set_crs(5179, allow_override=True)   # 실제 좌표계로 덮어쓰기
        gdf = gdf.to_crs(4326)                          # 위경도로 변환
        if region is not None:
            available = gdf["title"].tolist()           # 필터 전에 목록 저장
            gdf = gdf[gdf["title"] == region]
            if len(gdf) == 0:
                raise ValueError(f"지역 없음: {region}. 가능: {available}")
        return gdf.union_all()
    except Exception as e:
        if isinstance(e, ValueError) and "지역 없음" in str(e):
            raise
        import json
        import pyproj
        from shapely.geometry import shape
        from shapely.ops import unary_union, transform
        print(f"[Canvas] geopandas 사용 불가({type(e).__name__}) — "
              f"json+pyproj 폴백으로 경계 로드")
        feats = json.load(open(GEOJSON, encoding="utf-8"))["features"]
        if region is not None:
            available = [f["properties"]["title"] for f in feats]
            feats = [f for f in feats if f["properties"]["title"] == region]
            if not feats:
                raise ValueError(f"지역 없음: {region}. 가능: {available}") from None
        tr = pyproj.Transformer.from_crs(5179, 4326, always_xy=True).transform
        return unary_union([transform(tr, shape(f["geometry"])) for f in feats])


class Canvas:
    def __init__(self, region: str | None = None, resolution: int = 100):
        """
        region: None=전국, "서울특별시"/"경기도" 등=해당 시도만
        resolution: 한 변당 격자 칸 수 (bounding box 기준)
        """
        self.region = region
        self.resolution = resolution

        # 경계 로드 (CRS 처리는 load_sido 안에 있음)
        self.boundary = load_sido(region)        # 여러 시도를 하나의 폴리곤으로 합침
        self._prepared = prep(self.boundary)     # 점-포함 판정 빠르게

        # bounding box
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.boundary.bounds

        # 격자 생성 + 마스킹
        self._build_grid()

    def _build_grid(self):
        """bounding box 를 resolution×resolution 격자로 나누고, 육지 안 점만 남김."""
        lons = np.linspace(self.min_lon, self.max_lon, self.resolution)
        lats = np.linspace(self.min_lat, self.max_lat, self.resolution)
        self.grid_lon, self.grid_lat = np.meshgrid(lons, lats)

        # 각 격자점이 육지(경계 폴리곤) 안에 있는지
        flat_lon = self.grid_lon.ravel()
        flat_lat = self.grid_lat.ravel()
        mask = np.array([
            self._prepared.contains(Point(lo, la))
            for lo, la in zip(flat_lon, flat_lat)
        ])
        self.land_mask = mask.reshape(self.grid_lon.shape)   # True=육지, False=바다

    def summary(self):
        total = self.land_mask.size
        land = self.land_mask.sum()
        print(f"지역: {self.region or '전국'}")
        print(f"bounding box: lon [{self.min_lon:.2f}, {self.max_lon:.2f}], "
              f"lat [{self.min_lat:.2f}, {self.max_lat:.2f}]")
        print(f"격자: {self.resolution}×{self.resolution} = {total}칸")
        print(f"육지 칸: {land} ({land/total*100:.1f}%), 바다 칸: {total-land}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 전국
    canvas = Canvas(region=None, resolution=100)
    canvas.summary()

    # 육지 마스크 시각화
    plt.figure(figsize=(6, 8))
    plt.scatter(
        canvas.grid_lon[canvas.land_mask],
        canvas.grid_lat[canvas.land_mask],
        s=2, c="green", label="육지"
    )
    plt.scatter(
        canvas.grid_lon[~canvas.land_mask],
        canvas.grid_lat[~canvas.land_mask],
        s=2, c="lightblue", label="바다(제외)"
    )
    plt.legend()
    plt.title("Canvas land mask (전국)")

    out = ROOT / "results" / "exploration" / "canvas_mask_korea.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"저장: {out}")