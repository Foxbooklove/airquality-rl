"""드론 운동학 + 비행 가능 영역 — env · mcts · visualize 가 공유하는 단일 진실.

왜 따로 빼는가:
  '행동 → 다음 위치' 규칙이 세 곳에 흩어져 있었다.
    - env.step            실제 이동
    - visualize_search    트리 가지를 지도에 투영
    - (신규) mcts         마스킹하려면 노드마다 위치를 알아야 함
  세 곳이 어긋나면 화면·수읽기·실제가 서로 다른 곳을 가리킨다. 실제로 지금
  visualize_search 쪽이 env 보다 정확했다(깊이별 배터리 감소를 반영) — 이런 불일치를
  없애려고 여기 한 곳에만 둔다.

핵심 사실 하나: **드론 운동학은 우리가 정확히 안다.** MuZero 가 잠재공간에서
배워야 하는 건 농도장의 거동이고, 드론이 어디로 가는지는 결정적이다. 그래서 MCTS
노드에 위치를 실제 좌표로 태울 수 있고, 그래야 바다 판정이 가능하다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed"


# ---------------------------------------------------------------- 운동학
def canvas_span(extent) -> float:
    """캔버스 대각선 길이(도). 이동거리 스케일의 기준."""
    min_lon, max_lon, min_lat, max_lat = extent
    return float(np.hypot(max_lon - min_lon, max_lat - min_lat))


def reach_frac(battery: float, max_battery: float) -> float:
    """남은 배터리 비율만큼만 이동 가능 (env.step 의 하드 제약)."""
    return float(min(1.0, battery / max(max_battery, 1e-6)))


def step_offsets(actions, battery: float, max_battery: float,
                 max_step_frac: float, span: float):
    """행동 → 변위 (dlon, dlat). 클립하지 않는다.

    actions: [2] 또는 [K, 2]. (theta_norm in (-1,1), dist_norm in (0,1))
    반환도 스칼라 또는 [K] 배열.

    배터리 제약을 여기서 적용하는 이유는 샘플러가 '정규 공간' 행동만 다뤄야
    _unsquash 역변환이 성립하기 때문 (sampling.py 주석 참고).
    """
    a = np.asarray(actions, dtype=float)
    theta = a[..., 0] * np.pi
    dist = a[..., 1] * max_step_frac * span * reach_frac(battery, max_battery)
    return dist * np.cos(theta), dist * np.sin(theta)


def step_position(lon, lat, actions, battery, max_battery, max_step_frac,
                  extent, clip: bool = False):
    """(위치, 행동) → 다음 위치. clip=True 면 캔버스 경계로 자른다.

    clip 을 기본 False 로 둔 이유: 경계 이탈을 잘라서 없던 일로 만들면 정책에
    신호가 안 간다(README 한계 #3). 이탈 여부를 호출자가 보고 기각할 수 있게
    자르지 않은 좌표를 준다.
    """
    dlon, dlat = step_offsets(actions, battery, max_battery, max_step_frac,
                              canvas_span(extent))
    lon1, lat1 = np.asarray(lon) + dlon, np.asarray(lat) + dlat
    if clip:
        min_lon, max_lon, min_lat, max_lat = extent
        lon1 = np.clip(lon1, min_lon, max_lon)
        lat1 = np.clip(lat1, min_lat, max_lat)
    return lon1, lat1


def action_toward(lon, lat, target_lon, target_lat, battery, max_battery,
                  max_step_frac, extent, frac: float = 1.0):
    """'저 지점 쪽으로 한 걸음' 을 정규 공간 행동 [theta, dist] 으로.

    폴백에 쓴다 — 후보가 전부 막혔을 때 측정소 방향으로 물러나기 위한 것.
    측정소는 정의상 육지에 있고 사람 사는 곳이라 안전한 방향을 가리킨다.

    frac: 목표까지 거리의 이 비율만 간다 (1.0 이면 갈 수 있는 만큼 최대).
    """
    dlon, dlat = float(target_lon) - float(lon), float(target_lat) - float(lat)
    if dlon == 0.0 and dlat == 0.0:
        return np.array([0.0, 0.0], dtype=float)     # 이미 도착 = 제자리
    theta = np.arctan2(dlat, dlon)
    want = np.hypot(dlon, dlat) * frac
    span = canvas_span(extent)
    full = max_step_frac * span * reach_frac(battery, max_battery)
    dist_norm = 0.0 if full <= 0 else float(np.clip(want / full, 0.0, 1.0))
    return np.array([float(theta / np.pi), dist_norm], dtype=float)


STAY = np.array([0.0, 0.0], dtype=float)
"""제자리 대기. 현재 위치는 항상 육지이므로 **무조건 유효한** 최후 폴백.

시간 진행을 넣으면 이건 폴백이 아니라 진짜 전략이 된다 — 같은 지점을 다시 재는
것이 시간 변화를 추적하는 유효한 수가 되기 때문. 정적 필드에서는 얻을 게 없다.
"""


# ---------------------------------------------------------------- 비행 가능 영역
class FlightMask:
    """경로 검사용 고해상 육지 마스크 + 연결 성분.

    왜 canvas.land_mask 를 그대로 안 쓰는가:
      그건 보상 합산과 시작 위치용이고 해상도가 canvas.resolution(기본 60)이다.
      전국 bbox 에서 한 칸이 약 8km — 경로가 바다를 건너는지 판정하기엔 너무 거칠어서
      실제로 육지인 경로를 막거나 바다 경로를 통과시킨다.

    빌드 비용(512×512 = 26만 점 판정)이 있어 디스크에 캐시한다.
    """

    def __init__(self, canvas, resolution: int = 512, cache: bool = True):
        self.extent = (canvas.min_lon, canvas.max_lon,
                       canvas.min_lat, canvas.max_lat)
        self.resolution = resolution
        self.lons = np.linspace(canvas.min_lon, canvas.max_lon, resolution)
        self.lats = np.linspace(canvas.min_lat, canvas.max_lat, resolution)
        self.mask = self._build(canvas, cache)

        # 셀 크기(도) — 경로 샘플 간격을 정하는 데 쓴다
        self.dlon = (canvas.max_lon - canvas.min_lon) / (resolution - 1)
        self.dlat = (canvas.max_lat - canvas.min_lat) / (resolution - 1)
        self.cell = float(min(self.dlon, self.dlat))
        self._labels = None

    # ---------- 빌드 / 캐시 ----------
    def _cache_path(self, canvas) -> Path:
        reg = (canvas.region or "전국").replace(" ", "")
        return CACHE / f"flightmask_{reg}_{self.resolution}.npz"

    def _build(self, canvas, use_cache: bool) -> np.ndarray:
        fp = self._cache_path(canvas)
        if use_cache and fp.exists():
            blob = np.load(fp)
            m = blob["mask"]
            if m.shape == (self.resolution, self.resolution):
                print(f"[비행마스크] 캐시 사용: {fp.name}")
                return m.astype(bool)

        import shapely
        print(f"[비행마스크] {self.resolution}×{self.resolution} 생성 중... (최초 1회)")
        gl, ga = np.meshgrid(self.lons, self.lats)
        if hasattr(shapely, "contains_xy"):
            # shapely 2.x 벡터화 — prepared.contains 루프보다 100배 빠르다
            # (26만 점: 0.02초 vs 수 초~수 분). 결과는 동일함을 확인했다.
            m = shapely.contains_xy(canvas.boundary, gl.ravel(), ga.ravel()
                                    ).reshape(gl.shape)
        else:
            from shapely.geometry import Point
            from shapely.prepared import prep
            prepared = prep(canvas.boundary)
            m = np.array([prepared.contains(Point(lo, la))
                          for lo, la in zip(gl.ravel(), ga.ravel())]
                         ).reshape(gl.shape)
        if use_cache:
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fp, mask=m)
            print(f"[비행마스크] 캐시 저장: {fp.name} "
                  f"(육지 {m.sum()}/{m.size} = {m.mean()*100:.1f}%)")
        return m

    # ---------- 판정 ----------
    def in_bounds(self, lon, lat):
        min_lon, max_lon, min_lat, max_lat = self.extent
        return ((np.asarray(lon) >= min_lon) & (np.asarray(lon) <= max_lon)
                & (np.asarray(lat) >= min_lat) & (np.asarray(lat) <= max_lat))

    def _index(self, lon, lat):
        """위경도 → 마스크 격자 인덱스 (최근접). searchsorted 는 한 칸 위로
        치우치므로 직접 계산한다."""
        min_lon, _, min_lat, _ = self.extent
        j = np.rint((np.asarray(lon, dtype=float) - min_lon) / self.dlon)
        i = np.rint((np.asarray(lat, dtype=float) - min_lat) / self.dlat)
        return (np.clip(i, 0, self.resolution - 1).astype(int),
                np.clip(j, 0, self.resolution - 1).astype(int))

    def is_land(self, lon, lat):
        """경계 밖은 False. 벡터화 — lon/lat 이 배열이면 배열 반환."""
        ok = self.in_bounds(lon, lat)
        i, j = self._index(lon, lat)
        return ok & self.mask[i, j]

    def path_clear(self, lon0, lat0, lon1, lat1, n_samples: int | None = None):
        """출발→도착 선분 전체가 육지인가. [K] 배열 입력을 한 번에 처리.

        바다 횡단을 막는 이유는 안전이다 — 물 위에서 기능 고장이면 회수할 방법이
        없다. 도착점만 보면 남해를 가로질러 섬으로 건너뛰는 경로가 통과한다.

        샘플 간격은 마스크 셀 크기보다 촘촘해야 한다. 안 그러면 좁은 해협을
        건너뛰고 통과 판정이 난다.
        """
        lon0 = np.atleast_1d(np.asarray(lon0, dtype=float))
        lat0 = np.atleast_1d(np.asarray(lat0, dtype=float))
        lon1 = np.atleast_1d(np.asarray(lon1, dtype=float))
        lat1 = np.atleast_1d(np.asarray(lat1, dtype=float))

        if n_samples is None:
            longest = float(np.max(np.hypot(lon1 - lon0, lat1 - lat0)) or 0.0)
            # 셀 하나에 최소 2점 (Nyquist) — 좁은 해협 누락 방지
            n_samples = int(np.clip(np.ceil(longest / self.cell) * 2 + 2, 4, 512))

        t = np.linspace(0.0, 1.0, n_samples)[None, :]        # [1, n]
        lons = lon0[:, None] + (lon1 - lon0)[:, None] * t     # [K, n]
        lats = lat0[:, None] + (lat1 - lat0)[:, None] * t
        return self.is_land(lons, lats).all(axis=1)           # [K]

    def valid_step(self, lon0, lat0, lon1, lat1):
        """이 한 수가 허용되는가 = 도착점이 캔버스 안 + 경로 전체가 육지."""
        inb = self.in_bounds(lon1, lat1)
        return np.atleast_1d(inb) & self.path_clear(lon0, lat0, lon1, lat1)

    # ---------- 연결 성분 ----------
    @property
    def labels(self):
        """육지 연결 성분 라벨. 바다 횡단을 막으면 드론은 시작한 덩어리에 갇힌다."""
        if self._labels is None:
            from scipy.ndimage import label
            lab, n = label(self.mask)
            self._labels = lab
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0                      # 0 = 바다
            self._n_components = int(n)
            self._largest = int(sizes.argmax())
            print(f"[비행마스크] 육지 연결 성분 {n}개, "
                  f"최대 성분이 육지의 {sizes[self._largest]/self.mask.sum()*100:.1f}%")
        return self._labels

    def component_at(self, lon, lat) -> int:
        i, j = self._index(lon, lat)
        return self.labels[i, j]

    def mainland_mask(self) -> np.ndarray:
        """가장 큰 연결 성분(본토)만 True. 시작 위치를 여기로 제한하면
        제주·섬에서 시작해 갇히는 에피소드를 막는다."""
        _ = self.labels
        return self._labels == self._largest

    def summary(self):
        _ = self.labels
        print(f"비행마스크 {self.resolution}×{self.resolution}: "
              f"육지 {self.mask.sum()}칸 ({self.mask.mean()*100:.1f}%), "
              f"셀 {self.cell:.4f}° (약 {self.cell*111:.1f}km), "
              f"연결성분 {self._n_components}개")


if __name__ == "__main__":
    # 확인:  python -m src.env.motion
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.env.canvas import Canvas

    canvas = Canvas(region=None, resolution=60)
    fm = FlightMask(canvas, resolution=512)
    fm.summary()

    out = ROOT / "results" / "exploration"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 7))
    ext = [fm.extent[0], fm.extent[1], fm.extent[2], fm.extent[3]]
    ax[0].imshow(fm.mask, origin="lower", extent=ext, cmap="Greens")
    ax[0].set_title(f"고해상 육지 마스크 {fm.resolution}²")
    ax[1].imshow(fm.mainland_mask(), origin="lower", extent=ext, cmap="Blues")
    ax[1].set_title("최대 연결 성분 (본토)")
    plt.tight_layout()
    plt.savefig(out / "flightmask.png", dpi=110, bbox_inches="tight")
    print(f"저장: {out / 'flightmask.png'}")
