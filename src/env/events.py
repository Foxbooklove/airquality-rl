"""고농도 사건 목록 — 학습 시나리오이자 평가 분모.

왜 '사건' 인가:
  측정으로 확인한 것들이 목적함수를 여기로 몰았다.
    - 공간 보간 정확도(RMSE)는 이미 천장이다. 측정소 601개로 줄일 수 있는
      여지의 87%가 소진돼 있고, 관측을 5배 늘려야 나머지의 25%를 가져간다.
    - 광역 고농도 사건은 쉽다. 전국이 다 나쁠 때는 초과 판정 AUPRC 가 이미
      0.99 라 드론이 기여할 게 없다.
    - 어려운 건 **평상시의 국지 고농도**다. 평상시 80㎍/㎥ 초과는 0.28%,
      초과확률 AUPRC 0.28, 관측을 늘리면 +14% 개선된다. 여지가 여기 있다.
    - 그리고 위성으로도 안 된다. 초과 지점의 5km 이웃이 같이 초과할 확률이
      20% 뿐이라(기저 대비 71배지만 절대값은 낮다) GEMS 화소(7×8km) 안에서
      신호가 희석된다. 현장 관측이 필요한 규모다.

사건 정의 (기본값):
  - 임계     PM10 >= 80 ㎍/㎥ (환경부 예보 '나쁨' 81 에 준함)
  - 국지     그 시각 전국 평균이 30~70 분위 (광역 오염 사건 제외)
  - 지속     3시간 이상 — **드론이 도달할 수 있는 시간 안에 존재하는 사건**
             (전체 국지 사건의 75%가 1~2시간 스파이크인데, 기지에서 출발해
              도착하면 이미 끝나 있다. 원리적으로 못 잡는 것을 분모에 넣으면
              지표가 무의미해진다.)

유형 분류 — 나중에 '어떤 사건을 못 잡는가' 분석의 축이다:
  plume     3측정소 이상 · 3시간 이상 · 중심이 30km 이상 이동
  point     측정소 1곳에서 3시간 이상 지속 (점오염원 / 국지 정체)
  spike     2시간 이하 (기본 설정에서는 지속 조건에 걸려 제외됨)
  wide      전국 평균이 높고 10측정소 이상 (광역 유입)
  other     그 외

실행:  python -m src.env.events            # 목록 생성 + 요약 + 캐시
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed"

# 시공간 군집 파라미터: 1시간을 몇 km 로 환산해 이웃을 정의할지.
# 20km/h 는 대략적인 오염 이동 속도 규모다.
HOURS_TO_KM = 20.0
LINK_RADIUS_KM = 30.0


@dataclass
class EventSpec:
    pollutant: str = "PM10"
    threshold: float = 80.0
    min_duration_h: float = 3.0
    local_only: bool = True            # 광역 사건 제외
    nat_lo_q: float = 0.30             # '평상시' 전국 평균 분위 하한
    nat_hi_q: float = 0.70
    completeness: float = 0.9          # 이 비율 이상 관측된 측정소만 사용


def _station_matrix(df: pd.DataFrame, pollutant: str, completeness: float):
    piv = df.pivot_table(index="datetime", columns="station", values=pollutant,
                         aggfunc="mean").sort_index()
    coord = df.groupby("station")[["lat", "lon"]].first()
    keep = [c for c in piv.columns if piv[c].notna().mean() > completeness]
    piv = piv[keep]
    coord = coord.loc[keep]
    return piv, coord


def build_events(df: pd.DataFrame, spec: EventSpec = EventSpec()) -> pd.DataFrame:
    """초과 (측정소, 시각) 을 시공간으로 군집해 사건 목록으로.

    반환 컬럼:
      event_id, start, end, duration_h, n_stations, stations(리스트),
      lat, lon (중심), peak, move_km, nat_mean, kind
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    piv, coord = _station_matrix(df, spec.pollutant, spec.completeness)
    A = piv.to_numpy()
    T = piv.index
    lat = coord.lat.to_numpy()
    lon = coord.lon.to_numpy()
    latm = np.deg2rad(np.nanmean(lat))
    xy = np.c_[lon * 111.0 * np.cos(latm), lat * 111.0]     # km 평면 근사

    nat = np.nanmean(A, axis=1)
    lo, hi = np.nanquantile(nat, spec.nat_lo_q), np.nanquantile(nat, spec.nat_hi_q)

    ti, si = np.where(A >= spec.threshold)
    if len(ti) == 0:
        return pd.DataFrame()
    hrs = (T[ti] - T[0]).total_seconds().to_numpy() / 3600.0

    pts = np.c_[xy[si], hrs * HOURS_TO_KM]
    pairs = cKDTree(pts).query_pairs(LINK_RADIUS_KM, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(len(ti), len(ti)))
    n, lab = connected_components(g, directed=False)

    rows = []
    for k in range(n):
        m = lab == k
        h, s, t = hrs[m], si[m], ti[m]
        dur = float(h.max() - h.min()) + 1.0
        stns = sorted(set(s.tolist()))
        natv = float(np.nanmean(nat[t]))
        # 중심 이동거리 — 플룸 판별
        move = 0.0
        if dur >= 3:
            hi_r = np.round(h).astype(int)
            cs = np.array([xy[s[hi_r == u]].mean(0) for u in np.unique(hi_r)])
            if len(cs) >= 2:
                move = float(np.linalg.norm(cs[-1] - cs[0]))
        is_local = lo < natv < hi
        if len(stns) >= 3 and dur >= 3 and move >= 30:
            kind = "plume"
        elif len(stns) == 1 and dur >= 3:
            kind = "point"
        elif dur <= 2:
            kind = "spike"
        elif (not is_local) and len(stns) >= 10:
            kind = "wide"
        else:
            kind = "other"
        rows.append(dict(
            event_id=k, start=T[t.min()], end=T[t.max()], duration_h=dur,
            n_stations=len(stns), stations=[coord.index[x] for x in stns],
            lat=float(lat[stns].mean()), lon=float(lon[stns].mean()),
            peak=float(np.nanmax(A[m.nonzero()[0] * 0 + t, s])) if False
                 else float(np.nanmax(A[t, s])),
            move_km=move, nat_mean=natv, is_local=bool(is_local), kind=kind))
    ev = pd.DataFrame(rows)

    if spec.local_only:
        ev = ev[ev.is_local]
    ev = ev[ev.duration_h >= spec.min_duration_h]
    return ev.sort_values("start").reset_index(drop=True)


def summarize(ev: pd.DataFrame) -> str:
    if ev.empty:
        return "사건 없음"
    L = [f"사건 {len(ev):,}건  (지속 {ev.duration_h.min():.0f}~{ev.duration_h.max():.0f}h, "
         f"최고농도 {ev.peak.max():.0f}㎍/㎥)"]
    L.append("  유형별: " + "  ".join(
        f"{k} {v}건" for k, v in ev.kind.value_counts().items()))
    L.append("  측정소 수: " + "  ".join(
        f"{k}곳 {v}건" for k, v in ev.n_stations.value_counts().head(5).sort_index().items()))
    mo = ev.start.dt.month.value_counts().sort_index()
    L.append("  월별: " + " ".join(f"{m}월{mo.get(m,0):3d}" for m in range(1, 13)))
    hh = ev.start.dt.hour.value_counts().sort_index()
    L.append("  발생 시각: " + " ".join(f"{h:02d}시{hh.get(h,0):3d}" for h in range(0, 24, 3)))
    return "\n".join(L)


def load_or_build(spec: EventSpec = EventSpec(), year: int = 2024,
                  cache: bool = True) -> pd.DataFrame:
    fp = CACHE / (f"events_{year}_{spec.pollutant}_{spec.threshold:.0f}"
                  f"_{spec.min_duration_h:.0f}h"
                  f"{'_local' if spec.local_only else ''}.pkl")
    if cache and fp.exists():
        return pd.read_pickle(fp)
    from src.data.preprocess import AirQualityPreprocessor
    df = AirQualityPreprocessor(year=year).load()
    df = df.dropna(subset=["lat", "lon", spec.pollutant])
    ev = build_events(df, spec)
    if cache:
        fp.parent.mkdir(parents=True, exist_ok=True)
        ev.to_pickle(fp)
    return ev


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="고농도 사건 목록 생성")
    ap.add_argument("--pollutant", default="PM10")
    ap.add_argument("--threshold", type=float, default=80.0)
    ap.add_argument("--min-duration", type=float, default=3.0)
    ap.add_argument("--all", action="store_true", help="광역 사건도 포함")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    spec = EventSpec(pollutant=a.pollutant, threshold=a.threshold,
                     min_duration_h=a.min_duration, local_only=not a.all)
    ev = load_or_build(spec, cache=not a.rebuild)
    print(f"[사건] {spec.pollutant} >= {spec.threshold:.0f}㎍/㎥, "
          f"{spec.min_duration_h:.0f}시간 이상"
          + (", 국지만" if spec.local_only else ", 전체"))
    print(summarize(ev))
    if not ev.empty:
        print("\n상위 5건 (최고농도 순):")
        cols = ["start", "duration_h", "n_stations", "peak", "move_km", "kind"]
        print(ev.nlargest(5, "peak")[cols].to_string(index=False))
