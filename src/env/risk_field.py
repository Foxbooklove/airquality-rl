"""위험도장 (risk field) — 지역 통계를 캔버스 격자 위의 연속장으로.

왜 '측정소 기준 연속장'인가:
  호흡기 질환 통계는 시도(17) 또는 시군구(~250) 단위 이산값이다. 이걸 격자에
  그대로 칠하면 행정경계에서 계단이 생기고, 그 계단이 보상에 그대로 실려서
  정책이 '경계선 넘기'를 학습해 버린다. 지역값을 측정소 지점에 앵커로 붙이고
  IDW 로 보간하면 경계가 부드럽고, 농도장과 같은 지점 집합 위에서 정의되므로
  두 장(場)의 좌표계가 자동으로 맞는다.

정규화 규약 (중요):
  w 는 **육지 칸 평균이 1** 이 되도록 정규화한다. 이래야
    - w ≡ 1 이면 위험가중 정보이득이 기존 정보이득과 **정확히 같아진다**
      (가중치를 켜기 전/후 비교가 성립)
    - 보상 스케일이 위험도 데이터의 절대 단위(진료인원? 10만명당?)에 안 흔들려서
      가중합 λ 를 데이터 바꿔도 다시 안 잡아도 된다.

쓰는 법:
    RiskField.uniform(canvas)                       # 데이터 없음 = 기존 동작
    RiskField.from_points(lon, lat, val, canvas)    # 임의 지점값 → 연속장
    RiskField.from_region_table(df, canvas, ...)    # 지역 통계 → 측정소 앵커 → 연속장
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 주소 앞머리 약칭 → 통계표에 흔히 쓰이는 정식 시도명.
# stations.csv 의 addr 는 "경남 창원시 ...", 통계표는 "경상남도" 라 서로 안 맞는다.
SIDO_ALIAS = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시",
    "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도",
    "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도",
    "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도",
}
# 통계표가 옛 명칭을 쓰는 경우까지 흡수 (강원도/전라북도 → 현행명)
_LEGACY = {"강원도": "강원특별자치도", "전라북도": "전북특별자치도",
           "제주도": "제주특별자치도", "세종시": "세종특별자치시"}


def canon_sido(name: str) -> str:
    """시도명을 정식 명칭 하나로 통일. 약칭·구명칭·공백 모두 흡수."""
    s = str(name).strip()
    if s in SIDO_ALIAS:
        return SIDO_ALIAS[s]
    if s in _LEGACY:
        return _LEGACY[s]
    # "경상남도 " 처럼 이미 정식명이면 그대로, 앞 2글자 약칭이면 매핑
    return SIDO_ALIAS.get(s[:2], s)


def canon_region(sido: str, sigungu: str | None = None) -> str:
    """(시도, 시군구) → 매칭용 정규 키. 공백 제거 + 시도명 통일.

    '창원시 마산합포구' 처럼 시군구가 2단인 경우가 있어 공백을 없앤 뒤 붙인다
    ('창원시마산합포구'). 통계표가 '창원시' 까지만 주면 아래 매칭에서
    최장 접두 일치로 처리한다.
    """
    key = canon_sido(sido)
    if sigungu:
        key += str(sigungu).replace(" ", "").strip()
    return key


class RiskField:
    """캔버스 격자 [res, res] 위의 위험 가중치 w. 육지 평균 1로 정규화됨."""

    def __init__(self, w: np.ndarray, canvas, name: str = "risk"):
        w = np.asarray(w, dtype=float)
        if w.shape != canvas.grid_lon.shape:
            raise ValueError(
                f"위험도장 shape {w.shape} 가 캔버스 격자 {canvas.grid_lon.shape} 와 다름")
        land = np.asarray(canvas.land_mask, dtype=bool)
        if not land.any():
            raise ValueError("캔버스에 육지 칸이 없음")

        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)              # 음수 위험도는 의미가 없다
        m = float(w[land].mean())
        if m <= 0:
            raise ValueError(f"위험도장의 육지 평균이 0 이하({m}) — 매칭이 전부 실패했을 수 있음")
        self.w = w / m                          # 육지 평균 = 1
        self.canvas = canvas
        self.land = land
        self.name = name

    # ---------------- 생성자들 ----------------
    @classmethod
    def uniform(cls, canvas, name: str = "uniform") -> "RiskField":
        """위험도 없음 = 모든 칸 동일. 이걸 쓰면 기존 정보이득과 완전히 동일."""
        return cls(np.ones_like(canvas.grid_lon, dtype=float), canvas, name)

    @classmethod
    def from_points(cls, lon, lat, val, canvas, power: float = 2.0,
                    smooth_deg: float = 0.05, name: str = "risk") -> "RiskField":
        """지점값 → IDW 보간 연속장.

        power       : 거리 감쇠 지수. 클수록 앵커 지점에 뾰족하게 붙는다.
        smooth_deg  : 거리에 더하는 완충 반경(도 단위). 0 이면 앵커 지점에서
                      1/0 발산 → 한 칸만 극단값이 되므로 반드시 양수를 준다.
                      0.05도 ≈ 5km 정도.

        Kriging 대신 IDW 인 이유: 위험도장은 에피소드 내내 안 변하는 정적 가중치라
        variogram 추정의 이점이 없고, IDW 는 항상 잘 정의되고 훨씬 싸다.
        """
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        val = np.asarray(val, dtype=float).ravel()
        ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
        lon, lat, val = lon[ok], lat[ok], val[ok]
        if len(val) == 0:
            raise ValueError("보간할 앵커 지점이 하나도 없음")

        gl, ga = canvas.grid_lon, canvas.grid_lat                  # [res, res]
        d = np.hypot(gl[..., None] - lon, ga[..., None] - lat)     # [res, res, P]
        wgt = 1.0 / np.power(d + smooth_deg, power)
        w = (wgt * val).sum(-1) / wgt.sum(-1)
        return cls(w, canvas, name)

    @classmethod
    def from_region_table(cls, table, canvas, value_col: str,
                          sido_col: str = "시도", sigungu_col: str | None = "시군구",
                          stations_csv: str | Path | None = None,
                          name: str | None = None, **idw) -> "RiskField":
        """지역 통계 DataFrame → 측정소에 앵커 → 연속장.

        table       : 지역 단위 통계 (시도별 17행이든 시군구별 ~250행이든)
        value_col   : 위험도로 쓸 값 컬럼 (예: 인구10만명당_호흡기질환_진료인원)
        sigungu_col : None 이면 시도 단위 통계로 취급

        매칭은 '공백 제거 후 최장 접두 일치'다. 통계표가 '창원시' 까지만 주고
        측정소 주소는 '창원시마산합포구' 여도 붙는다. 매칭 실패한 측정소는
        전국 중앙값을 받는다 (드롭하면 그 지역이 IDW 에서 통째로 비어버림).
        """
        import pandas as pd

        stations_csv = Path(stations_csv or ROOT / "data" / "raw" / "stations.csv")
        st = pd.read_csv(stations_csv)
        st = st.dropna(subset=["lat", "lon", "addr"]).reset_index(drop=True)

        # 통계표 쪽 정규 키 → 값
        lut: dict[str, float] = {}
        for _, r in table.iterrows():
            v = pd.to_numeric(r[value_col], errors="coerce")
            if not np.isfinite(v):
                continue
            key = canon_region(r[sido_col],
                               r[sigungu_col] if sigungu_col else None)
            lut[key] = float(v)
        if not lut:
            raise ValueError(f"통계표에서 '{value_col}' 유효값을 하나도 못 읽음")

        # 측정소 주소 → 정규 키 (addr: "경남 창원시 마산합포구 진동면 ...")
        keys = sorted(lut, key=len, reverse=True)     # 최장 접두 우선
        med = float(np.median(list(lut.values())))
        vals, n_hit = [], 0
        for addr in st["addr"].astype(str):
            tok = addr.split()
            probe = canon_region(tok[0], "".join(tok[1:3]) if len(tok) > 1 else None)
            hit = next((k for k in keys if probe.startswith(k)), None)
            vals.append(lut[hit] if hit else med)
            n_hit += hit is not None

        if n_hit == 0:
            raise ValueError(
                "측정소 주소와 통계표 지역명이 하나도 매칭되지 않음. "
                f"통계표 키 예시: {keys[:3]} / 측정소 주소 예시: "
                f"{st['addr'].iloc[0]!r}")
        print(f"[위험도장] 측정소 {len(st)}개 중 {n_hit}개 지역 매칭 "
              f"(미매칭 {len(st) - n_hit}개는 전국 중앙값 {med:.3g})")

        return cls.from_points(st["lon"], st["lat"], vals, canvas,
                               name=name or value_col, **idw)

    @classmethod
    def from_csv(cls, path, canvas, value_col: str, **kw) -> "RiskField":
        """지역 통계 CSV 파일 경로 버전 (from_region_table 의 편의 래퍼)."""
        import pandas as pd
        return cls.from_region_table(pd.read_csv(path), canvas,
                                     value_col=value_col, **kw)

    # ---------------- 유틸 ----------------
    def summary(self):
        v = self.w[self.land]
        print(f"위험도장 '{self.name}': 육지 {self.land.sum()}칸, "
              f"평균 {v.mean():.3f} (정규화됨), 범위 [{v.min():.3f}, {v.max():.3f}], "
              f"상위10% 컷 {np.percentile(v, 90):.3f}")

    def plot(self, out_path):
        """위험도장이 상식적으로 생겼는지 눈으로 확인 (수도권이 밝은가 등)."""
        import matplotlib.pyplot as plt
        from src.agent.visualize_search import use_korean_font
        use_korean_font()
        c = self.canvas
        shown = np.where(self.land, self.w, np.nan)
        plt.figure(figsize=(6, 8))
        plt.imshow(shown, origin="lower", cmap="magma",
                   extent=[c.min_lon, c.max_lon, c.min_lat, c.max_lat])
        plt.colorbar(label="위험 가중치 (육지 평균=1)")
        plt.title(f"risk field: {self.name}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"저장: {out_path}")
