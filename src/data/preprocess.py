"""AirKorea 대기질 데이터 전처리.

역할:
  - build_raw_parquet(): XLSX(월별 12시트) → raw parquet.  단발성(무거움), 한 번만.
  - process():           raw → datetime 변환 + 위경도 매핑 → 분석용 parquet.  반복 가능.
  - get_snapshot():      특정 시각의 전 측정소 관측(위경도 포함) 한 장.  환경/시각화 입력.

모델·환경은 get_snapshot() 만 보면 됨 (전처리 세부는 이 파일 안에 격리).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# 확정자료 원본 컬럼(한글) → 영문
_RENAME = {
    "지역": "region", "망": "network",
    "측정소코드": "code", "측정소명": "station",
    "측정일시": "datetime_raw",
    "아황산가스(SO2)": "SO2", "일산화탄소(CO)": "CO",
    "오존(O3)": "O3", "이산화질소(NO2)": "NO2",
    "미세먼지(PM10)": "PM10", "초미세먼지(PM25)": "PM25",
    "주소": "addr",
}
POLLUTANTS = ["SO2", "CO", "O3", "NO2", "PM10", "PM25"]


class AirQualityPreprocessor:
    def __init__(self, year: int = 2024):
        self.year = year
        self.raw_dir = ROOT / "data" / "raw"
        self.proc_dir = ROOT / "data" / "processed"
        self.proc_dir.mkdir(parents=True, exist_ok=True)

        self.xlsx = self.raw_dir / f"한국환경공단_에어코리아_최종확정 측정자료_{year}1231.xlsx"
        self.raw_parquet = self.proc_dir / f"measurements_{year}_raw.parquet"
        self.proc_parquet = self.proc_dir / f"measurements_{year}.parquet"
        self.stations_csv = self.raw_dir / "stations.csv"

    # ---------- 1단계: XLSX → raw parquet (단발성, 무거움) ----------
    def build_raw_parquet(self, force: bool = False):
        """월별 12시트를 모두 읽어 합친 뒤 raw parquet 저장. 한 번만 돌리면 됨."""
        if self.raw_parquet.exists() and not force:
            print(f"raw parquet 이미 있음: {self.raw_parquet.name} (force=True로 재생성)")
            return

        print("XLSX 전체 시트 읽는 중... (12개월, 오래 걸림)")
        sheets = pd.read_excel(self.xlsx, sheet_name=None)   # 모든 시트를 dict로
        df = pd.concat(sheets.values(), ignore_index=True)
        df = df.rename(columns=_RENAME)
        print(f"  합친 전체: {len(df):,} 행")

        df.to_parquet(self.raw_parquet, index=False)
        print(f"  저장: {self.raw_parquet.name} ({self.raw_parquet.stat().st_size/1e6:.1f} MB)")

    # ---------- 2단계: raw → datetime + 위경도 (반복 가능) ----------
    def process(self, save: bool = True) -> pd.DataFrame:
        """raw parquet 에 datetime 변환(24시 처리)과 위경도 매핑을 적용."""
        if not self.raw_parquet.exists():
            raise FileNotFoundError("raw parquet 없음. 먼저 build_raw_parquet() 실행.")

        df = pd.read_parquet(self.raw_parquet)

        # datetime 변환: YYYYMMDDHH(HH=1~24). 24시는 다음날 0시로.
        s = df["datetime_raw"].astype(str).str.zfill(10)
        date_part = s.str[:8]
        hour_part = s.str[8:].astype(int)
        df["datetime"] = (
            pd.to_datetime(date_part, format="%Y%m%d")
            + pd.to_timedelta(hour_part, unit="h")   # 0시 + HH시간 → 1시=01:00, 24시=익일 00:00
        )
        df = df.drop(columns=["datetime_raw"])

        # 위경도 매핑: 측정소명으로 inner join (stations.csv에 없는 측정소는 자동 제외)
        stations = pd.read_csv(self.stations_csv)
        coords = stations[["stationName", "lat", "lon"]].rename(columns={"stationName": "station"})
        df = df.merge(coords, on="station", how="inner")

        if save:
            df.to_parquet(self.proc_parquet, index=False)
        return df

    def load(self) -> pd.DataFrame:
        """이미 처리된 parquet 을 읽음 (없으면 process 실행)."""
        if self.proc_parquet.exists():
            return pd.read_parquet(self.proc_parquet)
        return self.process()

    # ---------- 3단계: 스냅샷 추출 (환경/시각화 입력) ----------
    def get_snapshot(self, dt, pollutant: str = "PM10", df: pd.DataFrame | None = None) -> pd.DataFrame:
        """특정 시각 dt 의 전 측정소 관측 한 장. 결측(NaN) 측정소는 제외.
        반환 컬럼: station, lat, lon, <pollutant>
        """
        if df is None:
            df = self.load()
        dt = pd.Timestamp(dt)
        snap = df[df["datetime"] == dt][["station", "lat", "lon", pollutant]]
        return snap.dropna(subset=[pollutant]).reset_index(drop=True)


if __name__ == "__main__":
    pre = AirQualityPreprocessor(year=2024)

    pre.build_raw_parquet()          # 이미 있으면 건너뜀
    df = pre.process()

    print(f"처리 후 전체: {len(df):,} 행")
    print(f"측정소 수: {df['station'].nunique()}개")
    print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"저장: {pre.proc_parquet.name}")

    # 스냅샷 동작 확인 — 임의 시각 하나
    snap = pre.get_snapshot("2024-07-01 13:00", "PM10", df=df)
    print(f"\n2024-07-01 13시 PM10 스냅샷: {len(snap)}개 측정소")
    print(snap.head())