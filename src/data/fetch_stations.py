"""AirKorea 측정소 목록을 받아 data/raw/stations.csv 로 저장."""
import sys
from pathlib import Path

import requests
import pandas as pd

# configs/secrets.py 에서 키 불러오기
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 레포 루트를 경로에 추가
from configs.secrets import SERVICE_KEY

URL = "https://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getMsrstnList"

params = {
    "serviceKey": SERVICE_KEY,   # requests 가 자동 인코딩 → Decoding 키를 넣는 이유
    "returnType": "json",
    "numOfRows": 1000,           # totalCount(673)보다 크게 → 한 번에 전부
    "pageNo": 1,
}

resp = requests.get(URL, params=params, timeout=10)
resp.raise_for_status()          # HTTP 에러면 여기서 멈춤
data = resp.json()

# 정상 응답인지 확인
header = data["response"]["header"]
if header["resultCode"] != "00":
    raise RuntimeError(f"API 에러: {header['resultMsg']}")

items = data["response"]["body"]["items"]
print(f"받은 측정소 수: {len(items)}")

# DataFrame 으로 변환 + 필드 정리
df = pd.DataFrame(items)
df = df.rename(columns={"dmX": "lat", "dmY": "lon"})   # dmX=위도, dmY=경도
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")  # 문자열 → 숫자
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

# 좌표 없는 측정소 확인 (있으면 밀도 계산에서 빠짐)
missing = df[df[["lat", "lon"]].isna().any(axis=1)]
print(f"좌표 결측 측정소: {len(missing)}개")

# 저장
out = Path(__file__).resolve().parents[2] / "data" / "raw" / "stations.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")   # 엑셀에서 한글 안 깨지게 utf-8-sig
print(f"저장 완료: {out}")