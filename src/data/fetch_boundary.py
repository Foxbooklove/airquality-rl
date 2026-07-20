"""한국 행정구역 경계 GeoJSON 다운로드 (statgarten/maps)."""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/statgarten/maps/main/json"

# 전국 시도 경계 (마스킹용). 파일명에 한글 있어서 URL 인코딩 필요
import urllib.parse
fname = "전국_시도_경계.json"
url = f"{BASE}/{urllib.parse.quote(fname)}"
out = out_dir / "korea_sido.geojson"

print(f"다운로드 중... ({fname})")
urllib.request.urlretrieve(url, out)
print(f"저장 완료: {out} ({out.stat().st_size/1e6:.1f} MB)")