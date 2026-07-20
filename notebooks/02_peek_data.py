"""확정자료 구조만 빠르게 확인 (앞 20줄만)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "raw" / "한국환경공단_에어코리아_최종확정 측정자료_20241231.xlsx"

# 앞 20행만 읽기 (전체 로딩 X)
df = pd.read_excel(path, nrows=20)

print("=== 컬럼 ===")
print(df.columns.tolist())
print("\n=== 앞 5줄 ===")
print(df.head())
print("\n=== 데이터 타입 ===")
print(df.dtypes)