# airquality-rl

RL-based drone observation path planning for air quality spatial map reconstruction.
AirKorea fixed-station data 기반, Google chip placement 방법론 이식.

## 구조
- src/data/ — API 호출, 결측 처리, 격자 매핑 (전처리 전부)
- src/env/ — RL 환경 (state/action/reward, Kriging 농도장)
- src/agent/ — 정책망, PPO
- src/train.py, src/eval.py — 학습 / 평가
- 
otebooks/ — 탐색·시각화용
- configs/ — 격자 해상도, 하이퍼파라미터
