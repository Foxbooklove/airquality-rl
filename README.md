# airquality-rl

드론 자율 관측으로 대기질 농도장을 복원하는 강화학습 프로토타입.
Sampled MuZero 기반.

## 문제 정의

POMDP. 숨은 상태는 전국 대기질 농도장이고, 에이전트는 이를 직접 볼 수 없다.

- **관측**: 고정 측정소 값 + 드론이 이동하며 측정한 값
- **belief**: Kriging 농도장 (mean) + 불확실성 (variance)
- **행동**: 연속 (θ, d) — 방향과 이동거리
- **보상**: 정보이득 = 초기 대비 육지 칸 Kriging 분산 총합 감소 비율

상태 전이 구조:

- 농도장은 action 과 **무관**하게 진행 (프로토타입은 스냅샷 고정)
- action 은 드론 위치만 변경
- 관측만 위치에 의존 — 즉 action 은 "농도장의 어느 지점을 열어볼지"만 결정

## 설치

```bash
pip install torch pykrige folium branca matplotlib geopandas pandas requests pyarrow openpyxl
```

## 데이터 준비 (최초 1회)

`data/`, `configs/` 는 git 에 포함되지 않으므로 clone 후 직접 준비해야 한다.

1. **인증키** — `configs/secrets.py` 생성

   ```python
   SERVICE_KEY = "공공데이터포털_Decoding_인증키"
   ```

2. **경계 · 측정소 목록**

   ```bash
   python src/data/fetch_boundary.py    # data/raw/korea_sido.geojson
   python src/data/fetch_stations.py    # data/raw/stations.csv
   ```

3. **확정자료 엑셀** — 공공데이터포털에서 직접 다운로드 후 배치

   ```
   data/raw/한국환경공단_에어코리아_최종확정 측정자료_20241231.xlsx
   ```

4. **전처리** (무거움, 1회)

   ```bash
   python src/data/preprocess.py        # data/processed/measurements_2024.parquet
   ```

## 실행

```bash
python -m src.agent.run_real       # 학습 + figure 생성
python -m src.agent.inspect_mcts   # MCTS 탐색 트리 디버깅
python -m src.agent.run_skeleton   # 더미 환경으로 배선만 확인 (데이터 불필요)
```

설정은 `src/agent/run_real.py` 상단 `SETTINGS` 에서 조절한다.

| 항목 | 설명 |
|---|---|
| `REGION` | `None`=전국, `"서울특별시"` 등=해당 시도 |
| `RESOLUTION` | Kriging 격자 해상도. 크면 정확하지만 느림 |
| `BBOX_PAD` | `None`=전국 측정소 전부. 숫자=캔버스 주변만 사용 |
| `TTL` | 드론 이동 예산 (스텝 수) |
| `MAX_STEP_FRAC` | 스텝당 최대 이동거리 (캔버스 대각선 대비 비율) |
| `EPISODES` | 학습 에피소드 수 |

결과는 `results/` 에 저장된다: `proto_trajectory_map.html` (folium 지도),
`proto_uncertainty.png`, `proto_learning_curve.png`, `proto_baseline.png`,
`proto_muzero.pt`.

## 구조

```
src/
├── agent/
│   ├── config.py          하이퍼파라미터
│   ├── networks.py        h(표현) · g(동역학) · f(예측) + DeepSets · CNN
│   ├── muzero.py          3망 묶음 (initial/recurrent inference)
│   ├── sampling.py        tanh squashing 행동 샘플링 + IS 보정 π̂
│   ├── mcts.py            Sampled MuZero MCTS
│   ├── losses.py          가치 + 보상 + 정책 손실
│   ├── replay.py          replay buffer (※ 아직 학습 루프에 연결 안 됨)
│   ├── run_real.py        실데이터 학습 + figure 생성
│   ├── run_skeleton.py    더미 환경 배선 확인
│   ├── inspect_mcts.py    MCTS 트리 디버깅
│   └── visualize*.py      정적 figure / folium 지도 / MCTS 트리
├── data/                  AirKorea 수집 · 전처리
└── env/
    ├── canvas.py          격자 + 육지 마스킹
    ├── air_quality_env.py 실제 환경 (스텝마다 Kriging 재계산)
    └── dummy_env.py       더미 환경 + RewardFn 슬롯
```

### 확장을 위해 열어둔 곳

- **보상**: `dummy_env.py` 의 `RewardFn` 인터페이스. 현재는 정보이득만 사용.
  복원 정확도 등으로 교체 시 환경 · 모델 코드는 건드리지 않아도 된다.
- **다중 드론**: 상태에 드론 축을 유지 (`env.drones`, 현재 1대).
  belief 는 드론 수와 무관한 공유 자원.
- **이산 행동**: `sampling.py` 의 샘플러만 카테고리컬로 교체하면 된다.

## 현재 한계

프로토타입 단계이며, 아래는 파악된 문제다.

1. **학습이 진행되지 않는다.** 학습된 정책이 랜덤 baseline 을 유의하게
   이기지 못한다. 원인은 두 가지로 파악되었다.
   - 에피소드당 gradient step 이 1회뿐이다 (300 에피소드 = 300 업데이트).
     MuZero 는 통상 수만 회 이상이 필요하다.
   - `losses.compute_losses` 가 항상 `trajectory[0]` 에서만 unroll 하여,
     TTL=40 궤적 중 앞 3스텝만 학습에 쓰이고 나머지는 버려진다.

   `replay.py` 를 추가해 두었으나 아직 학습 루프에 연결하지 않았다.

2. **드론이 바다 위를 비행한다.** `land_mask` 가 보상 계산과 시작 위치에만
   쓰이고 이동 제약에는 적용되지 않는다. 비행금지 구역 마스킹도 미구현
   (`sampling.py` 에 TODO).

3. **경계 이탈에 페널티가 없다.** `np.clip` 으로 좌표만 자르므로 캔버스
   경계로 밀어붙여도 학습 신호가 없다.

4. **가치 헤드가 스칼라 MSE 다.** 논문은 categorical support 를 쓴다
   (`networks.py` 에 TODO).

5. **하이퍼파라미터가 분산되어 있다.** `config.py`, `run_real.py` 의
   `SETTINGS`, 그리고 `main()` 내 직접 지정이 섞여 있어 실험 재현이 어렵다.
