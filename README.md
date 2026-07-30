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

모든 학습·시연은 하나의 엔트리포인트로 한다.

```bash
python -m src.agent.train                     # 헤드리스, 300 에피소드, 바닥부터
python -m src.agent.train --mode live         # 브라우저로 보면서 학습 (무한)
python -m src.agent.train --mode window       # matplotlib 창으로 보면서
python -m src.agent.train --load best         # 최고 기록에서 이어서
python -m src.agent.train --load latest       # 마지막 상태에서 이어서
python -m src.agent.train --load results/x.pt # 특정 파일에서
python -m src.agent.train --help              # 전체 옵션
```

`--mode live` 로 띄우면 콘솔에 주소가 찍힌다. 보는 쪽은 파이썬 없이 브라우저로
주소만 열면 되고, 같은 네트워크면 다른 기기에서도 접속된다. 화면에서
일시정지/재개(스페이스바)가 가능하다.

기본값은 **바닥부터 학습**이다. 이어서 하려면 `--load` 를 명시해야 한다
(실수로 이전 결과에 이어붙는 사고 방지).

주요 옵션 — 모드에 따라 기본값이 갈린다.

| 옵션 | headless | live / window |
|---|---|---|
| `--episodes` | 300 | 0 (무한) |
| `--ttl` (이동 예산) | 40 | 12 |
| `--sampled` (K) | 8 | 4 (트리가 깊어져 수읽기가 잘 보임) |
| `--figures` | on | off |

그 밖에 `--region`, `--resolution`, `--snapshot`, `--bbox-pad`, `--sims`,
`--lr`, `--no-train`, `--port` 등이 있다.

출력은 `results/` 아래 용도별로 나뉜다.

```
results/
├── checkpoints/   ckpt_latest.pt · ckpt_best.pt · gains.npy
├── figures/       learning_curve.png · uncertainty.png · baseline.png
├── maps/          trajectory.html  (folium)
└── debug/         mcts_candidates_step*.png · mcts_tree_step*.png
```

| 체크포인트 | 저장 시점 |
|---|---|
| `ckpt_latest.pt` | `--save-every` 에피소드마다 |
| `ckpt_best.pt` | 최근 10 에피소드 평균 정보이득이 최고를 갱신할 때만 |

단일 에피소드는 시작 위치가 무작위라 편차가 커서, best 판단은 이동평균으로 한다.

체크포인트에는 가중치뿐 아니라 **옵티마이저 상태 · 에피소드 수 · 기록 · best 기준값**이
함께 저장된다. 이것들을 빼면 재시작 때 Adam 이 처음부터 시작해 학습이 흔들리고,
best 기준값이 리셋되어 기존 `ckpt_best.pt` 를 더 나쁜 모델로 덮어쓴다.

디버깅용 보조 스크립트:

```bash
python -m src.agent.inspect_mcts   # MCTS 탐색 트리 뜯어보기
python -m src.agent.run_skeleton   # 더미 환경으로 배선만 확인 (데이터 불필요)
```

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
│   ├── train.py           통합 엔트리포인트 (학습 · 실시간 시연)
│   ├── live_server.py     브라우저 실시간 모니터링 (MJPEG 스트리밍)
│   ├── run_skeleton.py    더미 환경 배선 확인
│   ├── inspect_mcts.py    MCTS 트리 디버깅
│   └── visualize*.py      figure / folium 지도 / MCTS 트리 / 실시간 탐색
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

5. **정보이득이 드물게 음수가 된다.** pykrige 가 관측점 추가 시 variogram 을
   다시 추정하므로, 국소 분산은 줄어도 총합이 늘 수 있다. variogram 을 최초 1회만
   추정해 고정하면 해결된다.

6. **에피소드마다 바뀌는 것이 시작 위치뿐이다.** 가시 측정소 집합과 스냅샷이
   고정이라, 정책이 특정 농도장에 과적합될 수 있다. `reset()` 에서 스냅샷 시각과
   분할을 무작위화하면 완화된다.