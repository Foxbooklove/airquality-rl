# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

드론 자율 관측으로 대기질 농도장을 복원하는 Sampled MuZero 프로토타입.
코드·주석·문서는 전부 한국어로 쓴다 (기존 파일 스타일 유지).

## 실행 환경 주의 (2026-07 현재)

**이 개발 머신에서는 `python` 이 아니라 `py -3.13` 을 써야 한다.**

Windows 애플리케이션 제어 정책(Smart App Control / WDAC)이 서명 안 된 확장 모듈
DLL 을 차단하고 있다. 상황:

| 인터프리터 | 상태 |
|---|---|
| `python` = 3.12 | `pandas` DLL 차단 → 아무것도 안 됨 |
| `py -3.13` | pandas·torch·shapely·pyproj·scipy·flask·pykrige·openpyxl 정상. **`pyarrow` 와 GDAL 만** 차단 |

그래서 두 곳에 폴백을 두었다 (정책이 풀리면 자동으로 원래 경로를 쓴다):

- **경계 로드** — [canvas.py](src/env/canvas.py) 의 `load_sido()` 가 geopandas 실패 시
  `json` + `shapely` + `pyproj` 로 읽는다. 좌표 변환은 양쪽이 동일하고 결과도 일치함을 확인했다
- **테이블 저장/로드** — [preprocess.py](src/data/preprocess.py) 가 parquet 엔진이 없으면
  `.pkl` 로 떨어진다. `parquet_ok()` / `_save()` / `_read()` 참고.
  기존 parquet 은 읽을 수 없으므로 XLSX 에서 다시 만들어야 한다:
  `py -3.13 -m src.data.preprocess --rebuild` (396MB XLSX, 수 분, `.pkl` 약 1GB)

근본 해결은 정책을 푸는 것이다. `results/` 에 옛 산출물이 있는 것으로 보아
정책이 나중에 켜졌다.

## 실행

모든 모듈은 **레포 루트에서 `python -m` 으로** 실행한다. import 가 전부
절대경로(`src.agent.…`)이고 각 파일이 `Path(__file__).parents[2]` 를 ROOT 로
잡으므로, 파일 경로로 직접 실행하면 깨진다.

```bash
python -m src.agent.train                 # 헤드리스 학습, 300 에피소드, 바닥부터
python -m src.agent.train --mode live     # 브라우저 실시간 모니터링 (무한)
python -m src.agent.train --mode window   # matplotlib 창
python -m src.agent.train --load best|latest|<경로>   # 이어서 학습
python -m src.agent.train --help
```

`--mode` 에 따라 `--episodes / --ttl / --sampled / --figures` 기본값이 갈린다
([train.py:88-98](src/agent/train.py#L88-L98)). `--load` 를 생략하면 **항상 바닥부터**다.

보조 스크립트:

```bash
python -m src.agent.run_skeleton   # 더미 환경 배선 확인 — 데이터 불필요, 가장 빠른 smoke test
python -m src.agent.inspect_mcts   # MCTS 트리 디버깅 → results/debug/*.png
python -m src.env.canvas           # 육지 마스크 시각화
python -m src.env.motion           # 고해상 비행 마스크 + 연결 성분 시각화
python -m src.data.preprocess      # XLSX → parquet (무거움, 최초 1회)
python -m src.data.analyze_timescale --pollutant PM10 --max-lag 72
                                   # 농도장 시간 스케일 (시간 진행 도입 판단 근거)
```

테스트 프레임워크는 없다. 변경 후 검증은 `run_skeleton` (배선) → 짧은
`train --episodes 5` (실데이터) 순으로 한다.

## 데이터 준비 (clone 직후 1회)

`data/`, `configs/secrets.py`, `results/` 는 gitignore 다. README "데이터 준비"
절차대로 `configs/secrets.py` (공공데이터포털 인증키) → `fetch_boundary.py` →
`fetch_stations.py` → XLSX 수동 배치 → `preprocess.py` 순으로 만들어야
`train.py` 가 돈다.

`requirements.txt` 에 **flask 가 빠져 있다** — `--mode live` 를 쓰려면 따로 설치.

## 구조

`src/env` ↔ `src/agent` 는 [dummy_env.py](src/env/dummy_env.py) 의 `Observation`
dataclass 하나로만 연결된다. 이 계약(모든 텐서에 배치축 1 유지: `points [1,N,3]`,
`mask [1,N]`, `belief_grid [1,2,G,G]`, `drone [1,3]`)만 지키면 환경을 통째로
갈아끼워도 에이전트 코드는 안 건드린다. `DummyDroneEnv` 와 `AirQualityEnv` 가
정확히 같은 `reset/step` 시그니처를 갖는 이유다.

### 에이전트 (`src/agent/`)

- [networks.py](src/agent/networks.py) — h(표현: DeepSets + CNN + 드론 MLP) ·
  g(동역학) · f(예측: 대각 가우시안 정책 + 스칼라 가치)
- [muzero.py](src/agent/muzero.py) — 세 망을 `initial_inference` /
  `recurrent_inference` 로 묶은 얇은 래퍼
- [sampling.py](src/agent/sampling.py) — tanh squashing 행동 샘플러 + IS 보정 π̂
- [mcts.py](src/agent/mcts.py) — Sampled MuZero MCTS (노드마다 K개만 확장, prior=1/K)
- [losses.py](src/agent/losses.py) — 가치(MSE) + 보상(MSE) + 정책(교차엔트로피)
- [train.py](src/agent/train.py) — 유일한 엔트리포인트. self-play · 학습 ·
  체크포인트 · figure 생성을 전부 여기서 한다
- [config.py](src/agent/config.py) — 튜닝 대상 하이퍼파라미터만. **구조를 가르는
  결정**(belief 표현 방식, 정책 분포 종류, set encoder 종류)은 config 가 아니라
  해당 모듈 안에서 정한다

### 이 프로젝트에서 꼭 알아야 할 규칙

**이동 규칙은 [motion.py](src/env/motion.py) 한 곳에만 있다.** `env.step` ·
MCTS(수읽기) · `visualize_search`(화면 투영)가 같은 `step_position` 을 쓴다.
예전엔 세 곳에 복사돼 있었고 실제로 어긋나 있었다 — 시각화만 깊이별 배터리
감소를 반영하고 MCTS 는 안 했다. 이동 공식을 바꿀 일이 생기면 여기만 고친다.

**행동 공간은 '정규 공간'이다.** 샘플러는 항상 (θ∈(-1,1), dist∈(0,1)) 만 다루고,
배터리에 따른 이동거리 제한은 `motion.step_offsets` 가 적용한다
(`reach_frac = battery/max_battery`). 이 제약을 샘플러로 옮기면 `_unsquash`
역변환이 불가능해져 IS 보정 log-prob 이 틀어진다.

**비행 마스킹은 `_expand` 한 곳에 걸면 트리 전체에 적용된다.** 신경망 정책이
MCTS 로 들어가는 통로가 [mcts.py](src/agent/mcts.py) 의 `_expand` 하나뿐이다
(루트는 `initial_inference`, 깊은 노드는 `recurrent_inference` 경유). 마스킹을
따로 전파할 필요가 없다.

**연속 가우시안이라 로짓 마스킹이 불가하다 — 기각 샘플링을 쓴다.** 카테고리컬이면
로짓을 `-inf` 로 밀면 되지만 정책이 연속 분포다. `sample(validate=..., fallback=...)`
로 넉넉히 뽑아 유효한 것만 남긴다. 기하 판정은 호출자(mcts)가 하고 `sampling.py` 는
분포만 안다.

**기각 샘플링은 IS 보정을 깨지 않는다.** 절단분포는 `q(a)/Z` 이고 `Z` 는 그 노드의
모든 샘플에 동일한 상수라, `is_corrected_policy_target` 과 `losses` 의 log 공간
softmax 에서 상쇄된다.

**폴백 행동을 유효 후보와 섞으면 안 된다.** 제자리 대기(`motion.STAY`)는 제안분포의
극단 꼬리라 `beta_logprob ≈ -12` 다. 섞으면 `log(N) - beta` 가 폭발해 IS 보정 질량을
100% 독식한다(측정 확인). 그래서 **유효 후보가 0개일 때만** 쓴다 — 그러면 후보
집합이 1개가 되고 `log_softmax` 가 정확히 0이라 정책 손실이 0이 되어 자동으로
안전하다. 가치·보상 손실은 정상 적용된다(그 전이는 실제로 일어났으므로 맞다).

**MCTS 노드는 위치와 배터리를 들고 있다.** 노드는 잠재상태지만 **드론 운동학은
정확히 안다**(모르는 건 농도장이다). 부모 위치 + 행동으로 자식 위치를 정확히
계산하므로 바다 판정이 가능하다. 배터리도 깊이마다 1씩 줄여야 한다 — 이동거리가
잔량에 비례하므로, 예전처럼 모든 깊이에 루트 배터리를 넘기면 수읽기가 '갈 수 없는
곳'을 계획에 넣는다.

**경로 검사에는 `canvas.land_mask` 를 쓰면 안 된다.** 그건 보상 합산·시작 위치용이고
해상도가 `canvas.resolution`(기본 60)이다. 전국 bbox 에서 한 칸이 약 8km 라 바다
횡단 판정에 못 쓴다. [motion.py](src/env/motion.py) 의 `FlightMask` 가 별도로
고해상(기본 512, 한 칸 약 1.5km) 마스크를 만들어 `data/processed/flightmask_*.npz`
에 캐시한다.

**바다 횡단 금지는 안전 요구사항이다.** 물 위에서 기능 고장이면 회수할 방법이 없다.
그래서 도착점만이 아니라 **경로 전체**를 검사한다. 부수 효과로 드론이 시작한 육지
연결 성분에 갇히므로, `AirQualityEnv(flight_mask=...)` 를 주면 시작 위치가
본토(최대 연결 성분)로 제한된다. 안 주면 제주·섬에서 시작해 갇히는 에피소드가 섞인다.

**tanh squashing 은 되돌릴 수 없는 결정이 아니라 정책 붕괴 대책이다.** clamp 를
쓰면 mean 이 범위 밖으로 밀릴 때 샘플이 전부 경계값으로 잘린다. 변수변환이므로
log-prob 에 야코비안 보정이 필수 (`_squashed_logprob`).

**정책 타깃은 방문수가 아니라 IS 보정 π̂ 다** (`is_corrected_policy_target`).
β 에서 뽑았기 때문에 자주 뽑힌 행동이 과대표현되고, 이를 log 공간에서 보정한다.
Sampled MuZero 의 핵심이 여기 한 곳이다.

**체크포인트는 가중치만 저장하면 안 된다.** `save_ckpt` 는 optimizer state ·
episode · gains · best_ma 를 함께 넣는다. 빼먹으면 재시작 때 Adam 이 리셋되고,
`best_ma=-inf` 로 초기화돼 기존 `ckpt_best.pt` 를 더 나쁜 모델로 덮어쓴다.

**`ckpt_best.pt` 판정은 최근 10 에피소드 이동평균으로 한다.** 시작 위치가
무작위라 단일 에피소드 편차가 크기 때문.

**geojson 좌표계가 헤더와 다르다.** `korea_sido.geojson` 은 실제 UTMK(5179)인데
헤더에 4326 으로 잘못 적혀 있어 [canvas.py:29-30](src/env/canvas.py#L29-L30) 에서
`set_crs(5179, allow_override=True)` 후 4326 으로 변환한다. 이 두 줄을 지우면
측정소 좌표와 격자가 안 맞는다.

**Kriging 이 병목이다.** `env.step` 마다 `OrdinaryKriging` 을 처음부터 다시
푼다. 에피소드 하나가 수 초~수십 초 걸리는 원인이고, 학습 속도 관련 작업은
거의 전부 여기로 수렴한다.

### 환경 (`src/env/`)

- [canvas.py](src/env/canvas.py) — 격자 + 육지 마스크(저해상, 보상·시작위치용)
- [motion.py](src/env/motion.py) — **드론 운동학 + 비행 가능 영역.** `step_position`
  (env/mcts/visualize 공유), `FlightMask`(고해상 마스크·경로 검사·연결 성분),
  `action_toward`·`STAY`(폴백용)
- [air_quality_env.py](src/env/air_quality_env.py) — Kriging belief + 보상
- [rewards.py](src/env/rewards.py) · [risk_field.py](src/env/risk_field.py) —
  ⚠ 목적함수 재설계 중이라 **여기 항들은 확정된 게 아니다.** 골격(`RewardTerm` 상속
  → 등록 → `--reward name=weight`)만 재사용 대상

### 확장 슬롯

- 보상 교체 → `dummy_env.py` 의 `RewardFn` 인터페이스 (`AirQualityEnv(reward_fn=…)`)
- 이산 waypoint → `sampling.py` 의 샘플러만 카테고리컬로 교체 (mcts/networks 불변)
- 다중 드론 → `env.drones` 리스트를 N개로, belief 는 공유 자원이라 그대로
- 비행금지 구역 → `FlightMask.mask` 에서 해당 칸을 False 로 (경로 검사가 자동 반영)

## 알려진 미해결 문제

착수 전에 README "현재 한계" 절을 읽을 것. 특히:

- **학습이 사실상 안 된다.** 에피소드당 gradient step 이 1회뿐이고,
  `compute_losses` 가 항상 `trajectory[0]` 에서만 unroll 해 TTL=40 궤적의 앞
  3스텝만 쓴다. [replay.py](src/agent/replay.py) 를 이 두 문제를 풀려고 만들어
  뒀으나 **아직 `train.py` 에 연결하지 않았다** — 학습 개선 작업은 여기서 시작.
- **시간 진행이 없다.** `gt_mean` 이 `__init__` 에서 한 번 계산돼 300 에피소드가
  전부 같은 농도장이다. `Observation` 에 시간 특성도 없다. `g`(동역학망)가 배우는
  건 시간 변화가 아니라 belief 갱신이라, MuZero 를 붙인 명분이 약하다.
- **`visible_idx` 도 `__init__` 에서 정해진다** (`reset()` 아님) — 가시 측정소
  집합이 모든 에피소드에서 동일하다. 농도장 고정 + 측정소 배치 고정이면 "어디가
  빈 곳인가" 패턴이 하나뿐이라 일반화를 검증할 수 없다. 시작 시각을 무작위화하면
  같이 풀린다.
- **정보이득은 측정값과 무관하다** (기하만으로 결정). 그래서 최적 경로를 날기 전에
  다 계산할 수 있고, GP 센서 배치에서 탐욕이 (1−1/e) 근사를 보장한다는 알려진
  결과가 있어 RL 이 크게 이길 여지가 구조적으로 없다. 목적함수 재설계의 핵심 논점.
- **baseline 이 랜덤 하나뿐이다.** 진짜 비교 대상은 탐욕(myopic) — 매 스텝 즉시
  이득 최대. 이게 없어서 지금 성능 주장이 약하다.
- 가치 헤드가 스칼라 MSE (논문은 categorical support).
- pykrige 가 관측 추가마다 variogram 을 재추정해 정보이득이 드물게 음수가 된다.
- 예산 제약이 느슨하다 — TTL 40 에 스텝당 캔버스 대각선 6% 면 거의 어디든 갈 수
  있어 경로계획의 근시안 실패가 안 생긴다. TTL·스텝 거리를 줄여야 수읽기가 일한다.

**해결됨** (이전 버전의 한계였던 것)
- ~~드론이 바다 위를 난다~~ → `FlightMask` 경로 검사 + 기각 샘플링 (`--mask`, 기본 on)
- ~~경계 이탈에 페널티가 없다~~ → 경계 이탈도 기각 조건. `np.clip` 은 마스킹을 끌 때의
  안전망으로만 남음
