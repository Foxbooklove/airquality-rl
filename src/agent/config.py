"""Sampled MuZero 스켈레톤 하이퍼파라미터.

여기 값들은 '나중에 튜닝해도 구조 안 바뀌는' 것들이야.
belief 표현 방식 / 정책 헤드 분포 / set encoder 종류 같은
'구조를 가르는 결정'은 여기가 아니라 각 모듈에서 정해.
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # --- 관측(observation) 차원 ---
    point_feat_dim: int = 3        # 측정점 하나: (lon_norm, lat_norm, value_norm)
    drone_feat_dim: int = 3        # 드론: (x_norm, y_norm, battery_norm)
    belief_channels: int = 2       # belief 그리드: (Kriging mean, variance)
    grid_size: int = 16            # belief 그리드 해상도 (프로토타입은 작게)

    # --- 잠재 공간 ---
    latent_dim: int = 64
    hidden_dim: int = 128

    # --- 행동(연속 (theta, distance)) ---
    action_dim: int = 2            # [theta_norm(-1..1), dist_norm(0..1)]
    # 이산 waypoint로 바꾸려면 sampling.py의 샘플러만 교체 (구조는 그대로)

    # --- Sampled MuZero 탐색 ---
    num_sampled_actions: int = 8   # K: 노드마다 뽑는 행동 수
    num_simulations: int = 24      # MCTS 시뮬 횟수
    root_explore_std: float = 0.5  # 루트 탐색 노이즈 (0이면 끔).
                                   # 정책이 한 방향으로 굳는 것 방지
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    discount: float = 0.997

    # --- 보상 스케일 ---
    reward_scale: float = 1.0
    """가치·보상 타깃에 곱하는 상수. **손실과 MCTS 안에서만** 쓰고, 사용자에게
    보고되는 정보이득 수치는 원단위 그대로 둔다.

    왜 필요한가 (실측):
      정보이득은 스텝당 약 0.005, 에피소드 리턴 약 0.07 이다. 그러면
        - 가치·보상 MSE 가 1e-3 수준이라 그 헤드에 그래디언트가 사실상 안 간다.
          측정: loss=1.688 중 v 0.001 / r 0.000 / p 1.687
        - PUCT 의 score = value + u 에서 u ≈ pb_c_init·(1/K)·√N/(1+n) ≈ 0.86 인데
          value ≈ 0.007 이라 **탐색 항이 가치를 100배 압도**한다. 그래서 방문수가
          [3,4,4,4,4,4,3,4] 처럼 균등해지고 정책 타깃도 균등이 된다.
      리턴을 O(1) 로 올리면 두 문제가 같이 풀린다. 양의 상수배는 최적 정책을
      바꾸지 않는다.

    train.py 의 --reward-scale auto 가 초반 몇 에피소드의 리턴 중앙값을 보고
    자동으로 정한 뒤 고정한다(중간에 바뀌면 이미 학습한 가치가 어긋난다).
    목적함수를 갈아끼워도 다시 잡을 필요가 없게 하려는 것이다.
    """

    # --- 학습 손실 가중치 (학습 목표 함수의 '조성'; 슬롯) ---
    value_loss_coef: float = 1.0
    reward_loss_coef: float = 1.0
    policy_loss_coef: float = 1.0
    unroll_steps: int = 3          # g로 몇 스텝 펼쳐서 학습할지

    # --- 물리 스케일 (env와 공유) ---
    max_step_frac: float = 0.25    # 한 번에 이동 가능한 최대 거리 = 캔버스의 25%
    max_battery: float = 10.0      # TTL (이동 예산)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    seed: int = 0
    episodes: int = 1             # 스켈레톤은 1 에피소드 + 1 업데이트만 돌려봄
