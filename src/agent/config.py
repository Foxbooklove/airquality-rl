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
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    discount: float = 0.997

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
