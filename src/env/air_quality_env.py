"""실제 대기질 RL 환경 (프로토타입).

전이 구조(합의된 것):
  - 농도장: action 과 무관 (프로토타입은 스냅샷 고정 -> 시간진행 없음)
  - action: 드론 위치만 변경
  - 관측: 드론 위치가 '농도장의 어느 지점을 여느냐'만 결정

belief = Kriging(관측점들) 의 (mean, variance).
기본 reward = 정보이득 = 드론이 한 점 측정 -> Kriging 재계산 -> 육지 칸 분산 총합 감소분.
(분산 감소는 측정값과 무관, 위치만으로 결정 — 그래서 정답 C 없이 계산됨)

reward_fn 을 주면 그쪽이 우선한다. env 는 before/after belief 를 모두 들고 있으므로
(mean_prev/var_prev vs mean/var) 복합 보상이 임의의 지표를 계산할 수 있다.
호흡기 위험도 가중 등은 src/env/rewards.py 의 CompositeReward 참고.

상태는 '드론 축'을 열어둠(num_drones=1). belief 는 드론 수와 무관한 공유 자원.
나중에 다중 드론이면 drones 를 N개로 늘리고 obs 의 drone 축만 확장.

dummy_env 와 reset/step/Observation 인터페이스 동일 -> run 루프 안 바뀜.
"""
from __future__ import annotations

import numpy as np
import torch
from pykrige.ok import OrdinaryKriging

from src.agent.config import ModelConfig
from src.env import motion
from src.env.dummy_env import Observation, RewardFn


class AirQualityEnv:
    def __init__(self, canvas, snapshot, cfg: ModelConfig,
                 pollutant: str = "PM10", visible_ratio: float = 0.5,
                 value_scale: float = 100.0, reward_fn: RewardFn | None = None,
                 random_start: bool = True, bbox_pad: float | None = None,
                 seed: int = 0, flight_mask=None):
        """
        canvas   : Canvas (min/max lon·lat, resolution, grid_lon/lat, land_mask)
        snapshot : DataFrame [station, lat, lon, <pollutant>]  (get_snapshot 결과)
        """
        self.canvas = canvas
        self.cfg = cfg
        self.pollutant = pollutant
        self.value_scale = value_scale
        self.reward_fn = reward_fn            # None 이면 내장 정보이득
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)

        # 캔버스 격자 (Kriging 실행용 1D 좌표축)
        self.gx = canvas.grid_lon[0, :]       # [res]
        self.gy = canvas.grid_lat[:, 0]       # [res]
        self.land = np.asarray(canvas.land_mask, dtype=bool)   # [res, res]
        self.extent = (canvas.min_lon, canvas.max_lon,
                       canvas.min_lat, canvas.max_lat)
        # 바다 횡단을 막으면 드론은 시작한 육지 덩어리에 갇힌다. 시작 위치를
        # 본토(최대 연결 성분)로 제한하지 않으면 제주·섬에서 시작해 그 안에서만
        # 맴도는 에피소드가 섞인다.
        self.flight_mask = flight_mask
        self._start_pool = None               # 시작 위치 후보 캐시 (첫 reset 에서 계산)

        # 전체 스냅샷 = 시뮬레이터의 '진짜 농도장' (드론 측정값 생성용)
        s = snapshot.dropna(subset=[pollutant]).reset_index(drop=True)
        # bbox_pad=None 이면 필터 없음(전국 측정소 전부 사용).
        # 농도장은 행정구역을 안 따르므로 캔버스 밖 측정소도 경계 추정에 유용.
        # 숫자를 주면 캔버스 bbox + 그 비율만큼 여유를 둔 범위로 컷.
        if bbox_pad is not None:
            pad = bbox_pad * max(canvas.max_lon - canvas.min_lon,
                                 canvas.max_lat - canvas.min_lat)
            s = s[(s["lon"] >= canvas.min_lon - pad) & (s["lon"] <= canvas.max_lon + pad) &
                  (s["lat"] >= canvas.min_lat - pad) & (s["lat"] <= canvas.max_lat + pad)]
            s = s.reset_index(drop=True)
        if len(s) < 5:
            raise ValueError(f"사용 가능한 측정소가 너무 적음: {len(s)}개")
        self.n_stations = len(s)
        self.st_lon = s["lon"].to_numpy(float)
        self.st_lat = s["lat"].to_numpy(float)
        self.st_val = s[pollutant].to_numpy(float)
        self.gt_mean = self._krige(self.st_lon, self.st_lat, self.st_val)[0]

        # A(초기 가시 측정소) = 스냅샷의 일부만 -> belief 를 일부러 불확실하게
        n = len(self.st_val)
        k = max(4, int(round(n * visible_ratio)))
        self.visible_idx = self.rng.choice(n, size=k, replace=False)

    # ---------------- 표준 RL 인터페이스 ----------------
    def reset(self) -> Observation:
        self.battery = self.cfg.max_battery
        # 관측점 = A 측정소 (lon, lat, value)
        self.obs_lon = list(self.st_lon[self.visible_idx])
        self.obs_lat = list(self.st_lat[self.visible_idx])
        self.obs_val = list(self.st_val[self.visible_idx])

        # 드론(들) — 축 열어둠. 지금은 1대. 무작위 육지 지점(또는 중심)에서 시작.
        cx, cy = self._random_land_point() if self.random_start else self._land_center()
        self.drones = [{"lon": cx, "lat": cy}]

        self.mean, self.var = self._krige(
            np.array(self.obs_lon), np.array(self.obs_lat), np.array(self.obs_val))
        self.var0 = max(float(self.var[self.land].sum()), 1e-9)  # 정규화 기준
        # 직전 스텝 belief — RewardFn 이 before/after 를 모두 봐야 하므로 env 가 들고 있는다
        self.mean_prev, self.var_prev = self.mean.copy(), self.var.copy()
        self.traj = [(cx, cy)]                # PPT용 궤적 기록
        # 복합 보상은 에피소드 시작 시점에 정규화 기준을 잡아야 한다
        if self.reward_fn is not None and hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset(self)
        return self._obs()

    def step(self, action: torch.Tensor):
        # 이동 규칙은 src/env/motion.py 한 곳에만 둔다 — mcts(수읽기)와
        # visualize_search(화면 투영)가 같은 함수를 쓴다. 세 곳이 어긋나면
        # 화면·수읽기·실제 이동이 서로 다른 곳을 가리킨다.
        # 배터리 하드 제약(남은 비율만큼만 이동)도 그 안에 있다 — 샘플러는
        # log-prob 역변환을 위해 '정규 공간' 행동만 다뤄야 하므로 여기서 적용.
        d = self.drones[0]
        lon, lat = motion.step_position(
            d["lon"], d["lat"], action.detach().cpu().numpy(), self.battery,
            self.cfg.max_battery, self.cfg.max_step_frac, self.extent,
            clip=True)
        lon, lat = float(lon), float(lat)
        # clip=True 는 마스킹을 안 쓸 때의 안전망이다. 마스킹을 켜면 애초에
        # 경계 밖 행동이 샘플되지 않으므로 여기서 잘릴 일이 없다.
        d["lon"], d["lat"] = lon, lat
        self.traj.append((lon, lat))

        # 갱신 전 belief 를 보관 — RewardFn 이 before/after 차이를 봐야 한다
        self.mean_prev, self.var_prev = self.mean, self.var
        var_before = float(self.var_prev[self.land].sum())

        # 드론이 그 위치의 '진짜 값'을 측정 -> 관측점에 추가
        measured = self._ground_truth(lon, lat)
        self.obs_lon.append(lon); self.obs_lat.append(lat); self.obs_val.append(measured)
        self.mean, self.var = self._krige(
            np.array(self.obs_lon), np.array(self.obs_lat), np.array(self.obs_val))
        var_after = float(self.var[self.land].sum())

        # 정보이득 = 초기 대비 육지 분산 총합 감소 비율 (reward_fn 있으면 대체)
        if self.reward_fn is None:
            reward = (var_before - var_after) / self.var0
        else:
            reward = self.reward_fn(self, action)

        self.battery -= 1.0
        done = self.battery <= 0
        return self._obs(), reward, done

    # ---------------- 내부 ----------------
    def _krige(self, lon, lat, val):
        ok = OrdinaryKriging(lon, lat, val, variogram_model="spherical",
                             verbose=False, enable_plotting=False,
                             pseudo_inv=True)   # 중복 좌표 -> 특이행렬 방지
        z, ss = ok.execute("grid", self.gx, self.gy)          # [res, res]
        return np.asarray(z), np.asarray(ss)

    def _ground_truth(self, lon, lat) -> float:
        i = int(np.argmin(np.abs(self.gy - lat)))
        j = int(np.argmin(np.abs(self.gx - lon)))
        return float(self.gt_mean[i, j])

    def _land_center(self):
        ys, xs = np.where(self.land)
        return float(self.gx[int(round(xs.mean()))]), float(self.gy[int(round(ys.mean()))])

    def _random_land_point(self):
        """무작위 시작 지점. flight_mask 가 있으면 본토(최대 연결 성분)로 제한.

        바다 횡단 금지 하에서는 시작한 덩어리를 벗어날 수 없으므로, 섬에서
        시작하면 그 에피소드는 사실상 무의미해진다.
        """
        if self._start_pool is None:
            ys, xs = np.where(self.land)
            if self.flight_mask is not None:
                main = self.flight_mask.mainland_mask()
                ii, jj = self.flight_mask._index(self.gx[xs], self.gy[ys])
                keep = main[ii, jj]                    # 벡터화 — 에피소드마다 도는 코드
                if keep.any():
                    ys, xs = ys[keep], xs[keep]
                    print(f"[시작위치] 본토 육지칸 {keep.sum()}/{len(keep)} 로 제한")
            self._start_pool = (ys, xs)                # 한 번만 계산해 재사용
        ys, xs = self._start_pool
        k = self.rng.integers(len(xs))
        return float(self.gx[xs[k]]), float(self.gy[ys[k]])

    def _resample(self, grid: np.ndarray) -> np.ndarray:
        """[res,res] -> [grid_size,grid_size] 최근접 리샘플 (CNN 입력용)."""
        G = self.cfg.grid_size
        ii = np.linspace(0, grid.shape[0] - 1, G).round().astype(int)
        jj = np.linspace(0, grid.shape[1] - 1, G).round().astype(int)
        return grid[np.ix_(ii, jj)]

    def _norm_lon(self, lon):
        return (np.asarray(lon) - self.canvas.min_lon) / (
            self.canvas.max_lon - self.canvas.min_lon + 1e-9)

    def _norm_lat(self, lat):
        return (np.asarray(lat) - self.canvas.min_lat) / (
            self.canvas.max_lat - self.canvas.min_lat + 1e-9)

    def _obs(self) -> Observation:
        pts = np.stack([self._norm_lon(self.obs_lon),
                        self._norm_lat(self.obs_lat),
                        np.asarray(self.obs_val) / self.value_scale], axis=-1)
        pts = torch.tensor(pts, dtype=torch.float32)                  # [N, 3]

        mean = np.nan_to_num(self._resample(self.mean)) / self.value_scale
        var = np.nan_to_num(self._resample(self.var))
        var = var / (var.max() + 1e-9)
        grid = torch.tensor(np.stack([mean, var]), dtype=torch.float32)  # [2, G, G]

        d = self.drones[0]
        drone = torch.tensor([float(self._norm_lon(d["lon"])),
                              float(self._norm_lat(d["lat"])),
                              self.battery / self.cfg.max_battery],
                             dtype=torch.float32)
        return Observation(
            points=pts.unsqueeze(0),
            mask=torch.ones(1, pts.shape[0]),
            belief_grid=grid.unsqueeze(0),
            drone=drone.unsqueeze(0),
        )
