"""실제 대기질 RL 환경 (프로토타입).

전이 구조(합의된 것):
  - 농도장: action 과 무관 (프로토타입은 스냅샷 고정 -> 시간진행 없음)
  - action: 드론 위치만 변경
  - 관측: 드론 위치가 '농도장의 어느 지점을 여느냐'만 결정

belief = Kriging(관측점들) 의 (mean, variance).
정보이득 reward = 드론이 한 점 측정 -> Kriging 재계산 -> 육지 칸 분산 총합 감소분.
(분산 감소는 측정값과 무관, 위치만으로 결정 — 그래서 정답 C 없이 계산됨)

상태는 '드론 축'을 열어둠(num_drones=1). belief 는 드론 수와 무관한 공유 자원.
나중에 다중 드론이면 drones 를 N개로 늘리고 obs 의 drone 축만 확장.

dummy_env 와 reset/step/Observation 인터페이스 동일 -> run 루프 안 바뀜.
"""
from __future__ import annotations

import numpy as np
import torch
from pykrige.ok import OrdinaryKriging

from src.agent.config import ModelConfig
from src.env.dummy_env import Observation, RewardFn


class AirQualityEnv:
    def __init__(self, canvas, snapshot, cfg: ModelConfig,
                 pollutant: str = "PM10", visible_ratio: float = 0.5,
                 value_scale: float = 100.0, reward_fn: RewardFn | None = None,
                 random_start: bool = True, seed: int = 0):
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

        # 전체 스냅샷 = 시뮬레이터의 '진짜 농도장' (드론 측정값 생성용)
        s = snapshot.dropna(subset=[pollutant]).reset_index(drop=True)
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
        self.traj = [(cx, cy)]                # PPT용 궤적 기록
        return self._obs()

    def step(self, action: torch.Tensor):
        theta = float(action[0]) * np.pi                       # [-1,1]->[-pi,pi]
        span = np.hypot(self.canvas.max_lon - self.canvas.min_lon,
                        self.canvas.max_lat - self.canvas.min_lat)
        dist = float(action[1]) * self.cfg.max_step_frac * span

        d = self.drones[0]
        lon = np.clip(d["lon"] + dist * np.cos(theta),
                      self.canvas.min_lon, self.canvas.max_lon)
        lat = np.clip(d["lat"] + dist * np.sin(theta),
                      self.canvas.min_lat, self.canvas.max_lat)
        d["lon"], d["lat"] = lon, lat
        self.traj.append((lon, lat))

        var_before = float(self.var[self.land].sum())

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
        ys, xs = np.where(self.land)
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
