"""출격(sortie) 환경 — 격자 위 이산 행동, 플랫폼 선택, 배터리·귀환 제약.

기존 air_quality_env.AirQualityEnv 와의 차이:
  행동    연속 (theta, dist)  ->  격자 칸 하나 고르기 (카테고리컬)
  시작    무작위 육지 지점    ->  **정책이 측정소 중 하나를 플랫폼으로 선택**
  배터리  스텝마다 1 소모     ->  이동 거리만큼 소모, 소진 시 기지 귀환
  대수    1대                 ->  N대를 순차 출격 (belief 는 누적)

왜 이 구조인가:
  귀환 제약이 탐욕을 무너뜨린다. 근시안 정책은 이득이 큰 칸으로 직진하다
  돌아올 몫을 남기지 못한다. 귀환 제약이 붙은 orienteering 은 NP-hard 이고,
  순수 센서 배치(탐욕이 (1-1/e) 근사를 보장)와 달리 수읽기가 실제로 일한다.

행동 공간이 하나로 통일된다:
  플랫폼 선택도 '격자 칸 고르기', 비행 목표도 '격자 칸 고르기'. 다른 건
  **마스크뿐**이다. 첫 수엔 측정소가 있는 칸만, 이후엔 도달 가능한 칸만 연다.
  덕분에 정책 헤드가 하나로 끝난다.

귀환은 코드가 아니라 마스크에서 나온다:
  유효 칸 = { c : d(현재,c) + d(c,기지) <= 배터리, 육지, 경로가 바다 안 건넘 }
  배터리가 줄면 이 집합이 기지 쪽으로 저절로 수축한다. '귀환 모드' 분기가 없다.

첫 수는 즉시 보상이 0이다 — 측정소 값은 이미 아는 값이라 새 정보가 없다.
근시안 정책은 플랫폼을 고를 근거가 아예 없고, 가치가 전부 미래에 있다.
RL 이 이겨야 할 지점이 여기다.
"""
from __future__ import annotations

import numpy as np
import torch
from pykrige.ok import OrdinaryKriging

from src.agent.config import ModelConfig
from src.env.dummy_env import Observation, RewardFn

# 행동 단계
PHASE_LAUNCH = 0    # 플랫폼(측정소 칸) 선택 — 에피소드당 1회
PHASE_FLY = 1       # 비행 목표 선택


class SortieEnv:
    def __init__(self, canvas, snapshot, cfg: ModelConfig, flight_mask,
                 pollutant: str = "PM10", visible_ratio: float = 0.4,
                 value_scale: float = 100.0, reward_fn: RewardFn | None = None,
                 n_drones: int = 5, battery: float | None = None,
                 bbox_pad: float | None = None, seed: int = 0):
        """
        canvas       : Canvas (격자 + 육지 마스크)
        flight_mask  : motion.FlightMask — 경로 검사용 고해상 마스크 (필수)
        n_drones     : 순차 출격할 드론 수
        battery      : 드론 1대의 이동 예산(거리, 캔버스 대각선 대비 비율).
                       None 이면 cfg.max_battery * cfg.max_step_frac 로 환산
        """
        self.canvas = canvas
        self.cfg = cfg
        self.fmask = flight_mask
        self.pollutant = pollutant
        self.value_scale = value_scale
        self.reward_fn = reward_fn
        self.n_drones = int(n_drones)
        self.rng = np.random.default_rng(seed)

        self.gx = canvas.grid_lon[0, :]
        self.gy = canvas.grid_lat[:, 0]
        self.res = len(self.gx)
        self.land = np.asarray(canvas.land_mask, dtype=bool)
        self.extent = (canvas.min_lon, canvas.max_lon,
                       canvas.min_lat, canvas.max_lat)
        self.span = float(np.hypot(canvas.max_lon - canvas.min_lon,
                                   canvas.max_lat - canvas.min_lat))
        # 배터리를 '도(deg) 단위 이동 거리 예산'으로 환산해 둔다
        self.max_battery = float(
            battery if battery is not None
            else cfg.max_battery * cfg.max_step_frac * self.span)

        # --- 측정소 ---
        s = snapshot.dropna(subset=[pollutant]).reset_index(drop=True)
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

        # --- 격자 좌표 캐시 (마스크 계산에 매번 쓰임) ---
        self._GL, self._GA = np.meshgrid(self.gx, self.gy)      # [res, res]
        self._land_flat = self.land.ravel()

        # --- 플랫폼 후보: 측정소가 들어있는 육지 칸 ---
        # 본토 연결 성분으로 제한한다. 바다 횡단 금지라 섬에서 출격하면 갇힌다.
        main = flight_mask.mainland_mask()
        pj = np.clip(np.rint((self.st_lon - canvas.min_lon)
                             / (self.gx[1] - self.gx[0])), 0, self.res - 1).astype(int)
        pi = np.clip(np.rint((self.st_lat - canvas.min_lat)
                             / (self.gy[1] - self.gy[0])), 0, self.res - 1).astype(int)
        plat = np.zeros((self.res, self.res), dtype=bool)
        for i, j in zip(pi, pj):
            if self.land[i, j] and main[flight_mask._index(self.gx[j], self.gy[i])]:
                plat[i, j] = True
        self.platform_mask = plat
        if not plat.any():
            raise ValueError("플랫폼 후보(측정소가 있는 본토 육지 칸)가 없습니다")

        self.visible_ratio = visible_ratio

    # ---------------- 표준 RL 인터페이스 ----------------
    def reset(self) -> Observation:
        n = len(self.st_val)
        k = max(4, int(round(n * self.visible_ratio)))
        self.visible_idx = self.rng.choice(n, size=k, replace=False)
        self.obs_lon = list(self.st_lon[self.visible_idx])
        self.obs_lat = list(self.st_lat[self.visible_idx])
        self.obs_val = list(self.st_val[self.visible_idx])

        self.phase = PHASE_LAUNCH
        self.home = None                 # (lon, lat) 플랫폼
        self.pos = None                  # 현재 드론 위치
        self.battery = 0.0
        self.drone_i = 0                 # 몇 번째 드론인지
        self.traj = []                   # [(lon, lat, drone_i)] 궤적 기록
        self.sorties = []                # 드론별 경로

        self.mean, self.var = self._krige(
            np.array(self.obs_lon), np.array(self.obs_lat), np.array(self.obs_val))
        self.var0 = max(float(self.var[self.land].sum()), 1e-9)
        self.mean_prev, self.var_prev = self.mean.copy(), self.var.copy()
        if self.reward_fn is not None and hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset(self)
        return self._obs()

    def action_mask(self) -> np.ndarray:
        """[res, res] bool — 지금 고를 수 있는 칸.

        첫 수: 측정소가 있는 본토 칸(플랫폼 후보).
        이후: 육지 + 경로가 바다를 안 건넘 + **왕복 가능**
              d(현재,c) + d(c,기지) <= 남은 배터리
              배터리가 줄면 이 집합이 기지 쪽으로 수축한다 = 귀환이 자동으로 나온다.
        """
        if self.phase == PHASE_LAUNCH:
            return self.platform_mask.copy()

        lon0, lat0 = self.pos
        hlon, hlat = self.home
        d_go = np.hypot(self._GL - lon0, self._GA - lat0)
        d_back = np.hypot(self._GL - hlon, self._GA - hlat)
        ok = self.land & ((d_go + d_back) <= self.battery + 1e-12)
        ok &= (d_go > 1e-12)                      # 제자리는 새 정보가 없다
        if not ok.any():
            return ok
        # 경로 검사는 후보가 정해진 뒤에만 (비싸다)
        ii, jj = np.where(ok)
        clear = self.fmask.path_clear(
            np.full(len(ii), lon0), np.full(len(ii), lat0),
            self.gx[jj], self.gy[ii])
        out = np.zeros_like(ok)
        out[ii[clear], jj[clear]] = True
        return out

    def step(self, action):
        """action: 격자 평탄 인덱스(int) 또는 (i, j)."""
        i, j = self._decode(action)
        lon, lat = float(self.gx[j]), float(self.gy[i])

        if self.phase == PHASE_LAUNCH:
            # 플랫폼 선택 — 이동도 측정도 없다. 즉시 보상 0.
            self.home = (lon, lat)
            self.pos = (lon, lat)
            self.battery = self.max_battery
            self.phase = PHASE_FLY
            self.drone_i = 1
            self.traj.append((lon, lat, self.drone_i))
            self.sorties.append([(lon, lat)])
            return self._obs(), 0.0, False

        # --- 비행: 이동 + 측정 ---
        d = float(np.hypot(lon - self.pos[0], lat - self.pos[1]))
        self.battery = max(0.0, self.battery - d)
        self.pos = (lon, lat)
        self.traj.append((lon, lat, self.drone_i))
        self.sorties[-1].append((lon, lat))

        self.mean_prev, self.var_prev = self.mean, self.var
        var_before = float(self.var_prev[self.land].sum())
        measured = self._ground_truth(lon, lat)
        self.obs_lon.append(lon); self.obs_lat.append(lat)
        self.obs_val.append(measured)
        self.mean, self.var = self._krige(
            np.array(self.obs_lon), np.array(self.obs_lat), np.array(self.obs_val))
        var_after = float(self.var[self.land].sum())

        if self.reward_fn is None:
            reward = (var_before - var_after) / self.var0
        else:
            reward = self.reward_fn(self, action)

        # --- 더 갈 곳이 없으면 귀환하고 다음 드론 ---
        done = False
        if not self.action_mask().any():
            self.sorties[-1].append(self.home)      # 기지로 복귀 (기록용)
            if self.drone_i >= self.n_drones:
                done = True
            else:
                self.drone_i += 1
                self.pos = self.home
                self.battery = self.max_battery
                self.sorties.append([self.home])
                self.traj.append((self.home[0], self.home[1], self.drone_i))
        return self._obs(), reward, done

    # ---------------- 내부 ----------------
    def _decode(self, action):
        if isinstance(action, (tuple, list, np.ndarray)) and len(action) == 2:
            return int(action[0]), int(action[1])
        a = int(action.item() if torch.is_tensor(action) else action)
        return a // self.res, a % self.res

    def _krige(self, lon, lat, val):
        ok = OrdinaryKriging(lon, lat, val, variogram_model="spherical",
                             verbose=False, enable_plotting=False,
                             pseudo_inv=True)
        z, ss = ok.execute("grid", self.gx, self.gy)
        return np.asarray(z), np.asarray(ss)

    def _ground_truth(self, lon, lat) -> float:
        i = int(np.argmin(np.abs(self.gy - lat)))
        j = int(np.argmin(np.abs(self.gx - lon)))
        return float(self.gt_mean[i, j])

    def _resample(self, grid: np.ndarray) -> np.ndarray:
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
        pts = torch.tensor(pts, dtype=torch.float32)

        mean = np.nan_to_num(self._resample(self.mean)) / self.value_scale
        var = np.nan_to_num(self._resample(self.var))
        var = var / (var.max() + 1e-9)
        grid = torch.tensor(np.stack([mean, var]), dtype=torch.float32)

        if self.pos is None:                     # 아직 플랫폼 선택 전
            dx = dy = 0.5
            bat = 0.0
        else:
            dx = float(self._norm_lon(self.pos[0]))
            dy = float(self._norm_lat(self.pos[1]))
            bat = self.battery / max(self.max_battery, 1e-9)
        drone = torch.tensor([dx, dy, bat], dtype=torch.float32)
        return Observation(points=pts.unsqueeze(0),
                           mask=torch.ones(1, pts.shape[0]),
                           belief_grid=grid.unsqueeze(0),
                           drone=drone.unsqueeze(0))
