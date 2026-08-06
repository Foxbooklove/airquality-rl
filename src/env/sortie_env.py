"""출격 환경 — 시간이 흐르는 격자 위에서 국지 고농도 사건을 찾는다.

목적함수 (측정으로 확정된 것):
  도달 가능한 국지 고농도 사건(PM10 >= 80, 전국평균 보통, 3시간 이상 지속)을
  제한된 출격 예산으로 찾아낸다. 근거는 src/env/events.py 주석 참고.

핵심 설계 — 정직한 정답:
  우리는 농도장의 정답을 모른다. 601개 점의 실측만 있고 그 사이는 우리 보간이다.
  그래서 드론이 아무 칸에나 가면 **우리가 만든 값을 되읽을 뿐** 아무것도 발견하지
  못한다. 이 순환을 끊으려고 측정소를 둘로 나눈다.

    가시(visible)  belief 를 만드는 데 쓴다. 매시간 실제로 보고한다.
    은닉(hidden)   belief 에 없다. 드론이 그 칸에 가야만 **실측값**을 읽는다.

  드론이 은닉 측정소가 없는 칸에 가면 관측이 없다(보상 0, belief 갱신 없음).
  가짜 관측을 만들어 넣지 않는다. 그래서 문제가 정확히
  "고정 측정망이 놓치는 지점 중 어디를 보강할 것인가" 가 된다.

시간이 흐른다:
  이동 거리 / 속도 만큼 시각이 진행하고, 그때마다 가시 측정소가 새 값을 보고해
  belief 가 갱신된다. 사건이 3~25시간 지속이므로 정지 화면이면 정의가 무의미하다.

행동 공간이 하나로 통일된다:
  플랫폼 선택도 '격자 칸 고르기', 비행 목표도 '격자 칸 고르기'. 다른 건 마스크뿐.
  귀환도 마스크에서 나온다 — 유효 칸 = {c : d(현재,c)+d(c,기지) <= 배터리}.
  배터리가 줄면 집합이 기지 쪽으로 수축하므로 '귀환 모드' 분기가 없다.

variogram 을 고정한다:
  pykrige 는 관측을 추가할 때마다 variogram 을 재추정하는데, 그 변화가 belief 를
  통째로 밀어버린다(측정: 같은 상태에서 후보 순위 상관 r=0.04, 정보이득이 음수로
  나오는 경우도 발생). 에피소드 시작 시 한 번 추정해 고정한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pykrige.ok import OrdinaryKriging

from src.agent.config import ModelConfig
from src.env.dummy_env import Observation

PHASE_LAUNCH = 0
PHASE_FLY = 1
KM_PER_DEG_LAT = 111.0


@dataclass
class SortieSpec:
    pollutant: str = "PM10"
    threshold: float = 80.0        # 사건 임계 (예보 '나쁨' 81 에 준함)
    hidden_ratio: float = 0.35     # 은닉 측정소 비율
    battery_h: float = 12.0        # 드론 1대의 비행 가능 **시간**
    """배터리를 거리가 아니라 시간으로 센다.

    거리 기준이면 짧은 이동이 싸서, 제자리에 있고 싶을 때 두 칸 사이를 왕복하는
    게 최적이 된다(실측: 학습된 정책이 18.5km 왕복을 9시간 반복, 탐지 0).
    시간 기준이면 제자리든 45km 이동이든 1시간이면 같은 값을 쓴다. 실제 드론
    배터리도 거리가 아니라 체공 시간으로 표시되므로 물리적으로도 맞다.
    """
    speed_kmh: float = 70.0        # 순항 속도 — 이동 시간 계산
    max_hop_km: float = 45.0       # 한 수의 최대 이동거리.
    """제한하지 않으면 배터리를 한두 번의 큰 점프로 다 써서 에피소드가 10스텝에
    끝난다(측정). 걸음을 잘게 나눠야 방문 지점이 늘고 수읽기도 의미가 생긴다."""
    n_drones: int = 3
    event_prob: float = 0.7        # 에피소드를 사건 시각에서 시작할 확률
    detect_radius_km: float = 20.0
    """격자 간격(전국 res=36 에서 약 19km)보다 작으면 대부분의 칸에서 측정이
    아예 불가능하다. 반경 8km 일 때 육지칸의 27%, 20km 면 약 70% 에서 측정된다."""
    w_detect: float = 1.0          # 초과 발견 보상 (희소)
    w_dense: float = 0.15          # 측정값 크기 보상 (조밀, 그래디언트용)


class SortieEnv:
    def __init__(self, canvas, table, coords, cfg: ModelConfig, flight_mask,
                 events: pd.DataFrame, spec: SortieSpec = SortieSpec(), seed: int = 0):
        """
        table   : [시각 × 측정소] DataFrame (preprocess 결과를 피벗한 것)
        coords  : 측정소별 lat/lon DataFrame (table 컬럼 순서와 동일)
        events  : events.load_or_build() 결과
        """
        self.canvas, self.cfg, self.spec = canvas, cfg, spec
        self.fmask = flight_mask
        self.rng = np.random.default_rng(seed)

        self.T = table.index
        self.A = table.to_numpy(dtype=float)          # [시각, 측정소]
        self.names = list(table.columns)
        self.slat = coords.lat.to_numpy()
        self.slon = coords.lon.to_numpy()
        self.latm = np.deg2rad(float(np.nanmean(self.slat)))
        self.sxy = np.c_[self.slon * KM_PER_DEG_LAT * np.cos(self.latm),
                         self.slat * KM_PER_DEG_LAT]

        self.gx = canvas.grid_lon[0, :]
        self.gy = canvas.grid_lat[:, 0]
        self.res = len(self.gx)
        self.land = np.asarray(canvas.land_mask, dtype=bool)
        self.extent = (canvas.min_lon, canvas.max_lon, canvas.min_lat, canvas.max_lat)
        self._GL, self._GA = np.meshgrid(self.gx, self.gy)
        # 격자 칸의 km 좌표 (거리·배터리 계산은 전부 km 로 한다)
        self._GX = self._GL * KM_PER_DEG_LAT * np.cos(self.latm)
        self._GY = self._GA * KM_PER_DEG_LAT

        # 사건: 시각 인덱스로 미리 변환해 둔다
        self.events = events.reset_index(drop=True)
        pos = {t: i for i, t in enumerate(self.T)}
        self._ev_start = np.array([pos.get(t, -1) for t in self.events.start])
        self._ev_end = np.array([pos.get(t, -1) for t in self.events.end])
        ok = (self._ev_start >= 0) & (self._ev_end >= 0)
        self.events = self.events[ok].reset_index(drop=True)
        self._ev_start, self._ev_end = self._ev_start[ok], self._ev_end[ok]

        # 플랫폼 후보 = 본토 육지의 측정소 칸
        main = flight_mask.mainland_mask()
        self._si = np.clip(np.rint((self.slat - canvas.min_lat)
                                   / (self.gy[1] - self.gy[0])), 0, self.res - 1).astype(int)
        self._sj = np.clip(np.rint((self.slon - canvas.min_lon)
                                   / (self.gx[1] - self.gx[0])), 0, self.res - 1).astype(int)
        plat = np.zeros((self.res, self.res), bool)
        for i, j in zip(self._si, self._sj):
            if self.land[i, j] and main[flight_mask._index(self.gx[j], self.gy[i])]:
                plat[i, j] = True
        self.platform_mask = plat
        if not plat.any():
            raise ValueError("플랫폼 후보가 없습니다")

    # ------------------------------------------------------------ 에피소드
    def reset(self) -> Observation:
        s = self.spec
        # --- 시작 시각: 사건 중심 샘플링 (일부는 무사건 시각) ---
        self.event_id = -1
        if len(self.events) and self.rng.random() < s.event_prob:
            k = int(self.rng.integers(len(self.events)))
            self.event_id = k
            # 사건 시작 직전~초반에서 출발 (도달할 시간을 준다)
            self.t = max(0, int(self._ev_start[k]) - int(self.rng.integers(0, 3)))
        else:
            self.t = int(self.rng.integers(0, len(self.T) - 48))
        self.t0 = self.t

        # --- 가시/은닉 분할: 매 에피소드 새로 뽑는다 ---
        # __init__ 에 두면 '어디가 빈 곳인가' 패턴이 하나뿐이라 일반화를 못 본다.
        n = len(self.names)
        perm = self.rng.permutation(n)
        n_hid = int(round(n * s.hidden_ratio))
        self.hidden = np.sort(perm[:n_hid])
        self.visible = np.sort(perm[n_hid:])

        self.phase = PHASE_LAUNCH
        self.home = self.pos = None
        self.battery = 0.0
        self.drone_i = 0
        self.traj, self.sorties = [], []
        self.found = []                    # 탐지한 (측정소idx, 시각idx, 값)
        self.visited_hidden = set()        # 통계용 (보상 판정에는 안 씀)
        self.measured = set()              # (측정소idx, 시각idx) — 보상 중복 방지
        self.last_seen = {}                # 측정소idx -> (시각, 값). 관측 기억용
        self.prev_pos_km = None

        self._fit_variogram()
        self._update_belief()
        self.reward_terms = {"detect": 0.0, "dense": 0.0}
        return self._obs()

    # ------------------------------------------------------------ 행동
    def action_mask(self) -> np.ndarray:
        if self.phase == PHASE_LAUNCH:
            return self.platform_mask.copy()
        x0, y0 = self.pos_km
        hx, hy = self.home_km
        d_go = np.hypot(self._GX - x0, self._GY - y0)
        d_back = np.hypot(self._GX - hx, self._GY - hy)
        # 한 수는 최소 1시간. 제자리(d_go=0)도 1시간치 배터리를 쓴다 —
        # 실제 드론도 같은 자리에 떠 있으려면 동력이 든다.
        h_go = np.maximum(1.0, d_go / self.spec.speed_kmh)
        h_back = d_back / self.spec.speed_kmh
        ok = (self.land & ((h_go + h_back) <= self.battery + 1e-9)
              & (d_go <= self.spec.max_hop_km))
        if not ok.any():
            return ok
        ii, jj = np.where(ok)
        clear = self.fmask.path_clear(
            np.full(len(ii), self.pos[0]), np.full(len(ii), self.pos[1]),
            self.gx[jj], self.gy[ii])
        out = np.zeros_like(ok)
        out[ii[clear], jj[clear]] = True
        return out

    def step(self, action):
        i, j = self._decode(action)
        lon, lat = float(self.gx[j]), float(self.gy[i])
        s = self.spec

        if self.phase == PHASE_LAUNCH:
            self.home = self.pos = (lon, lat)
            self.home_km = self.pos_km = (float(self._GX[i, j]), float(self._GY[i, j]))
            self.battery = s.battery_h
            self.phase, self.drone_i = PHASE_FLY, 1
            self.traj.append((lon, lat, 1))
            self.sorties.append([(lon, lat)])
            return self._obs(), 0.0, False       # 즉시 보상 0 — 가치가 전부 미래에 있다

        # --- 이동(또는 제자리): 걸린 시간만큼 배터리를 쓰고 시각이 흐른다 ---
        nx, ny = float(self._GX[i, j]), float(self._GY[i, j])
        d_km = float(np.hypot(nx - self.pos_km[0], ny - self.pos_km[1]))
        dh = max(1.0, d_km / s.speed_kmh)          # 최소 1시간
        self.battery = max(0.0, self.battery - dh)
        self.prev_pos_km = self.pos_km
        self.pos, self.pos_km = (lon, lat), (nx, ny)
        self.t = min(len(self.T) - 1, self.t + int(round(dh)))
        self.traj.append((lon, lat, self.drone_i))
        self.sorties[-1].append((lon, lat))

        # --- 측정: 은닉 측정소가 반경 안에 있을 때만 실측을 얻는다 ---
        # 보상은 **(측정소, 시각)** 단위다. 같은 지점을 다시 재도 시각이 다르면
        # 새 관측이다 — 13->19->...->100 으로 오르는 걸 지켜보다 잡는 것이
        # 유효한 전략이어야 한다. 예전엔 지점 단위라 한 번 가면 그 지점이 영원히
        # 죽었고, 그래서 '기다렸다 잡기' 가 원천 봉쇄돼 있었다.
        reward = 0.0
        d_hid = np.hypot(self.sxy[self.hidden, 0] - nx, self.sxy[self.hidden, 1] - ny)
        near = np.where(d_hid <= s.detect_radius_km)[0]
        for h in near:
            gi = int(self.hidden[h])
            if (gi, self.t) in self.measured:
                continue
            val = self.A[self.t, gi]
            if not np.isfinite(val):
                continue
            self.measured.add((gi, self.t))
            self.visited_hidden.add(gi)
            self.last_seen[gi] = (self.t, float(val))
            r = s.w_dense * float(np.clip(val / s.threshold, 0.0, 2.0))
            self.reward_terms["dense"] += r
            if float(val) >= s.threshold:
                r += s.w_detect
                self.reward_terms["detect"] += s.w_detect
                self.found.append((gi, self.t, float(val)))
            reward += r

        self._update_belief()

        # --- 갈 곳이 없으면 귀환하고 다음 드론 ---
        done = False
        if not self.action_mask().any():
            self.sorties[-1].append(self.home)
            if self.drone_i >= s.n_drones:
                done = True
            else:
                self.drone_i += 1
                self.pos, self.pos_km = self.home, self.home_km
                self.battery = s.battery_h
                self.sorties.append([self.home])
                self.traj.append((self.home[0], self.home[1], self.drone_i))
        return self._obs(), reward, done

    # ------------------------------------------------------------ belief
    def _fit_variogram(self):
        """에피소드 시작 시 한 번만 추정하고 고정한다."""
        row = self.A[self.t]
        idx = self.visible[np.isfinite(row[self.visible])]
        if len(idx) < 10:
            idx = np.where(np.isfinite(row))[0]
        ok = OrdinaryKriging(self.slon[idx], self.slat[idx], row[idx],
                             variogram_model="spherical", verbose=False,
                             enable_plotting=False, pseudo_inv=True)
        self.vparams = list(ok.variogram_model_parameters)

    def _update_belief(self):
        """현재 시각의 가시 측정소 + 드론이 발견한 실측으로 belief 재계산."""
        row = self.A[self.t]
        idx = self.visible[np.isfinite(row[self.visible])]
        lon, lat, val = list(self.slon[idx]), list(self.slat[idx]), list(row[idx])
        for gi, (ts, v) in self.last_seen.items():   # 드론이 얻은 실측(지점별 최신)
            lon.append(self.slon[gi]); lat.append(self.slat[gi]); val.append(v)
        ok = OrdinaryKriging(np.array(lon), np.array(lat), np.array(val),
                             variogram_model="spherical",
                             variogram_parameters=self.vparams,
                             verbose=False, enable_plotting=False, pseudo_inv=True)
        z, ss = ok.execute("grid", self.gx, self.gy)
        self.mean, self.var = np.asarray(z), np.asarray(ss)

    # ------------------------------------------------------------ 유틸
    def _decode(self, action):
        if isinstance(action, (tuple, list)) and len(action) == 2:
            return int(action[0]), int(action[1])
        a = int(action.item() if torch.is_tensor(action) else action)
        return a // self.res, a % self.res

    def truth_at_hidden(self):
        """평가용 — 현재 시각 은닉 측정소의 실측값과 초과 여부."""
        row = self.A[self.t, self.hidden]
        return row, row >= self.spec.threshold

    def _obs(self) -> Observation:
        G = self.cfg.grid_size
        ii = np.linspace(0, self.res - 1, G).round().astype(int)
        jj = np.linspace(0, self.res - 1, G).round().astype(int)
        mean = np.nan_to_num(self.mean[np.ix_(ii, jj)]) / self.spec.threshold
        var = np.nan_to_num(self.var[np.ix_(ii, jj)])
        var = var / (var.max() + 1e-9)
        grid = torch.tensor(np.stack([mean, var]), dtype=torch.float32)

        row = self.A[self.t]
        vi = self.visible[np.isfinite(row[self.visible])]
        pts = np.stack([(self.slon[vi] - self.extent[0]) / (self.extent[1] - self.extent[0]),
                        (self.slat[vi] - self.extent[2]) / (self.extent[3] - self.extent[2]),
                        row[vi] / self.spec.threshold], axis=-1)
        pts = torch.tensor(pts, dtype=torch.float32)

        if self.pos is None:
            dx = dy = 0.5; bat = 0.0; hx = hy = 0.5
        else:
            dx = (self.pos[0] - self.extent[0]) / (self.extent[1] - self.extent[0])
            dy = (self.pos[1] - self.extent[2]) / (self.extent[3] - self.extent[2])
            hx = (self.home[0] - self.extent[0]) / (self.extent[1] - self.extent[0])
            hy = (self.home[1] - self.extent[2]) / (self.extent[3] - self.extent[2])
            bat = self.battery / self.spec.battery_h
        o = Observation(points=pts.unsqueeze(0), mask=torch.ones(1, pts.shape[0]),
                        belief_grid=grid.unsqueeze(0),
                        drone=torch.tensor([dx, dy, bat], dtype=torch.float32).unsqueeze(0))
        o.home = torch.tensor([hx, hy], dtype=torch.float32).unsqueeze(0)

        # --- 기억 평면: 어디를 얼마나 오래전에 쟀고 무엇을 읽었나 ---
        # 이게 없으면 문제가 마르코프가 아니다. 같은 지점을 다시 재는 게 유효한
        # 전략인데(값이 오르면 그때 잡는다), 언제 뭘 쟀는지 모르면 판단할 수 없다.
        rec = np.zeros((self.res, self.res), dtype=np.float32)
        lastv = np.zeros((self.res, self.res), dtype=np.float32)
        for gi, (ts, v) in self.last_seen.items():
            gi_i, gi_j = self._si[gi], self._sj[gi]
            age = max(0, self.t - ts)
            rec[gi_i, gi_j] = float(np.exp(-age / 6.0))      # 최근일수록 1
            lastv[gi_i, gi_j] = float(np.clip(v / self.spec.threshold, 0, 2))
        mem = torch.tensor(np.stack([rec[np.ix_(ii, jj)], lastv[np.ix_(ii, jj)]]),
                           dtype=torch.float32)
        o.memory = mem.unsqueeze(0)

        # --- 직전 위치 (왕복을 스스로 알아채게) ---
        if self.prev_pos_km is None:
            px, py = dx, dy
        else:
            px = (self.prev_pos_km[0] / (KM_PER_DEG_LAT * np.cos(self.latm))
                  - self.extent[0]) / (self.extent[1] - self.extent[0])
            py = (self.prev_pos_km[1] / KM_PER_DEG_LAT
                  - self.extent[2]) / (self.extent[3] - self.extent[2])
        o.prev = torch.tensor([float(px), float(py)], dtype=torch.float32).unsqueeze(0)

        # --- 시각: 하루 주기 + 에피소드 경과 ---
        hh = self.T[self.t].hour
        o.time = torch.tensor([np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24),
                               np.clip((self.t - self.t0) / 48.0, 0, 1)],
                              dtype=torch.float32).unsqueeze(0)
        return o


def make_table(year: int = 2024, pollutant: str = "PM10", completeness: float = 0.9):
    """[시각 × 측정소] 테이블과 좌표. 환경이 매번 피벗하지 않도록 미리 만든다."""
    from src.data.preprocess import AirQualityPreprocessor
    df = AirQualityPreprocessor(year=year).load().dropna(subset=["lat", "lon", pollutant])
    piv = df.pivot_table(index="datetime", columns="station", values=pollutant,
                         aggfunc="mean").sort_index()
    coord = df.groupby("station")[["lat", "lon"]].first()
    keep = [c for c in piv.columns if piv[c].notna().mean() > completeness]
    return piv[keep], coord.loc[keep]
