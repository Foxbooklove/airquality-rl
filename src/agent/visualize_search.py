"""MCTS 수읽기 과정 실시간 시각화 (바둑 AI 중계 방식).

화면 구성:
  배경   — 현재 belief 의 불확실성(Kriging 분산)
  파랑   — 확정 경로. 무인기가 실제로 지나온 길 (지워지지 않음)
  주황   — 탐색 중인 가지. 리프까지 뻗었다가 backup 후 사라짐
  초록   — 탐색 결과 선택된 다음 한 수 (곧 파랑에 편입)

좌표 투영에 관하여:
  MCTS 노드는 '잠재상태'라 실제 위경도가 없다. 그래서 부모 위치에 그 노드로
  오는 행동 벡터를 이어붙여 지리적 위치를 추정한다(env.step 과 동일한 규칙:
  theta*pi, dist*max_step_frac*span*max_reach, 캔버스 clip).
  즉 화면의 주황 가지는 '이 행동을 계속 두면 무인기가 갈 자리'의 투영이다.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def use_korean_font():
    """matplotlib 기본 폰트엔 한글 글리프가 없어 라벨이 □□□ 로 깨진다.
    OS 별 기본 한글 폰트를 찾아 적용 (없으면 조용히 넘어감)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic",      # Windows
                 "AppleGothic",        # macOS
                 "NanumGothic", "Noto Sans CJK KR", "NanumBarunGothic"):  # Linux
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


BLUE = "#1B5E9E"     # 확정 경로
ORANGE = "#E8833A"   # 탐색 중
GREEN = "#2E9E5B"    # 선택된 수
RED = "#C0392B"      # 현재 무인기 위치


def project_path(nodes, action_idx, start_lonlat, battery,
                 max_step_frac, max_battery, extent):
    """루트 위치에서 시작해 트리 경로를 위경도 좌표열로 투영.

    마스킹을 켜면 Node 가 이미 정확한 pos 를 들고 있으므로 그걸 그대로 읽는다
    (투영이 아니라 실측). 마스킹이 없으면 motion 의 규칙으로 계산한다 —
    어느 쪽이든 env.step 과 같은 함수를 쓰므로 화면과 실제가 어긋나지 않는다.
    """
    from src.env.motion import step_position

    # Node.pos 가 채워져 있으면 그대로 (mcts 가 계산해 둔 정확한 좌표)
    if len(nodes) > 1 and getattr(nodes[1], "pos", None) is not None:
        return [tuple(n.pos) for n in nodes[:len(action_idx) + 1]]

    lon, lat = start_lonlat
    pts = [(lon, lat)]
    for d, idx in enumerate(action_idx):
        a = nodes[d].actions[idx]
        bat = max(battery - d, 0.0)     # 깊이만큼 배터리가 줄어든 상태
        lon, lat = step_position(lon, lat, np.asarray(a, dtype=float), bat,
                                 max_battery, max_step_frac, extent, clip=True)
        lon, lat = float(lon), float(lat)
        pts.append((lon, lat))
    return pts


class LiveSearchView:
    """실시간 창에 탐색 과정을 그린다.

    save_dir 를 주면 프레임을 파일로도 저장(창 없는 환경에서 확인용).
    """

    def __init__(self, extent, land_mask, max_step_frac, max_battery,
                 descend_pause=0.06, backup_pause=0.04, commit_pause=0.6,
                 interactive=True, save_dir=None, on_frame=None,
                 gate=None, speed=None,
                 figsize=(7.6, 7.2), title="MCTS 수읽기"):
        self.extent = extent
        self.land = np.asarray(land_mask, dtype=bool)
        self.max_step_frac = max_step_frac
        self.max_battery = max_battery
        self.descend_pause = descend_pause
        self.backup_pause = backup_pause
        self.commit_pause = commit_pause
        self.interactive = interactive
        self.save_dir = save_dir
        self.on_frame = on_frame        # 프레임(JPEG bytes)을 받아갈 콜백
        self.gate = gate                # 일시정지면 여기서 블록되는 함수
        self.speed = speed              # 속도 배율을 돌려주는 함수 (없으면 1.0)
        self._frame = 0

        use_korean_font()
        if interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.07)
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        # 위도가 올라가면 경도 1도의 실제 거리가 짧아진다. 보정 안 하면 옆으로 퍼져 보임.
        mid_lat = (extent[2] + extent[3]) / 2
        self._aspect = 1.0 / max(np.cos(np.radians(mid_lat)), 1e-6)
        self.ax.set_aspect(self._aspect, adjustable="box")
        self.ax.tick_params(labelsize=9)
        self.ax.set_xlabel("lon", fontsize=9); self.ax.set_ylabel("lat", fontsize=9)

        self._im = None
        self._confirmed_line, = self.ax.plot([], [], "-", color=BLUE, lw=2.6,
                                             zorder=4, label="확정 경로")
        self._drone, = self.ax.plot([], [], "o", color=RED, ms=11, mec="white",
                                    mew=1.5, zorder=6)
        self._search_artists = []     # 매 시뮬 후 지워지는 것들
        self._commit_artists = []     # 한 수마다 지워지는 것들
        self._info = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes,
                                  va="top", ha="left", fontsize=9,
                                  bbox=dict(boxstyle="round,pad=0.3",
                                            fc="white", alpha=0.8, lw=0))
        self._legend_done = False

    # ---------- 배경 / 확정 경로 ----------
    def set_belief(self, var):
        bg = np.array(var, dtype=float)
        bg[~self.land] = np.nan
        if self._im is None:
            self._im = self.ax.imshow(bg, origin="lower", cmap="viridis",
                                      extent=self.extent, aspect="auto",
                                      alpha=0.85, zorder=1)
            self.ax.set_aspect(self._aspect, adjustable="box")  # imshow 가 덮어씀
            cb = self.fig.colorbar(self._im, ax=self.ax, fraction=0.04, pad=0.02)
            cb.set_label("uncertainty", fontsize=9)
            cb.ax.tick_params(labelsize=8)
        else:
            self._im.set_data(bg)
            self._im.set_clim(np.nanmin(bg), np.nanmax(bg))

    def set_confirmed(self, traj):
        if traj:
            xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
            self._confirmed_line.set_data(xs, ys)
            self._drone.set_data([xs[-1]], [ys[-1]])

    def set_info(self, text):
        self._info.set_text(text)

    # ---------- 탐색 애니메이션 ----------
    def on_simulation(self, info, drone_lonlat, battery):
        """시뮬레이션 1회: 리프까지 뻗었다가 backup 하고 사라진다."""
        pts = project_path(info["nodes"], info["action_idx"], drone_lonlat,
                           battery, self.max_step_frac, self.max_battery,
                           self.extent)
        # --- 뻗어나가기 ---
        for i in range(len(pts) - 1):
            seg, = self.ax.plot([pts[i][0], pts[i + 1][0]],
                                [pts[i][1], pts[i + 1][1]],
                                "-", color=ORANGE, lw=1.6, alpha=0.9, zorder=3)
            dot, = self.ax.plot([pts[i + 1][0]], [pts[i + 1][1]], "o",
                                color=ORANGE, ms=4, zorder=3)
            self._search_artists += [seg, dot]
            self._pause(self.descend_pause)

        # --- 리프 도달 표시 ---
        leaf, = self.ax.plot([pts[-1][0]], [pts[-1][1]], "o", color=ORANGE,
                             ms=9, mfc="none", mew=1.8, zorder=3)
        self._search_artists.append(leaf)
        self._pause(self.descend_pause)

        # --- backup: 리프에서 루트로 되짚어 오며 반짝 ---
        for i in range(len(pts) - 2, -1, -1):
            fl, = self.ax.plot([pts[i][0], pts[i + 1][0]],
                               [pts[i][1], pts[i + 1][1]],
                               "-", color="white", lw=3.0, alpha=0.95, zorder=5)
            self._pause(self.backup_pause)
            fl.remove()
            self._pause(self.backup_pause * 0.4)

        self._clear_search()

    def commit(self, action, drone_lonlat, battery):
        """탐색 종료 — 선택된 한 수를 초록으로 잠시 보여준다."""
        pts = project_path([_Stub(action)], [0], drone_lonlat, battery,
                           self.max_step_frac, self.max_battery, self.extent)
        ln, = self.ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                           "-", color=GREEN, lw=3.4, zorder=5)
        dt, = self.ax.plot([pts[1][0]], [pts[1][1]], "o", color=GREEN, ms=8,
                           zorder=5)
        self._commit_artists += [ln, dt]
        self._pause(self.commit_pause)
        self._clear_commit()

    # ---------- 내부 ----------
    def _clear_search(self):
        for a in self._search_artists:
            a.remove()
        self._search_artists = []

    def _clear_commit(self):
        for a in self._commit_artists:
            a.remove()
        self._commit_artists = []

    def _pause(self, sec):
        # 프레임을 웹으로 흘려보내기 (figure 를 만든 이 스레드에서 렌더)
        if self.on_frame is not None:
            import io
            buf = io.BytesIO()
            self.fig.savefig(buf, format="jpg", dpi=78,
                             facecolor=self.fig.get_facecolor())
            self.on_frame(buf.getvalue())

        # 일시정지 상태면 여기서 대기 (프레임 경계라 그림이 안 깨짐)
        if self.gate is not None:
            self.gate()

        # 속도 배율 적용 (크면 빠르게)
        mult = 1.0
        if self.speed is not None:
            try:
                mult = max(float(self.speed()), 0.05)
            except Exception:
                mult = 1.0
        sec = sec / mult

        if self.interactive:
            plt.pause(max(sec, 1e-3))
        else:
            self.fig.canvas.draw()
            if sec > 0:
                import time
                time.sleep(sec)      # 웹 모드에선 plt.pause 가 없으니 직접 대기

        if self.save_dir:
            from pathlib import Path
            d = Path(self.save_dir); d.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(d / f"frame_{self._frame:04d}.png", dpi=80)
            self._frame += 1

    def finish(self, out_path=None):
        if out_path:
            self.fig.savefig(out_path, dpi=130, bbox_inches="tight")
        if self.interactive:
            plt.ioff()


class _Stub:
    """commit() 에서 단일 행동을 project_path 에 넘기기 위한 껍데기."""
    def __init__(self, action):
        self.actions = [action]