"""농도장의 시간 스케일 분석 — 시간 진행을 넣을 가치가 있는지 판단하는 근거.

왜 이걸 먼저 재는가:
  환경에 시간 진행을 넣으면 '관측이 낡는다'는 성질이 생기고, 그게 탐욕 알고리즘의
  이론적 보장(adaptive monotonicity)을 깨서 RL 도입 명분이 된다. 그런데 그건
  **필드가 에피소드 시간 안에 실제로 변할 때만** 성립한다. PM10 공간 패턴이
  이틀씩 유지되면 40시간 에피소드에서 낡음이 미미해서 복잡도만 늘고 이득이 없다.

  그래서 먼저 숫자를 본다: 얼마나 빨리 변하고, 얼마나 빨리 움직이는가.

여기서 나오는 결론이 정하는 것:
  - 시간 진행을 넣을 가치가 있는가          → 패턴 탈상관 시간 vs 에피소드 길이
  - dt(스텝당 시간)를 얼마로 할까            → 자기상관이 의미있게 떨어지는 lag
  - TTL(에피소드 길이)을 얼마로 할까          → 필드가 1~2회 뒤집히는 정도
  - '추적'이 가능한 전략인가                 → 오염 이동 속도 vs 드론 속도

실행 (레포 루트에서):
    python -m src.data.analyze_timescale
    python -m src.data.analyze_timescale --pollutant O3 --max-lag 72
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "exploration"

# 경보 발령 기준 (대기환경보전법 시행규칙 별표 7). 이벤트 후보 탐색용.
#   PM: 권역 평균 기준 + 2시간 지속. 여기선 권역 정의가 없어 시도 평균으로 근사.
#   O3: 1개소라도 초과 + 지속조건 없음 → 근사 없이 정확히 계산됨.
ALARM = {
    "PM10":  {"주의보": 150.0, "경보": 300.0, "지속시간": 2, "집계": "mean"},
    "PM25":  {"주의보": 75.0,  "경보": 150.0, "지속시간": 2, "집계": "mean"},
    "O3":    {"주의보": 0.12,  "경보": 0.30,  "지속시간": 1, "집계": "max"},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="농도장 시간 스케일 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pollutant", default="PM10", choices=list(ALARM))
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--max-lag", type=int, default=48, help="자기상관/시차상관 최대 lag(시간)")
    p.add_argument("--min-complete", type=float, default=0.9,
                   help="이 비율 이상 관측이 있는 측정소만 사용")
    p.add_argument("--step-frac", type=float, default=0.06,
                   help="드론 스텝당 이동거리 (캔버스 대각선 대비). train.py 기본값과 맞춤")
    p.add_argument("--season", default=None,
                   help="'5-9' 처럼 월 범위로 제한. 오존은 5/1~9/15 만 경보제 시행")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 데이터 준비
def load_matrix(args):
    """parquet → [시각 × 측정소] 행렬 + 측정소 좌표.

    Kriging 입력이 시각마다 달라지므로 결측 패턴도 같이 본다.
    """
    from src.data.preprocess import AirQualityPreprocessor
    pre = AirQualityPreprocessor(year=args.year)
    df = pre.load()

    pol = args.pollutant
    print(f"[데이터] 전체 {len(df):,}행, 컬럼 {df.columns.tolist()}")
    if "region" in df.columns:
        regs = df["region"].dropna().unique()
        print(f"         region 값 {len(regs)}종 예시: {list(regs[:5])}")

    df = df.dropna(subset=["lat", "lon"])
    if args.season:
        lo, hi = (int(x) for x in args.season.split("-"))
        df = df[df["datetime"].dt.month.between(lo, hi)]
        print(f"         {lo}~{hi}월로 제한 → {len(df):,}행")

    # [시각 × 측정소] 피벗
    mat = df.pivot_table(index="datetime", columns="station", values=pol,
                         aggfunc="mean").sort_index()
    # 시간축을 빈틈없는 1시간 간격으로 (누락 시각은 NaN 행으로 삽입)
    full_idx = pd.date_range(mat.index.min(), mat.index.max(), freq="h")
    mat = mat.reindex(full_idx)

    complete = mat.notna().mean()
    keep = complete[complete >= args.min_complete].index
    print(f"[결측] 측정소 {mat.shape[1]}개 중 관측률 {args.min_complete:.0%} 이상 "
          f"{len(keep)}개 사용")
    print(f"       시각별 살아있는 측정소: 평균 {mat.notna().sum(1).mean():.0f}개, "
          f"최소 {mat.notna().sum(1).min()}개")
    mat = mat[keep]

    coords = (df[df["station"].isin(keep)]
              .groupby("station")[["lat", "lon"]].first().reindex(keep))
    print(f"[행렬] {mat.shape[0]:,}시각 × {mat.shape[1]}측정소, "
          f"{pol} 범위 {np.nanmin(mat.values):.3g}~{np.nanmax(mat.values):.3g}")
    return mat, coords


# ---------------------------------------------------------------- ① 자기상관
def autocorrelation(mat, max_lag):
    """측정소별 시간 자기상관의 평균. '값이 얼마나 유지되나'."""
    X = mat.values.astype(float)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    X = X / np.where(sd > 0, sd, 1.0)

    out = []
    for lag in range(0, max_lag + 1):
        a, b = (X, X) if lag == 0 else (X[:-lag], X[lag:])
        ok = np.isfinite(a) & np.isfinite(b)
        num = np.nansum(np.where(ok, a * b, 0.0), axis=0)
        cnt = ok.sum(axis=0)
        r = num / np.maximum(cnt, 1)
        out.append(np.nanmean(r[cnt > 100]))
    return np.array(out)


# ------------------------------------------------- ② 공간 패턴 지속성 (핵심)
def pattern_correlation(mat, max_lag):
    """t 와 t+lag 의 '공간 패턴' 상관 (anomaly correlation).

    왜 자기상관과 따로 보는가:
      전국이 같이 오르내리면 자기상관은 높게 나오지만 **어디가 상대적으로 높은지는
      안 변한다.** 그러면 드론이 재방문할 이유가 약하다. 우리가 알고 싶은 건
      '고농도 지역의 위치'가 바뀌는 속도이므로, 각 시각의 전국 평균을 뺀 뒤
      패턴만 비교한다.
    """
    X = mat.values.astype(float)
    # 시각별 공간평균 제거 → 남는 건 '어디가 상대적으로 높은가'
    X = X - np.nanmean(X, axis=1, keepdims=True)

    out = []
    for lag in range(0, max_lag + 1):
        a, b = (X, X) if lag == 0 else (X[:-lag], X[lag:])
        ok = np.isfinite(a) & np.isfinite(b)
        rs = []
        for i in range(a.shape[0]):
            m = ok[i]
            if m.sum() < 30:
                continue
            u, v = a[i, m], b[i, m]
            su, sv = u.std(), v.std()
            if su > 0 and sv > 0:
                rs.append(float(((u - u.mean()) * (v - v.mean())).mean() / (su * sv)))
        out.append(np.mean(rs) if rs else np.nan)
    return np.array(out)


# ------------------------------------------------ ③ 시차 교차상관 → 이동 속도
def advection_speed(mat, coords, max_lag):
    """측정소 쌍마다 '최대 상관을 주는 시차'를 찾아 이동 속도를 추정.

    A에서 오른 것이 lag 시간 뒤 B에서 오르면, 오염이 A→B로 (거리/lag) 속도로
    이동한다는 뜻. 드론 속도와 비교해 '추적'이 가능한 전략인지 판단한다.

    구현: [S,T]@[T,S] 행렬곱으로 lag별 전체 쌍 상관을 한 번에. BLAS 가 처리하므로
    600측정소 × 8760시각 × 48lag 도 수 분이면 끝난다.
    """
    X = mat.values.astype(float)
    X = X - np.nanmean(X, axis=1, keepdims=True)          # 공간평균 제거
    X = X - np.nanmean(X, axis=0, keepdims=True)          # 측정소별 평균 제거
    sd = np.nanstd(X, axis=0, keepdims=True)
    X = np.nan_to_num(X / np.where(sd > 0, sd, 1.0))      # 표준화 + 결측 0

    S = X.shape[1]
    best_r = np.full((S, S), -np.inf)
    best_lag = np.zeros((S, S), dtype=int)
    for lag in range(0, max_lag + 1):
        a, b = (X, X) if lag == 0 else (X[:-lag], X[lag:])
        C = (a.T @ b) / a.shape[0]                        # [S,S] 상관 근사
        upd = C > best_r
        best_r[upd] = C[upd]
        best_lag[upd] = lag
        if lag % 12 == 0:
            print(f"       lag {lag:3d}/{max_lag} …")

    # 측정소 쌍 거리 (km) — 위경도 근사
    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()
    latm = np.deg2rad(lat.mean())
    dy = (lat[:, None] - lat[None, :]) * 111.0
    dx = (lon[:, None] - lon[None, :]) * 111.0 * np.cos(latm)
    dist = np.hypot(dx, dy)

    # 유의미한 쌍만: 상관이 충분히 높고, 시차가 0이 아니고(=이동), 너무 가깝지 않은
    m = (best_r > 0.5) & (best_lag > 0) & (dist > 20.0) & np.isfinite(dist)
    if not m.any():
        print("       [주의] 조건을 만족하는 측정소 쌍이 없음 — 이동 신호가 약함")
        return None
    speeds = dist[m] / best_lag[m]                        # km/h
    return dict(speeds=speeds, best_r=best_r, best_lag=best_lag, dist=dist,
                n_pairs=int(m.sum()))


# ---------------------------------------------------------------- ④ 이벤트 목록
def alarm_events(mat, args):
    """경보 발령 기준을 넘은 시각 목록. 시나리오 스냅샷 후보가 된다.

    지금 기본 스냅샷 '2024-07-01 13:00' 이 아무 일도 없던 평범한 시각일 가능성이
    크다. 경보 축을 실험하려면 실제로 고농도였던 시각을 골라야 한다.
    """
    spec = ALARM[args.pollutant]
    agg = mat.max(axis=1) if spec["집계"] == "max" else mat.mean(axis=1)
    thr, hold = spec["주의보"], spec["지속시간"]

    over = agg >= thr
    if hold > 1:      # 연속 hold 시간 지속
        over = over.rolling(hold).sum() >= hold
    ev = pd.DataFrame({"datetime": agg.index[over.fillna(False)],
                       "value": agg[over.fillna(False)].values})
    ev["단계"] = np.where(ev["value"] >= spec["경보"], "경보", "주의보")
    return ev, agg, spec


# ---------------------------------------------------------------- 요약 / 그림
def decorr_time(curve, level):
    """상관이 level 아래로 처음 떨어지는 lag. 없으면 None."""
    idx = np.where(curve < level)[0]
    return int(idx[0]) if len(idx) else None


def main(argv=None):
    args = parse_args(argv)
    # Windows 기본 콘솔이 cp949 라 '—', '→' 같은 문자에서 죽는다
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.agent.visualize_search import use_korean_font
    use_korean_font()
    OUT.mkdir(parents=True, exist_ok=True)

    mat, coords = load_matrix(args)
    pol = args.pollutant

    print("\n[① 시간 자기상관]")
    ac = autocorrelation(mat, args.max_lag)
    print("[② 공간 패턴 지속성]")
    pc = pattern_correlation(mat, args.max_lag)
    print("[③ 시차 교차상관 → 이동 속도] (수 분 소요)")
    adv = advection_speed(mat, coords, args.max_lag)
    print("[④ 경보 이벤트]")
    ev, agg, spec = alarm_events(mat, args)

    # --- 드론 속도 (캔버스는 측정소 분포로 근사 — geopandas 의존 회피) ---
    latm = np.deg2rad(coords["lat"].mean())
    span_km = np.hypot((coords["lat"].max() - coords["lat"].min()) * 111.0,
                       (coords["lon"].max() - coords["lon"].min()) * 111.0 * np.cos(latm))
    drone_km = args.step_frac * span_km

    lags = np.arange(len(ac))
    print("\n" + "=" * 66)
    print(f"  {pol} 시간 스케일 요약")
    print("=" * 66)
    def fmt(lag):
        return f"{lag}h" if lag is not None else f">{args.max_lag}h"

    for name, c in (("자기상관(값 유지)", ac), ("패턴상관(위치 유지)", pc)):
        marks = " ".join(f"{h}h={c[h]:.2f}" for h in (6, 12, 24, 48) if h < len(c))
        print(f"  {name:22s} r<0.5: {fmt(decorr_time(c, 0.5)):>6s}  "
              f"r<1/e: {fmt(decorr_time(c, 1/np.e)):>6s}   [{marks}]")
    pat_half = decorr_time(pc, 0.5)
    print(f"\n  드론 속도            스텝당 {drone_km:.0f} km "
          f"(캔버스 대각선 {span_km:.0f} km × {args.step_frac})")
    if adv:
        s = adv["speeds"]
        print(f"  오염 이동 속도        중앙값 {np.median(s):.0f} km/h "
              f"(사분위 {np.percentile(s,25):.0f}~{np.percentile(s,75):.0f}, "
              f"쌍 {adv['n_pairs']:,}개)")
        print(f"  → 드론이 1시간에 {drone_km:.0f} km, 오염은 {np.median(s):.0f} km/h "
              f"{'→ 추적 가능' if drone_km >= np.median(s) else '→ 추적 어려움'}")
    print(f"\n  경보 기준 초과        {len(ev)}시각 "
          f"(주의보 {(ev['단계']=='주의보').sum()}, 경보 {(ev['단계']=='경보').sum()})")
    if pat_half:
        print(f"\n  판단 근거: 공간 패턴이 {pat_half}시간이면 절반으로 흐려진다.")
        print(f"    dt=1h 이면 TTL {pat_half*2}~{pat_half*3} 스텝에서 필드가 1~2회 뒤집힌다.")
        print(f"    현재 TTL=40 → {'적절' if 20 <= pat_half*2 <= 80 else 'TTL 조정 검토'}")
    else:
        print(f"\n  ⚠ 패턴 상관이 {args.max_lag}시간 내에 0.5 아래로 안 떨어졌다.")
        print(f"    필드가 매우 느리게 변한다는 뜻 → 시간 진행의 이득이 작을 수 있다.")
        print(f"    --max-lag 를 늘려 다시 확인할 것.")
    print("=" * 66)

    # --- 그림 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(lags, ac, label="자기상관 (값 유지)")
    axes[0].plot(lags, pc, label="패턴상관 (위치 유지)")
    for lv, ls in ((0.5, "--"), (1 / np.e, ":")):
        axes[0].axhline(lv, color="gray", ls=ls, lw=0.8)
    axes[0].set_xlabel("lag (시간)"); axes[0].set_ylabel("상관")
    axes[0].set_title(f"{pol} 시간 상관"); axes[0].legend(); axes[0].grid(alpha=.3)

    if adv:
        axes[1].hist(adv["speeds"], bins=60, range=(0, 200))
        axes[1].axvline(drone_km, color="r", lw=2,
                        label=f"드론 {drone_km:.0f} km/h")
        axes[1].set_xlabel("추정 이동 속도 (km/h)"); axes[1].set_ylabel("측정소 쌍")
        axes[1].set_title("오염 이동 속도 vs 드론"); axes[1].legend()
    else:
        axes[1].text(.5, .5, "이동 신호 없음", ha="center")

    axes[2].plot(agg.index, agg.values, lw=.4)
    axes[2].axhline(spec["주의보"], color="orange", ls="--", label="주의보")
    axes[2].axhline(spec["경보"], color="red", ls="--", label="경보")
    axes[2].set_title(f"{pol} 전국 {spec['집계']} 시계열"); axes[2].legend()

    plt.tight_layout()
    fp = OUT / f"timescale_{pol}.png"
    plt.savefig(fp, dpi=120, bbox_inches="tight"); plt.close()

    ev.to_csv(OUT / f"alarm_events_{pol}.csv", index=False, encoding="utf-8-sig")
    np.savez(OUT / f"timescale_{pol}.npz", autocorr=ac, patterncorr=pc,
             speeds=(adv["speeds"] if adv else np.array([])),
             drone_km=drone_km, span_km=span_km)
    print(f"\n[저장] {fp}")
    print(f"       {OUT / f'alarm_events_{pol}.csv'}  ← 시나리오 스냅샷 후보")
    print(f"       {OUT / f'timescale_{pol}.npz'}")

    if len(ev):
        print(f"\n  고농도 이벤트 상위 10:")
        print(ev.nlargest(10, "value").to_string(index=False))


if __name__ == "__main__":
    main()
