"""복합 보상 — 여러 지표를 항으로 만들어 가중합.

    R = Σ_k  λ_k · R_k

설계 규약 (λ 를 손으로 잡을 수 있게 하려고 지키는 것들):

  1. **모든 항은 무차원이고 에피소드 누적이 대략 [0, 1] 범위**다.
     각 항은 "reset 시점 총량 대비 이번 스텝에 줄인 비율" 로 정의한다.
     그래서 λ 는 '단위 환산 계수'가 아니라 순수한 '중요도'가 되고,
     위험도 데이터를 진료인원 → 10만명당 으로 바꿔도 다시 안 잡아도 된다.

  2. **가중치장 w 는 육지 평균 1로 정규화**되어 있다 (risk_field.py).
     따라서 w ≡ 1 이면 risk 항은 info 항과 **수치적으로 완전히 동일**하다.
     가중치를 켜기 전/후를 같은 축에서 비교할 수 있다.

  3. **정규화 기준은 reset 에서 한 번만 고정**한다. 스텝마다 다시 잡으면
     '분모가 줄어서 생긴 보상'이 섞여 들어가 정책이 그걸 학습해 버린다.

항을 추가하려면 RewardTerm 을 상속해 TERMS 에 등록만 하면 된다.
env / 모델 / MCTS 코드는 건드릴 필요 없다.

주의 — 항마다 '보상의 성질'이 다르다:
  info / risk / conc : 분산 감소만 보므로 **측정값과 무관**, 위치만으로 결정된다.
                       (실제 운용에서도 계산 가능하고, g망의 보상 예측이 결정적)
  hazard             : Kriging mean 이 들어가므로 **측정값에 의존**한다.
                       같은 위치라도 측정 결과에 따라 보상이 달라진다 →
                       g망 입장에서 보상이 확률적이 되고 학습이 느려질 수 있다.
                       이건 버그가 아니라 대가다. λ_hazard 를 크게 줄 거면
                       reward_loss 가 안 떨어지는 걸 감안해야 한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr          # 표준정규 CDF Φ (pykrige 의존성이라 항상 있음)

from src.env.dummy_env import RewardFn

_EPS = 1e-12

# 환경부 예보 기준 '나쁨' 시작 농도. hazard 항의 기본 임계값.
HAZARD_THRESHOLD = {
    "PM10": 81.0, "PM25": 36.0,
    "O3": 0.091, "NO2": 0.061, "SO2": 0.051, "CO": 9.01,
}


@dataclass
class StepContext:
    """한 스텝의 before/after 스냅샷. 항들은 이것만 보고 계산한다."""
    mean_before: np.ndarray     # [res, res]  Kriging 평균 (원단위, 예: ㎍/㎥)
    var_before: np.ndarray      # [res, res]  Kriging 분산
    mean_after: np.ndarray
    var_after: np.ndarray
    land: np.ndarray            # [res, res]  bool
    lon: float                  # 이번에 측정한 지점
    lat: float
    action: object


# ---------------------------------------------------------------- 항 인터페이스
class RewardTerm(ABC):
    """보상 한 항. reset 에서 정규화 기준을 잡고, 스텝마다 스칼라를 낸다."""
    name = "term"

    def reset(self, env, ctx0: StepContext) -> None:
        """에피소드 시작. ctx0 는 before/after 가 모두 초기 상태인 스냅샷."""

    @abstractmethod
    def __call__(self, env, ctx: StepContext) -> float: ...


class _WeightedVarDrop(RewardTerm):
    """공통 골격: reward = Σ_land W·(var_before − var_after) / Σ_land W·var_reset

    W 를 무엇으로 잡느냐만 서브클래스가 정한다. W 는 에피소드 내내 고정 —
    스텝마다 바뀌면 '가중치가 변해서 생긴 차이'가 보상에 섞인다.
    """

    def _weights(self, env, ctx0: StepContext) -> np.ndarray:
        raise NotImplementedError

    def reset(self, env, ctx0: StepContext) -> None:
        self._W = np.nan_to_num(self._weights(env, ctx0), nan=0.0)
        self._denom = max(float((self._W * ctx0.var_before)[ctx0.land].sum()), _EPS)

    def __call__(self, env, ctx: StepContext) -> float:
        drop = (self._W * (ctx.var_before - ctx.var_after))[ctx.land].sum()
        return float(drop) / self._denom


class InfoGainTerm(_WeightedVarDrop):
    """순수 정보이득 — 육지 칸 Kriging 분산 총합 감소 비율 (기존 보상과 동일)."""
    name = "info"

    def _weights(self, env, ctx0):
        return np.ones_like(ctx0.var_before)


class RiskWeightedInfoGainTerm(_WeightedVarDrop):
    """위험 가중 정보이득 — 호흡기 위험도가 높은 곳의 불확실성을 우선 해소.

    '어디를 먼저 알아야 하는가'를 사람 피해 기준으로 바꾸는 항.
    측정값과 무관하므로 실제 운용에서도 그대로 계산된다.
    """
    name = "risk"

    def __init__(self, risk_field):
        self.risk = risk_field

    def _weights(self, env, ctx0):
        return self.risk.w


class ConcWeightedInfoGainTerm(_WeightedVarDrop):
    """농도 가중 정보이득 — 지금 추정 농도가 높은 곳의 불확실성을 우선 해소.

    가중치는 reset 시점의 Kriging mean 을 그대로 쓴다(정규화 후). 스텝마다
    갱신하지 않는 이유는 위 _WeightedVarDrop 주석 참고.
    risk_field 를 같이 주면 (농도 × 위험도) 로 곱해진다 — '위험한 사람이
    많은 곳의, 농도가 높은 지점'.
    """
    name = "conc"

    def __init__(self, risk_field=None, floor: float = 0.0):
        self.risk = risk_field
        self.floor = floor          # 이 농도 미만은 관심 없음(0으로 깎기)

    def _weights(self, env, ctx0):
        c = np.clip(ctx0.mean_before - self.floor, 0.0, None)
        m = float(c[ctx0.land].mean())
        c = c / m if m > 0 else np.ones_like(c)     # 육지 평균 1로
        return c * self.risk.w if self.risk is not None else c


class HazardUncertaintyTerm(RewardTerm):
    """기준치 초과 판정의 불확실성 감소.

    호흡기 질환자에게 실제로 필요한 답은 농도 실수값이 아니라
    "여기 지금 나가도 되나(= 나쁨 기준을 넘었나)" 라는 이진 판정이다.
    Kriging 이 (mean, var) 를 주므로 초과확률과 그 엔트로피를 바로 쓸 수 있다:

        p = Φ((mean − thr) / √var)          초과확률
        H = −p·ln p − (1−p)·ln(1−p)         판정의 애매함 (p=0.5 에서 최대)
        reward = Σ_land w·(H_before − H_after) / Σ_land w·H_reset

    이 항은 '분산이 크지만 명백히 좋음/명백히 나쁨인 곳'은 무시하고
    **경계선에 걸친 애매한 지역**으로 무인기를 보낸다. info 항과 방향이
    자주 어긋나므로, 둘의 가중합이 곧 '정밀도 vs 판정' 트레이드오프다.

    ※ mean 에 의존 → 측정값에 의존 → 보상이 확률적. 모듈 상단 주석 참고.
    """
    name = "hazard"

    def __init__(self, threshold: float, risk_field=None):
        self.thr = float(threshold)
        self.risk = risk_field

    def _entropy(self, mean, var):
        sd = np.sqrt(np.clip(var, _EPS, None))
        p = np.clip(ndtr((mean - self.thr) / sd), _EPS, 1.0 - _EPS)
        return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))

    def reset(self, env, ctx0: StepContext) -> None:
        self._W = self.risk.w if self.risk is not None else np.ones_like(ctx0.var_before)
        h0 = self._entropy(ctx0.mean_before, ctx0.var_before)
        self._denom = max(float((self._W * h0)[ctx0.land].sum()), _EPS)

    def __call__(self, env, ctx: StepContext) -> float:
        h_b = self._entropy(ctx.mean_before, ctx.var_before)
        h_a = self._entropy(ctx.mean_after, ctx.var_after)
        return float((self._W * (h_b - h_a))[ctx.land].sum()) / self._denom


# ---------------------------------------------------------------- 가중합
class CompositeReward(RewardFn):
    """항들의 가중합. env.reward_fn 슬롯에 그대로 꽂힌다.

    last       : 직전 스텝의 항별 기여 {name: λ·R_k}
    ep_totals  : 이번 에피소드 항별 누적 (로깅/λ 튜닝용)
    """

    def __init__(self, terms: list[tuple[RewardTerm, float]]):
        self.terms = [(t, float(w)) for t, w in terms]
        if not self.terms:
            raise ValueError("보상 항이 하나도 없음")
        self.last: dict[str, float] = {}
        self.ep_totals: dict[str, float] = {}

    # env.reset() 이 불러줌
    def reset(self, env) -> None:
        ctx0 = _context(env, env.var, env.mean)      # before == after == 초기 상태
        # 가중치 0인 항은 reset 도 건너뛴다. __call__ 과 반드시 같은 조건이어야 한다 —
        # 예전엔 reset 만 전부 돌아서, risk_field 없이 --reward info=1 을 주면
        # (risk 항이 0인데도) RiskWeightedInfoGainTerm.reset 이 None.w 로 죽었다.
        for t, lam in self.terms:
            if lam == 0.0:
                continue
            t.reset(env, ctx0)
        self.last = {}
        self.ep_totals = {t.name: 0.0 for t, lam in self.terms if lam != 0.0}

    # env.step() 이 불러줌 — RewardFn 시그니처
    def __call__(self, env, action) -> float:
        ctx = _context(env, env.var_prev, env.mean_prev, action=action)
        total = 0.0
        self.last = {}
        for t, lam in self.terms:
            if lam == 0.0:
                continue                              # 꺼진 항은 계산도 건너뜀
            r = lam * t(env, ctx)
            self.last[t.name] = r
            self.ep_totals[t.name] = self.ep_totals.get(t.name, 0.0) + r
            total += r
        return total

    def describe(self) -> str:
        on = [f"{t.name}×{w:g}" for t, w in self.terms if w != 0.0]
        off = [t.name for t, w in self.terms if w == 0.0]
        s = " + ".join(on) if on else "(전부 0)"
        return s + (f"   [꺼짐: {', '.join(off)}]" if off else "")

    def breakdown_str(self) -> str:
        if not self.ep_totals:
            return ""
        return " ".join(f"{k}={v:+.3f}" for k, v in self.ep_totals.items()
                        if v != 0.0)


def _context(env, var_before, mean_before, action=None) -> StepContext:
    d = env.drones[0]
    return StepContext(mean_before=np.asarray(mean_before),
                       var_before=np.asarray(var_before),
                       mean_after=np.asarray(env.mean),
                       var_after=np.asarray(env.var),
                       land=env.land, lon=d["lon"], lat=d["lat"], action=action)


# ---------------------------------------------------------------- 조립
def parse_weights(spec: str) -> dict[str, float]:
    """'info=1,risk=2,hazard=0.5' → {'info':1.0,'risk':2.0,'hazard':0.5}"""
    out: dict[str, float] = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--reward 형식 오류: {part!r} (name=weight 이어야 함)")
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise ValueError(f"--reward 가중치가 숫자가 아님: {part!r}") from None
    if not out:
        raise ValueError("--reward 에 항이 하나도 없음")
    return out


TERM_NAMES = ("info", "risk", "conc", "hazard")


def build_reward(spec: str, risk_field=None, pollutant: str = "PM10",
                 hazard_threshold: float | None = None,
                 conc_floor: float = 0.0) -> CompositeReward:
    """--reward 문자열 → CompositeReward.

    risk_field 가 None 인데 risk/hazard 가중치가 0이 아니면 에러를 낸다
    (조용히 균등 가중으로 떨어지면 '위험도를 켰다고 착각한 실험'이 나온다).
    """
    weights = parse_weights(spec)
    unknown = set(weights) - set(TERM_NAMES)
    if unknown:
        raise ValueError(f"모르는 보상 항: {sorted(unknown)}. 가능: {list(TERM_NAMES)}")

    needs_risk = [k for k in ("risk",) if weights.get(k, 0.0) != 0.0]
    if needs_risk and risk_field is None:
        raise ValueError(
            f"{needs_risk} 항을 켰는데 위험도장이 없습니다. "
            "--risk-csv / --risk-col 로 지역 통계를 주거나, 해당 가중치를 0으로 두세요.")

    thr = hazard_threshold
    if thr is None:
        thr = HAZARD_THRESHOLD.get(pollutant)
    if weights.get("hazard", 0.0) != 0.0 and thr is None:
        raise ValueError(
            f"'{pollutant}' 의 기본 임계값이 없습니다. --hazard-threshold 로 지정하세요.")

    built = {
        "info": lambda: InfoGainTerm(),
        "risk": lambda: RiskWeightedInfoGainTerm(risk_field),
        "conc": lambda: ConcWeightedInfoGainTerm(risk_field, floor=conc_floor),
        "hazard": lambda: HazardUncertaintyTerm(thr or 0.0, risk_field),
    }
    return CompositeReward([(built[n](), weights.get(n, 0.0)) for n in TERM_NAMES])
