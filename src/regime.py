"""
레짐(시장 국면) 분류기.

네 가지로 나눈다.
    UPTREND    상승추세  — 방향이 있고 깔끔하다
    DOWNTREND  하락추세
    RANGE      횡보      — 방향이 없고 톱질한다
    HIGH_VOL   고변동    — 방향과 무관하게 변동성이 평소를 크게 벗어났다

HIGH_VOL 을 방향 판정보다 먼저 본다. 변동성이 터진 국면에서는 추세 전략의 손절 폭
가정이 무너지기 때문에, 방향이 뚜렷해 보여도 별도 국면으로 떼어내는 게 안전하다.

**이력 현상(hysteresis)을 넣는다.** 원시 판정을 그대로 쓰면 경계에서 매 봉 국면이
튀어서 전략 가중치가 요동친다. 새 국면이 confirm_bars 연속으로 나와야 전환한다.
전환 판정은 과거만 보므로 인과적이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .indicators import compute_all


class Regime(StrEnum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    HIGH_VOL = "high_vol"


ALL_REGIMES = tuple(Regime)

KOREAN = {
    Regime.UPTREND: "상승추세",
    Regime.DOWNTREND: "하락추세",
    Regime.RANGE: "횡보",
    Regime.HIGH_VOL: "고변동",
}

# 분류에 실제로 쓰는 특징열. 없으면 바로 터진다 — 조용히 다른 판정을 내리지 않는다.
REQUIRED = (
    "adx", "di_plus", "di_minus",
    "efficiency_ratio", "choppiness",
    "bbw_percentile", "atr_ratio", "ema_spread",
)


@dataclass(frozen=True)
class RegimeParams:
    """
    임계값과 정규화 앵커.

    앵커는 추측이 아니라 실측이다 — BTC/ETH/SOL 2년 5분봉(약 63만봉)의 분위수에서
    가져왔다. 타임프레임이나 시장을 바꾸면 scripts/calibrate_regime.py 로 다시 뽑아야 한다.
    """
    # ─ 고변동 판정 (방향보다 먼저 본다)
    vol_percentile: float = 90.0     # 볼린저 폭이 지난 200봉 중 상위 10%
    atr_ratio: float = 1.6           # 단기 ATR이 장기 ATR의 1.6배 이상

    # ─ 추세 점수 정규화 앵커 (0에 대응하는 값, 1에 대응하는 값)
    #   불린 3개를 AND로 묶지 않는다. 경계에서 셋이 동시에 성립해야 하는 구조는
    #   국면을 매 봉 깜빡이게 만들고, 추세 구간이 5봉짜리로 조각난다.
    adx_lo: float = 17.4             # p25
    adx_hi: float = 40.3             # p90
    eff_lo: float = 0.088            # p25
    eff_hi: float = 0.432            # p90
    chop_lo: float = 36.8            # p10 (낮을수록 추세)
    chop_hi: float = 55.8            # p75

    # scripts/calibrate_regime.py 스윕 결과. 20개 조합 중 세 기준
    # (추세합 30~40%, 횡보 45~55%, 추세 중위 지속 >= 15봉)을 만족한 유일한 지점.
    trend_score_min: float = 0.30

    # ─ 방향. EMA 간격이 0 근처일 때는 DI 우열만으로 판정한다.
    ema_spread_tol: float = 0.05

    # ─ 국면 전환에 필요한 연속 봉 수.
    #   지연과 깜빡임의 교환비를 정하는 값이다. 8봉이면 국면 전환을 40분 늦게
    #   인식하는 대신 전환 횟수가 절반 이하로 줄어든다. 차선책은 5봉(추세 34.9%,
    #   중위 13봉). walk-forward에서 민감도를 다시 봐야 하는 값.
    confirm_bars: int = 8


def _norm(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def trend_score(feats: pd.DataFrame, p: RegimeParams | None = None) -> pd.Series:
    """
    '추세가 있는가'를 0~1 연속값으로. 세 근거의 평균이다.
      · ADX             방향성 움직임의 강도
      · efficiency      순이동 / 총이동 — 곧게 뻗었는지
      · choppiness(역)  톱질 정도

    서로 상관은 있지만 같지 않아서, 평균이 어느 하나보다 안정적이다.
    """
    p = p or RegimeParams()
    return (
        _norm(feats["adx"], p.adx_lo, p.adx_hi)
        + _norm(feats["efficiency_ratio"], p.eff_lo, p.eff_hi)
        + _norm(-feats["choppiness"], -p.chop_hi, -p.chop_lo)
    ) / 3.0


def vol_score(feats: pd.DataFrame, p: RegimeParams | None = None) -> pd.Series:
    """고변동 국면의 강도 0~1."""
    p = p or RegimeParams()
    return (
        _norm(feats["bbw_percentile"], p.vol_percentile, 100.0)
        + _norm(feats["atr_ratio"], p.atr_ratio, 2.5)
    ).clip(0.0, 2.0) / 2.0


def classify_raw(feats: pd.DataFrame, p: RegimeParams | None = None) -> pd.Series:
    """이력 현상 없는 봉별 원시 판정."""
    p = p or RegimeParams()
    missing = [c for c in REQUIRED if c not in feats.columns]
    if missing:
        raise KeyError(f"레짐 분류에 필요한 특징열이 없습니다: {missing}")

    high_vol = (feats["bbw_percentile"] >= p.vol_percentile) | (feats["atr_ratio"] >= p.atr_ratio)
    trending = trend_score(feats, p) >= p.trend_score_min

    di_up = feats["di_plus"] > feats["di_minus"]
    # EMA 간격이 반대 방향으로 뚜렷할 때만 거부한다
    up = di_up & (feats["ema_spread"] > -p.ema_spread_tol)
    down = ~di_up & (feats["ema_spread"] < p.ema_spread_tol)

    # 기본은 횡보. 추세 조건과 고변동 조건이 순서대로 덮어쓴다.
    out = pd.Series(Regime.RANGE, index=feats.index, dtype="object")
    out[trending & up] = Regime.UPTREND
    out[trending & down] = Regime.DOWNTREND
    out[high_vol] = Regime.HIGH_VOL   # 고변동이 최종 우선

    # 입력이 아직 NaN인 워밍업 구간은 판정하지 않는다
    warm = feats[list(REQUIRED)].notna().all(axis=1)
    return out.where(warm)


def confidence(feats: pd.DataFrame, raw: pd.Series, p: RegimeParams | None = None) -> pd.Series:
    """
    판정을 얼마나 믿을 수 있는지 0~1로. 경계에 걸친 판정은 낮게 나온다.
    분류에 쓴 것과 같은 점수를 재사용해서, 화면에 뜨는 신뢰도가 실제 판정 근거와 일치한다.
    """
    p = p or RegimeParams()
    ts = trend_score(feats, p)

    # 추세: 임계값을 얼마나 넘겼는지 (임계값=0, 1.0=1)
    trend_conf = _norm(ts, p.trend_score_min, 1.0)
    # 횡보: 임계값에서 얼마나 멀어졌는지
    range_conf = _norm(-ts, -p.trend_score_min, 0.0)

    out = pd.Series(np.nan, index=feats.index)
    out = out.mask(raw.isin([Regime.UPTREND, Regime.DOWNTREND]), trend_conf)
    out = out.mask(raw == Regime.RANGE, range_conf)
    out = out.mask(raw == Regime.HIGH_VOL, vol_score(feats, p))
    return out.clip(0.0, 1.0)


def _apply_hysteresis(raw: pd.Series, confirm_bars: int) -> pd.Series:
    """
    새 국면이 confirm_bars 연속으로 나올 때만 전환한다.
    직전 상태와 최근 연속 카운트만 보므로 인과적이다.
    """
    values = raw.to_numpy(dtype=object)
    out = np.empty(len(values), dtype=object)
    out[:] = None

    current: object = None
    candidate: object = None
    streak = 0

    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            out[i] = current
            continue
        if current is None:
            current = v
            candidate, streak = v, 0
        elif v == current:
            candidate, streak = v, 0
        else:
            if v == candidate:
                streak += 1
            else:
                candidate, streak = v, 1
            if streak >= confirm_bars:
                current = v
                candidate, streak = v, 0
        out[i] = current

    return pd.Series(out, index=raw.index, dtype="object")


def _bars_in_regime(smoothed: pd.Series) -> pd.Series:
    """현재 국면이 몇 봉째 이어지는지."""
    changed = smoothed.ne(smoothed.shift(1))
    group = changed.cumsum()
    return smoothed.groupby(group).cumcount() + 1


def classify(
    feats: pd.DataFrame,
    params: RegimeParams | None = None,
) -> pd.DataFrame:
    """
    최종 레짐 판정.

    반환 열
        regime      Regime 값 (워밍업 구간은 None)
        confidence  0~1
        bars_in     현재 국면이 이어진 봉 수
        raw         이력 현상 적용 전 원시 판정 (디버깅·민감도 확인용)
    """
    p = params or RegimeParams()
    raw = classify_raw(feats, p)
    smoothed = _apply_hysteresis(raw, p.confirm_bars)
    conf = confidence(feats, smoothed, p)

    return pd.DataFrame({
        "regime": smoothed,
        "confidence": conf,
        "bars_in": _bars_in_regime(smoothed).where(smoothed.notna()),
        "raw": raw,
    })


def classify_from_ohlcv(df: pd.DataFrame, params: RegimeParams | None = None) -> pd.DataFrame:
    """OHLCV에서 바로 — 필요한 지표만 계산해서 넘긴다."""
    needed = ["adx", "efficiency_ratio", "choppiness", "bbw_percentile", "atr_ratio", "ema_spread"]
    return classify(compute_all(df, needed), params)


def regime_stats(regimes: pd.DataFrame) -> pd.DataFrame:
    """
    각 국면이 전체에서 얼마나 차지하고 평균 몇 봉 지속하는지.
    한 국면이 95%를 차지하면 분류기가 사실상 아무것도 안 하고 있다는 뜻이다.
    """
    r = regimes["regime"].dropna()
    if r.empty:
        return pd.DataFrame()

    runs = r.ne(r.shift(1)).cumsum()
    run_len = r.groupby(runs).size()
    run_label = r.groupby(runs).first()

    rows = []
    for reg in ALL_REGIMES:
        mask = run_label == reg
        bars = int((r == reg).sum())
        rows.append({
            "regime": KOREAN[reg],
            "bars": bars,
            "share_%": round(bars / len(r) * 100, 1),
            "runs": int(mask.sum()),
            "median_run": int(run_len[mask].median()) if mask.any() else 0,
            "max_run": int(run_len[mask].max()) if mask.any() else 0,
        })
    return pd.DataFrame(rows)
