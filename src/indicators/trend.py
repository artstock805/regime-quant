"""추세 계열 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (
    atr, ema, highest, hl2, linreg_r2, linreg_slope, lowest, rma, sma, true_range,
)
from .registry import register


@register("ema_spread", "trend", {"fast": 20, "slow": 50}, needs=("close",))
def ema_spread(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """빠른 EMA와 느린 EMA의 간격을 종가 대비 %로. 종목 가격대와 무관하게 비교 가능."""
    return (ema(df["close"], fast) - ema(df["close"], slow)) / df["close"] * 100.0


@register("ema_slope", "trend", {"period": 50, "lookback": 10}, needs=("close",))
def ema_slope(df: pd.DataFrame, period: int = 50, lookback: int = 10) -> pd.Series:
    line = ema(df["close"], period)
    return (line / line.shift(lookback) - 1.0) * 100.0


@register("sma_slope", "trend", {"period": 200, "lookback": 20}, needs=("close",))
def sma_slope(df: pd.DataFrame, period: int = 200, lookback: int = 20) -> pd.Series:
    line = sma(df["close"], period)
    return (line / line.shift(lookback) - 1.0) * 100.0


@register("price_vs_ema200", "trend", {"period": 200}, needs=("close",))
def price_vs_ema200(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return (df["close"] / ema(df["close"], period) - 1.0) * 100.0


@register(
    "adx", "trend", {"period": 14},
    outputs=("adx", "di_plus", "di_minus"),
    needs=("high", "low", "close"),
)
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder ADX. 레짐 분류에서 '추세가 있는가'의 1차 판단 근거로 쓴다."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_n = rma(true_range(df), period)
    di_plus = 100.0 * rma(pd.Series(plus_dm, index=df.index), period) / tr_n
    di_minus = 100.0 * rma(pd.Series(minus_dm, index=df.index), period) / tr_n

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return pd.DataFrame(
        {"adx": rma(dx, period), "di_plus": di_plus, "di_minus": di_minus}
    )


@register("adx_slope", "trend", {"period": 14, "lookback": 5}, needs=("high", "low", "close"))
def adx_slope(df: pd.DataFrame, period: int = 14, lookback: int = 5) -> pd.Series:
    """ADX가 오르는 중인지 — 추세의 '시작'과 '소진'을 가른다."""
    return adx(df, period)["adx"].diff(lookback)


@register(
    "aroon", "trend", {"period": 25},
    outputs=("aroon_up", "aroon_down", "aroon_osc"),
    needs=("high", "low"),
)
def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    win = period + 1
    up = _argpos(df["high"], win, np.argmax) / period * 100.0
    down = _argpos(df["low"], win, np.argmin) / period * 100.0
    return pd.DataFrame({"aroon_up": up, "aroon_down": down, "aroon_osc": up - down})


def _argpos(s: pd.Series, window: int, fn) -> pd.Series:
    """윈도우 안에서 최고/최저가 나온 위치(0~window-1). 스트라이드 뷰로 벡터화."""
    from .core import _place, _windows

    view = _windows(s, window)
    if view is None:
        return pd.Series(np.nan, index=s.index)
    pos = fn(view, axis=1).astype("float64")
    return _place(np.where(np.isnan(view).any(axis=1), np.nan, pos), s, window)


@register(
    "donchian", "trend", {"period": 20},
    outputs=("dc_upper", "dc_lower", "dc_mid", "dc_pos"),
    needs=("high", "low", "close"),
)
def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    돈치안 채널. 돌파 전략의 기준선.

    주의: 채널 상단은 **직전 봉까지의** 최고가여야 한다. 현재 봉의 고가를 포함하면
    현재 봉이 항상 채널 안에 있게 되어 돌파가 절대 잡히지 않는다.
    """
    upper = highest(df["high"], period).shift(1)
    lower = lowest(df["low"], period).shift(1)
    mid = (upper + lower) / 2.0
    span = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "dc_upper": upper, "dc_lower": lower, "dc_mid": mid,
        "dc_pos": (df["close"] - lower) / span,
    })


@register(
    "keltner", "trend", {"period": 20, "mult": 2.0},
    outputs=("kc_upper", "kc_lower", "kc_width", "kc_pos"),
    needs=("high", "low", "close"),
)
def keltner(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    basis = ema(df["close"], period)
    band = atr(df, period) * mult
    upper, lower = basis + band, basis - band
    span = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "kc_upper": upper, "kc_lower": lower,
        "kc_width": span / basis.replace(0, np.nan) * 100.0,
        "kc_pos": (df["close"] - lower) / span,
    })


@register(
    "supertrend", "trend", {"period": 10, "mult": 3.0},
    outputs=("st_line", "st_dir"),
    needs=("high", "low", "close"),
)
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Supertrend. 상태를 이어가야 하므로 루프로 계산한다(직전 값만 참조 — 인과적)."""
    band = atr(df, period) * mult
    mid = hl2(df)
    upper_basic = (mid + band).to_numpy()
    lower_basic = (mid - band).to_numpy()
    close = df["close"].to_numpy()

    n = len(df)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    line = np.full(n, np.nan)

    start = int(np.argmax(~np.isnan(upper_basic))) if np.any(~np.isnan(upper_basic)) else n
    for i in range(start, n):
        if np.isnan(upper_basic[i]):
            continue
        if i == start or np.isnan(upper[i - 1]):
            upper[i], lower[i], direction[i] = upper_basic[i], lower_basic[i], 1.0
        else:
            upper[i] = min(upper_basic[i], upper[i - 1]) if close[i - 1] <= upper[i - 1] else upper_basic[i]
            lower[i] = max(lower_basic[i], lower[i - 1]) if close[i - 1] >= lower[i - 1] else lower_basic[i]
            if close[i] > upper[i - 1]:
                direction[i] = 1.0
            elif close[i] < lower[i - 1]:
                direction[i] = -1.0
            else:
                direction[i] = direction[i - 1]
        line[i] = lower[i] if direction[i] > 0 else upper[i]

    return pd.DataFrame({"st_line": line, "st_dir": direction}, index=df.index)


@register(
    "psar", "trend", {"step": 0.02, "max_step": 0.2},
    outputs=("psar", "psar_dir"),
    needs=("high", "low", "close"),
)
def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"psar": out, "psar_dir": direction}, index=df.index)

    bull = True
    af = step
    sar = low[0]
    ep = high[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i - 1], low[max(0, i - 2)])
            if low[i] < sar:
                bull, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[max(0, i - 2)])
            if high[i] > sar:
                bull, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        out[i] = sar
        direction[i] = 1.0 if bull else -1.0

    return pd.DataFrame({"psar": out, "psar_dir": direction}, index=df.index)


@register(
    "vortex", "trend", {"period": 14},
    outputs=("vi_plus", "vi_minus"),
    needs=("high", "low", "close"),
)
def vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    vm_plus = (df["high"] - df["low"].shift(1)).abs()
    vm_minus = (df["low"] - df["high"].shift(1)).abs()
    tr_sum = true_range(df).rolling(period, min_periods=period).sum().replace(0, np.nan)
    return pd.DataFrame({
        "vi_plus": vm_plus.rolling(period, min_periods=period).sum() / tr_sum,
        "vi_minus": vm_minus.rolling(period, min_periods=period).sum() / tr_sum,
    })


@register(
    "ichimoku", "trend", {"conv": 9, "base": 26, "span": 52},
    outputs=("tenkan", "kijun", "cloud_pos"),
    needs=("high", "low", "close"),
)
def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span: int = 52) -> pd.DataFrame:
    """
    일목 기준선·전환선과 구름 대비 위치.

    선행스팬은 원래 base 봉 **앞으로** 그려진다. 시점 t에서 우리 눈에 보이는 구름은
    base 봉 전에 계산된 값이므로 shift(+base) — 과거 방향 이동이라 인과적이다.
    """
    tenkan = (highest(df["high"], conv) + lowest(df["low"], conv)) / 2.0
    kijun = (highest(df["high"], base) + lowest(df["low"], base)) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(base)
    senkou_b = ((highest(df["high"], span) + lowest(df["low"], span)) / 2.0).shift(base)

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    thickness = (cloud_top - cloud_bot).replace(0, np.nan)
    pos = pd.Series(np.nan, index=df.index)
    pos = pos.mask(df["close"] > cloud_top, 1.0)
    pos = pos.mask(df["close"] < cloud_bot, -1.0)
    pos = pos.mask(
        (df["close"] <= cloud_top) & (df["close"] >= cloud_bot),
        (df["close"] - cloud_bot) / thickness * 2.0 - 1.0,
    )
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun, "cloud_pos": pos})


@register("ma_ribbon", "trend", {"periods": (10, 20, 50, 100, 200)}, needs=("close",))
def ma_ribbon(df: pd.DataFrame, periods=(10, 20, 50, 100, 200)) -> pd.Series:
    """
    여러 EMA가 순서대로 정렬돼 있는 비율(-1 완전 하락정렬 ~ +1 완전 상승정렬).
    지표 하나가 아니라 '정렬 합의도'를 본다 — 추세의 견고함에 대한 대리 지표.
    """
    lines = [ema(df["close"], p) for p in periods]
    pairs = [(lines[i] < lines[i + 1]) for i in range(len(lines) - 1)]
    down = pd.concat(pairs, axis=1).sum(axis=1)
    total = len(pairs)
    aligned_up = total - down          # 짧은 게 위에 있는 쌍의 수
    out = (aligned_up - down) / total
    return out.where(lines[-1].notna())


@register("linreg_slope", "trend", {"period": 50}, needs=("close",))
def linreg_slope_ind(df: pd.DataFrame, period: int = 50) -> pd.Series:
    return linreg_slope(df["close"], period)


@register("linreg_r2", "trend", {"period": 50}, needs=("close",))
def linreg_r2_ind(df: pd.DataFrame, period: int = 50) -> pd.Series:
    return linreg_r2(df["close"], period)


@register("efficiency_ratio", "trend", {"period": 20}, needs=("close",))
def efficiency_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Kaufman 효율성 비율 = 순이동거리 / 총이동거리 (0~1).
    1에 가까우면 곧게 뻗은 추세, 0에 가까우면 톱질. 레짐 분류의 핵심 입력.
    """
    net = (df["close"] - df["close"].shift(period)).abs()
    gross = df["close"].diff().abs().rolling(period, min_periods=period).sum()
    return net / gross.replace(0, np.nan)


@register("choppiness", "trend", {"period": 14}, needs=("high", "low", "close"))
def choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index (0~100). 높으면 횡보, 낮으면 추세.
    efficiency_ratio 와 방향이 반대라 서로 교차 검증에 쓴다.
    """
    tr_sum = true_range(df).rolling(period, min_periods=period).sum()
    rng = highest(df["high"], period) - lowest(df["low"], period)
    ratio = tr_sum / rng.replace(0, np.nan)
    return 100.0 * np.log10(ratio.where(ratio > 0)) / np.log10(period)
