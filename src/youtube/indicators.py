"""
유튜브 전략이 수식에서 쓰는 변수.

지표 계산은 **이 프로젝트의 검증된 프리미티브**(src/indicators)를 그대로 쓴다.
따로 구현하면 같은 이름의 RSI 가 두 가지 값을 내고, 기준 구현과 0.000% 일치를
요구하는 대조 테스트(tests/test_reference_parity.py)의 보증이 여기에는 닿지 않는다.

변수 이름은 지표와 기간에서 정해진다 (LLM 프롬프트가 이 표를 그대로 싣는다).

    SMA(period)              sma_{period}
    EMA(period)              ema_{period}
    RSI(period)              rsi_{period}
    ATR(period)              atr_{period}
    VOLUME_SMA(period)       volume_sma_{period}
    HIGHEST(period)          highest_{period}     현재 봉 포함 N봉 고가의 최댓값
    LOWEST(period)           lowest_{period}      현재 봉 포함 N봉 저가의 최솟값
    BBANDS(period, std)      bb_upper_{period}, bb_mid_{period}, bb_lower_{period}
    STOCH(k, d)              stoch_k_{k}, stoch_d_{k}
    MACD(fast, slow, signal) macd, macd_signal, macd_hist

직전 봉 값은 앞에 `prev_` 를 붙인다. shift(1) 이므로 과거만 본다.

**FX 의 volume 은 틱 수다**(MT5 tick_volume). 거래소 체결량이 아니므로 "거래량
급증" 조건은 암호화폐에서와 같은 것을 재지 않는다.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import pandas as pd

from ..indicators import core
from ..indicators.momentum import macd, rsi, stoch

OHLCV = ("open", "high", "low", "close", "volume")
PREV = "prev_"
MAX_PERIOD = 1000


class IndicatorError(ValueError):
    """지원하지 않는 지표이거나 파라미터가 범위를 벗어났다."""


def _number(params: dict, key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise IndicatorError(f"{key} 는 유한한 숫자여야 합니다: {value!r}")
    return float(value)


def _period(params: dict, key: str = "period", default: int = 20) -> int:
    n = int(_number(params, key, default))
    if not 1 <= n <= MAX_PERIOD:
        raise IndicatorError(f"{key} 는 1~{MAX_PERIOD} 범위여야 합니다: {n}")
    return n


def _sma(df, p):
    n = _period(p)
    return {f"sma_{n}": core.sma(df["close"], n)}


def _ema(df, p):
    n = _period(p)
    return {f"ema_{n}": core.ema(df["close"], n)}


def _rsi(df, p):
    n = _period(p, default=14)
    return {f"rsi_{n}": rsi(df, n)}


def _atr(df, p):
    n = _period(p, default=14)
    return {f"atr_{n}": core.atr(df, n)}


def _volume_sma(df, p):
    n = _period(p)
    return {f"volume_sma_{n}": core.sma(df["volume"], n)}


def _highest(df, p):
    n = _period(p)
    return {f"highest_{n}": core.highest(df["high"], n)}


def _lowest(df, p):
    n = _period(p)
    return {f"lowest_{n}": core.lowest(df["low"], n)}


def _bbands(df, p):
    n = _period(p)
    k = _number(p, "std", 2.0)
    if not 0 < k <= 10:
        raise IndicatorError(f"std 는 0 초과 10 이하여야 합니다: {k}")
    mid = core.sma(df["close"], n)
    band = k * core.stdev(df["close"], n)
    return {f"bb_upper_{n}": mid + band, f"bb_mid_{n}": mid, f"bb_lower_{n}": mid - band}


def _stoch(df, p):
    k, d = _period(p, "k", 14), _period(p, "d", 3)
    out = stoch(df, k, d)
    return {f"stoch_k_{k}": out["stoch_k"], f"stoch_d_{k}": out["stoch_d"]}


def _macd(df, p):
    out = macd(df, _period(p, "fast", 12), _period(p, "slow", 26),
               _period(p, "signal", 9))
    return {name: out[name] for name in ("macd", "macd_signal", "macd_hist")}


_BUILDERS: dict[str, Callable[[pd.DataFrame, dict], dict[str, pd.Series]]] = {
    "SMA": _sma, "EMA": _ema, "RSI": _rsi, "ATR": _atr,
    "VOLUME_SMA": _volume_sma, "HIGHEST": _highest, "LOWEST": _lowest,
    "BBANDS": _bbands, "STOCH": _stoch, "MACD": _macd,
}
SUPPORTED = tuple(sorted(_BUILDERS))


def columns(df: pd.DataFrame, indicators: Iterable[dict]) -> dict[str, pd.Series]:
    """OHLCV 와 선언된 지표를 {변수 이름: Series} 로 돌려준다."""
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise IndicatorError(f"봉에 필요한 열이 없습니다: {missing}")
    out: dict[str, pd.Series] = {c: df[c] for c in OHLCV}
    for indicator in indicators:
        name = str(indicator.get("name", "")).upper()
        if name not in _BUILDERS:
            raise IndicatorError(
                f"지원하지 않는 지표: {name} (지원: {', '.join(SUPPORTED)})")
        out.update(_BUILDERS[name](df, indicator.get("params") or {}))
    return out


def with_previous(names: Iterable[str], cols: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """수식이 쓰는 `prev_x` 를 x.shift(1) 로 채운다. 모르는 이름은 그대로 둔다."""
    out = dict(cols)
    for name in names:
        base = name[len(PREV):]
        if name.startswith(PREV) and name not in out and base in cols:
            out[name] = cols[base].shift(1)
    return out
