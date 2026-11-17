"""
전략 명세 — LLM 이 내놓은 JSON 을 믿기 전에 거르는 곳.

    {
      "strategy_name": "rsi_ema_pullback",
      "target_market": "CRYPTO | FOREX | KR_STOCK | US_STOCK",
      "timeframe": "1m | 5m | 15m | 1h | 4h | 1d",
      "indicators_needed": [{"name": "RSI", "params": {"period": 14}}],
      "entry_conditions": {"long": "수식", "short": "수식"},
      "exit_conditions": {"take_profit_pct": 0.02, "stop_loss_pct": 0.01,
                          "indicator_exit": "설명"},
      "quant_assumptions": ["모호한 표현을 어떤 수치로 정했는지"]
    }

**형식 검사로 끝내지 않고 실제로 한 번 돌려 본다.** 선언하지 않은 변수, 값이
끝내 나오지 않는 긴 기간, 0 으로 나누는 수식은 형식만 봐서는 통과한다.
그런 명세가 저장되면 엔진 안에서 터지거나, 터지지도 않고 영원히 침묵한다.

exit_conditions 는 **기록만 한다.** 이 앱의 손절·목표는 엔진이 ATR 로 정하고
(risk.atr_stop_mult, risk.reward_ratio) 진입 직후 브로커에 건다. 영상의 청산
규칙을 따로 실행하면 그 보호 주문과 충돌한다.
"""

from __future__ import annotations

import json
import keyword
import math
import re

import numpy as np
import pandas as pd

from . import indicators
from .expr import evaluate, names_in

NAME = re.compile(r"^[a-z][a-z0-9_]{2,59}$")
MARKETS = ("CRYPTO", "FOREX", "KR_STOCK", "US_STOCK")
TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
SIDES = ("long", "short")
# 검증용 합성 봉 개수. 엔진은 학습 구간(기본 6개월)+워밍업을 받아 오므로
# 1시간봉이면 4,000봉쯤 된다. MAX_PERIOD(1000) 를 넉넉히 덮으면 충분하다.
CHECK_BARS = 1500


class SpecError(ValueError):
    """명세가 규칙에 맞지 않는다. 메시지는 LLM 재시도에 그대로 되먹인다."""


def check_name(name: object) -> str:
    """저장 파일 이름이 되므로 경로 문자가 들어갈 수 없게 좁게 받는다."""
    # fullmatch 여야 한다. match + `$` 는 끝의 줄바꿈 앞에서도 맞는다 ("con\n" 통과).
    if not isinstance(name, str) or not NAME.fullmatch(name) or keyword.iskeyword(name):
        raise SpecError(
            f"strategy_name 은 영문 소문자 스네이크케이스(3~60자)여야 합니다: {name!r}")
    return name


def _text(value: object, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SpecError(f"{field} 는 문자열이어야 합니다: {value!r}")
    return value.strip()


def _ratio(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise SpecError(f"{field} 는 유한한 숫자이거나 null 이어야 합니다: {value!r}")
    return float(value)


def _choice(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise SpecError(f"{field} 는 {' | '.join(allowed)} 중 하나여야 합니다: {value!r}")
    return str(value)


def _indicator(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise SpecError(f"indicators_needed 항목은 객체여야 합니다: {raw!r}")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise SpecError(f"params 는 객체여야 합니다: {params!r}")
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise SpecError(f"params.{key} 는 유한한 숫자여야 합니다: {value!r}")
    return {"name": _text(raw.get("name"), "indicators_needed.name").upper(),
            "params": {str(k): v for k, v in params.items()}}


def normalize(raw: object) -> dict:
    """아는 필드만 남기고 형을 맞춘다. 모르는 필드는 버린다."""
    if not isinstance(raw, dict):
        raise SpecError("전략 명세는 JSON 객체여야 합니다")
    needed = raw.get("indicators_needed") or []
    entry = raw.get("entry_conditions") or {}
    exits = raw.get("exit_conditions") or {}
    notes = raw.get("quant_assumptions") or []
    if not isinstance(needed, list) or not isinstance(notes, list) \
            or not isinstance(entry, dict) or not isinstance(exits, dict):
        raise SpecError("indicators_needed·quant_assumptions 는 배열, "
                        "entry_conditions·exit_conditions 는 객체여야 합니다")
    return {
        "strategy_name": check_name(raw.get("strategy_name")),
        "target_market": _choice(raw.get("target_market"), "target_market", MARKETS),
        "timeframe": _choice(raw.get("timeframe"), "timeframe", TIMEFRAMES),
        "indicators_needed": [_indicator(i) for i in needed],
        "entry_conditions": {s: _text(entry.get(s), f"entry_conditions.{s}")
                             for s in SIDES},
        "exit_conditions": {
            "take_profit_pct": _ratio(exits.get("take_profit_pct"), "take_profit_pct"),
            "stop_loss_pct": _ratio(exits.get("stop_loss_pct"), "stop_loss_pct"),
            "indicator_exit": _text(exits.get("indicator_exit"), "indicator_exit"),
        },
        "quant_assumptions": [_text(n, "quant_assumptions") for n in notes],
    }


def synthetic_bars(n: int = CHECK_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    spread = rng.uniform(0.1, 1.0, n)
    index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n), "high": close + spread,
        "low": close - spread, "close": close,
        "volume": rng.uniform(100, 1000, n),
    }, index=index)


def validate(spec: dict) -> None:
    """정규화된 명세를 합성 봉으로 실제로 계산해 본다. 문제가 있으면 SpecError."""
    bars = synthetic_bars()
    try:
        cols = indicators.columns(bars, spec["indicators_needed"])
    except indicators.IndicatorError as exc:
        raise SpecError(str(exc)) from exc

    silent = sorted(name for name, s in cols.items() if pd.isna(s.iloc[-1]))
    if silent:
        raise SpecError(f"지표 기간이 너무 깁니다 ({CHECK_BARS}봉으로도 값이 없음): {silent}")

    conditions = {s: spec["entry_conditions"][s] for s in SIDES}
    if not any(conditions.values()):
        raise SpecError("entry_conditions 의 long 과 short 가 모두 비어 있습니다")

    allowed = set(cols) | {indicators.PREV + name for name in cols}
    for side, expr in conditions.items():
        if not expr:
            continue
        unknown = names_in(expr) - allowed
        if unknown:
            raise SpecError(f"{side} 수식이 선언되지 않은 변수를 씁니다: {sorted(unknown)}. "
                            f"쓸 수 있는 변수: {sorted(allowed)}")
        evaluate(expr, indicators.with_previous(names_in(expr), cols), bars.index)


def parse(text: str) -> dict:
    """LLM 응답(JSON 문자열)을 검증된 명세로 바꾼다."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(f"JSON 이 아닙니다: {exc.msg}") from exc
    spec = normalize(raw)
    validate(spec)
    return spec
