"""
트레이딩뷰 MCP 서버.

    python -m src.tradingview.mcp_server

MCP 는 **클로드와 도구**를 잇는 프로토콜이다. 봇과 거래소를 잇는 것이 아니다.
따라서 이 서버는 자동매매를 하지 않는다 — 주문 도구를 일부러 넣지 않았다.
대화 중에 시세·지표·레짐을 조회하고, 우리 구현을 트레이딩뷰와 대조하는 용도다.

제공하는 도구
    tv_indicators     트레이딩뷰가 계산한 지표 스냅샷
    tv_vs_ours        같은 시점에서 우리 구현과 트레이딩뷰를 나란히 비교
    our_regime        우리 레짐 분류기의 현재 판정과 근거
    our_signals       전략별 신호와 앙상블 합산 점수
    list_fields       조회 가능한 지표·타임프레임 목록

주의
  · 트레이딩뷰는 **현재 스냅샷**만 준다. 과거 시계열은 주지 않는다.
  · 개인적 용도로만 쓴다. 재배포·상업적 재판매는 하지 않는다.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
from mcp.server.mcpserver import MCPServer                  # noqa: E402

from src.config import load_config                          # noqa: E402
from src.data.binance import fetch_klines                   # noqa: E402
from src.indicators import compute_all                      # noqa: E402
from src.regime import KOREAN, Regime, classify             # noqa: E402
from src.strategies import build_all                        # noqa: E402
from src.tradingview import scanner                         # noqa: E402

server = MCPServer(
    name="regime-quant-tradingview",
    instructions=(
        "트레이딩뷰 시세·지표 조회와, 이 프로젝트의 레짐·전략 신호 조회를 제공합니다. "
        "주문 실행 도구는 없습니다 — MCP 는 대화용 조회 채널이지 매매 경로가 아닙니다. "
        "트레이딩뷰 값은 현재 시점 스냅샷이며 과거 시계열은 제공하지 않습니다."
    ),
)

# 지표 계산에 필요한 워밍업. Wilder 평활이 기준 구현과 수렴하려면 400봉이 필요하다
# (tests/test_reference_parity.py 에서 실측). 넉넉히 800봉을 받는다.
WARMUP_BARS = 800
_BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _load_bars(symbol: str, interval: str, live: bool = True) -> pd.DataFrame:
    """지표를 낼 만큼의 최근 봉. live=True 면 진행 중인 봉도 포함한다."""
    minutes = _BAR_MINUTES.get(interval)
    if minutes is None:
        raise ValueError(f"지원하지 않는 타임프레임: {interval}")
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes * WARMUP_BARS)
    return fetch_klines(symbol, interval, start, now,
                        market="futures_um", closed_only=not live)


def _fmt(v: Any, digits: int = 4) -> Any:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), digits) if isinstance(v, (int, float)) else v


@server.tool(
    name="tv_indicators",
    description=(
        "트레이딩뷰가 계산한 지표 스냅샷을 가져옵니다. 로그인 불필요. "
        "symbols 는 ['BTCUSDT','ETHUSDT'] 처럼, interval 은 1m/5m/15m/30m/1h/4h/1d. "
        "fields 를 비우면 조회 가능한 전체를 돌려줍니다."
    ),
)
def tv_indicators(
    symbols: list[str],
    interval: str = "1h",
    fields: list[str] | None = None,
) -> dict:
    quotes = scanner.fetch(symbols, fields, interval=interval)
    return {
        "source": "tradingview-scanner",
        "interval": interval,
        "note": "현재 시점 스냅샷입니다. 과거 시계열이 아닙니다.",
        "quotes": [q.as_dict() for q in quotes],
    }


@server.tool(
    name="tv_vs_ours",
    description=(
        "같은 시점에서 우리 지표 구현과 트레이딩뷰 값을 나란히 비교합니다. "
        "구현이 표준과 다른지 확인하는 용도입니다. "
        "상대오차가 큰 지표는 정의나 파라미터가 다를 수 있습니다."
    ),
)
def tv_vs_ours(symbol: str, interval: str = "1h") -> dict:
    pairs = ["rsi", "adx", "di_plus", "di_minus", "atr", "macd", "macd_signal",
             "bb_upper", "bb_lower", "stoch_k", "stoch_d", "cci",
             "williams_r", "mfi", "momentum", "awesome_osc"]
    q = scanner.fetch_one(symbol, fields=["close", *pairs], interval=interval)
    df = _load_bars(symbol, interval, live=True)
    feats = compute_all(df, pairs)
    last = feats.iloc[-1]

    rows = []
    for name in pairs:
        mine, theirs = last.get(name), q.get(name)
        if mine is None or theirs is None or not np.isfinite(mine):
            rows.append({"indicator": name, "ours": _fmt(mine),
                         "tradingview": _fmt(theirs), "gap_in_sd": None})
            continue
        # 값 자체로 나누면 안 된다. CCI·모멘텀·AO 처럼 0을 넘나드는 지표는
        # 분모가 0에 가까워져 오차가 수만 %로 튄다 — 지표의 잘못이 아니라 척도의 잘못이다.
        # 그 지표가 실제로 얼마나 흔들리는지(최근 200봉 표준편차)로 나눈다.
        scale = float(feats[name].tail(200).std())
        gap = abs(float(mine) - theirs)
        rows.append({
            "indicator": name,
            "ours": _fmt(mine),
            "tradingview": _fmt(theirs),
            "gap": _fmt(gap),
            # 이 지표의 1 표준편차 대비 몇 배 차이인가. 0.1 이하면 사실상 같다.
            "gap_in_sd": None if not np.isfinite(scale) or scale <= 0
                         else round(gap / scale, 3),
        })

    return {
        "symbol": symbol, "interval": interval,
        "our_last_bar": df.index[-1].isoformat(),
        "our_bars_used": len(df),
        "close_ours": _fmt(df["close"].iloc[-1], 2),
        "close_tradingview": _fmt(q.get("close"), 2),
        "comparison": rows,
        "how_to_read": (
            "gap_in_sd 는 그 지표의 최근 200봉 표준편차 대비 차이입니다. "
            "0.1 이하면 사실상 같고, 1.0 을 넘으면 정의나 파라미터가 다를 수 있습니다."
        ),
        "caveat": (
            "트레이딩뷰는 자체 집계 피드를 쓰고 계산 윈도우도 공개돼 있지 않습니다. "
            "차이가 곧 우리 오류를 뜻하지 않습니다. 구현 검증은 "
            "tests/test_reference_parity.py (ta 라이브러리 대조)가 담당합니다."
        ),
    }


@server.tool(
    name="our_regime",
    description=(
        "이 프로젝트의 레짐 분류기가 지금 어떤 국면으로 보는지와 그 근거를 돌려줍니다. "
        "상승추세 / 하락추세 / 횡보 / 고변동 네 가지."
    ),
)
def our_regime(symbol: str, interval: str = "1h") -> dict:
    df = _load_bars(symbol, interval, live=False)
    feats = compute_all(df)
    reg = classify(feats)
    last, f = reg.iloc[-1], feats.iloc[-1]
    rid = last["regime"] if pd.notna(last["regime"]) else None

    return {
        "symbol": symbol, "interval": interval,
        "bar_time": df.index[-1].isoformat(),
        "regime": rid,
        "regime_korean": KOREAN.get(Regime(rid), "판정중") if rid else "판정중",
        "confidence": _fmt(last["confidence"], 3),
        "bars_in_regime": None if pd.isna(last["bars_in"]) else int(last["bars_in"]),
        "evidence": {
            "adx": _fmt(f.get("adx"), 2),
            "efficiency_ratio": _fmt(f.get("efficiency_ratio"), 3),
            "choppiness": _fmt(f.get("choppiness"), 2),
            "bbw_percentile": _fmt(f.get("bbw_percentile"), 1),
            "atr_ratio": _fmt(f.get("atr_ratio"), 3),
            "ema_spread": _fmt(f.get("ema_spread"), 3),
        },
    }


@server.tool(
    name="our_signals",
    description=(
        "전략별 현재 신호와 앙상블 합산 점수를 돌려줍니다. "
        "가중치는 config 의 전략 목록 기준이며, 국면 밖 전략은 발언권이 없습니다. "
        "이 신호로 실제 매매를 하지 마십시오 — 검증 5항목 전부 미달 상태입니다."
    ),
)
def our_signals(symbol: str, interval: str = "1h") -> dict:
    cfg = load_config()
    names = cfg.get("strategies")
    strategies = [s for s in build_all() if not names or s.name in set(names)]

    df = _load_bars(symbol, interval, live=False)
    feats = compute_all(df)
    reg = classify(feats)
    rid = reg["regime"].iloc[-1]
    rid = rid if isinstance(rid, str) else None

    votes = []
    for s in strategies:
        sig = s.generate(df, feats).iloc[-1]
        declared = bool(rid and any(str(r) == rid for r in s.regimes))
        votes.append({
            "strategy": s.name, "korean": s.korean,
            "direction": int(sig["direction"]) if declared else 0,
            "strength": _fmt(sig["strength"] if declared else 0.0, 3),
            "has_voice_in_regime": declared,
        })

    return {
        "symbol": symbol, "interval": interval,
        "bar_time": df.index[-1].isoformat(),
        "regime": KOREAN.get(Regime(rid), "판정중") if rid else "판정중",
        "votes": votes,
        "warning": (
            "실전 사용 금지. 5.9년 out-of-sample 검증에서 5항목 전부 미달했고 "
            "순열 검정에서 신호가 무작위 순서와 구분되지 않았습니다."
        ),
    }


@server.tool(
    name="list_fields",
    description="조회 가능한 지표 이름과 타임프레임 목록을 돌려줍니다.",
)
def list_fields() -> dict:
    return {
        "tradingview_fields": scanner.available_fields(),
        "intervals": scanner.available_intervals(),
        "our_indicator_count": len(compute_all(
            fetch_klines("BTCUSDT", "1d",
                         datetime.now(timezone.utc) - timedelta(days=400),
                         datetime.now(timezone.utc))).columns),
    }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
