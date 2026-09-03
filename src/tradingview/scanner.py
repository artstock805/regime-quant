"""
트레이딩뷰 스캐너 클라이언트.

`scanner.tradingview.com` 은 로그인·API 키 없이 종목별 **트레이딩뷰가 계산한**
지표값을 돌려준다. 우리가 직접 구현한 63종 지표를 대조할 기준값이 여기서 나온다.

이게 필요한 이유: 지표를 직접 짜면 미래 참조는 테스트로 잡을 수 있지만
**구현이 표준과 다른 것**은 잡을 수 없다. ADX 의 평활을 Wilder 가 아니라 단순평균으로
썼다든가, ATR 의 초기값 처리가 다르다든가 하는 차이는 자기 자신과는 일관되므로
내부 검증을 통과한다. 외부 기준이 있어야 드러난다.

한계
  · 캔들 히스토리는 주지 않는다. **현재 시점의 지표값 스냅샷**만 나온다.
    따라서 과거 전 구간 대조는 불가능하고, 지금 값이 맞는지만 확인할 수 있다.
  · 공식 문서가 없는 준공개 엔드포인트다. 스키마가 바뀔 수 있다.
  · 트레이딩뷰 이용약관상 개인적 용도로만 쓴다. 재배포·상업적 재판매는 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import requests

SCANNER = "https://scanner.tradingview.com/{market}/scan"
UA = "Mozilla/5.0 (regime-quant indicator cross-check)"

# 트레이딩뷰 컬럼은 '|<분>' 접미사로 타임프레임을 지정한다. 접미사가 없으면 일봉.
INTERVAL_SUFFIX = {
    "1m": "|1", "5m": "|5", "15m": "|15", "30m": "|30",
    "1h": "|60", "2h": "|120", "4h": "|240", "1d": "",
    "1w": "|1W", "1M": "|1M",
}

# 우리 지표 이름 → 트레이딩뷰 컬럼 이름.
# 대조 가능한 것만 넣는다. 트레이딩뷰에 없는 지표(taker_imbalance 등)는 대조 대상이 아니다.
COLUMN_MAP: dict[str, str] = {
    "close": "close",
    "rsi": "RSI",
    "adx": "ADX",
    "di_plus": "ADX+DI",
    "di_minus": "ADX-DI",
    "atr": "ATR",
    "macd": "MACD.macd",
    "macd_signal": "MACD.signal",
    "bb_upper": "BB.upper",
    "bb_lower": "BB.lower",
    "stoch_k": "Stoch.K",
    "stoch_d": "Stoch.D",
    "cci": "CCI20",
    "williams_r": "W.R",
    "mfi": "MoneyFlow",
    "momentum": "Mom",
    "awesome_osc": "AO",
    "ema20": "EMA20",
    "ema50": "EMA50",
    "sma200": "SMA200",
    "volume": "volume",
    "change_pct": "change",
}


class TradingViewError(RuntimeError):
    pass


@dataclass(frozen=True)
class Quote:
    """한 종목의 지표 스냅샷."""
    symbol: str
    interval: str
    values: dict[str, float | None]

    def get(self, name: str) -> float | None:
        return self.values.get(name)

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "interval": self.interval, **self.values}


def _ticker(symbol: str, exchange: str = "BINANCE", perpetual: bool = True) -> str:
    """BTCUSDT → BINANCE:BTCUSDT.P (무기한선물). 이미 완전한 티커면 그대로 둔다."""
    if ":" in symbol:
        return symbol
    suffix = ".P" if perpetual else ""
    return f"{exchange}:{symbol}{suffix}"


def fetch(
    symbols: Iterable[str],
    fields: Iterable[str] | None = None,
    interval: str = "1h",
    market: str = "crypto",
    exchange: str = "BINANCE",
    perpetual: bool = True,
    timeout: int = 20,
) -> list[Quote]:
    """
    지표 스냅샷을 받아온다.

    fields 는 **우리 쪽 이름**(rsi, adx, atr...)으로 준다. COLUMN_MAP 이 트레이딩뷰
    컬럼으로 번역한다. 매핑에 없는 이름을 주면 조용히 넘기지 않고 바로 터진다.
    """
    if interval not in INTERVAL_SUFFIX:
        raise TradingViewError(
            f"지원하지 않는 타임프레임: {interval} (가능: {', '.join(INTERVAL_SUFFIX)})"
        )
    names = list(fields) if fields is not None else list(COLUMN_MAP)
    unknown = [n for n in names if n not in COLUMN_MAP]
    if unknown:
        raise TradingViewError(
            f"대조 가능한 지표가 아닙니다: {unknown}\n"
            f"가능: {', '.join(sorted(COLUMN_MAP))}"
        )

    suffix = INTERVAL_SUFFIX[interval]
    columns = [COLUMN_MAP[n] + suffix for n in names]
    tickers = [_ticker(s, exchange, perpetual) for s in symbols]

    body = {
        "symbols": {"tickers": tickers, "query": {"types": []}},
        "columns": columns,
    }
    try:
        resp = requests.post(
            SCANNER.format(market=market), json=body, timeout=timeout,
            headers={"User-Agent": UA},
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise TradingViewError(f"스캐너 요청 실패: {exc}") from exc
    except ValueError as exc:
        raise TradingViewError(f"스캐너 응답이 JSON 이 아닙니다: {exc}") from exc

    rows = payload.get("data") or []
    if not rows:
        raise TradingViewError(
            f"응답에 데이터가 없습니다. 티커를 확인하세요: {tickers}"
        )

    out = []
    for row in rows:
        vals = row.get("d") or []
        out.append(Quote(
            symbol=row.get("s", "?"),
            interval=interval,
            values={n: (v if isinstance(v, (int, float)) else None)
                    for n, v in zip(names, vals)},
        ))
    return out


def fetch_one(symbol: str, **kwargs) -> Quote:
    quotes = fetch([symbol], **kwargs)
    if not quotes:
        raise TradingViewError(f"{symbol} 데이터를 받지 못했습니다")
    return quotes[0]


def available_fields() -> list[str]:
    return sorted(COLUMN_MAP)


def available_intervals() -> list[str]:
    return list(INTERVAL_SUFFIX)
