"""
MetaTrader 5 연결 — FX 데이터와 주문을 한 경로로 처리한다.

**왜 MT5 인가.** FX 는 암호화폐와 달리 무료 벌크 아카이브가 없다. 브로커마다
호가가 다르고, 우리가 주문을 낼 브로커의 호가로 신호를 계산해야 체결과 어긋나지
않는다. MT5 는 그 두 가지(데이터·주문)를 같은 연결에서 준다.

**스레드 직렬화가 필수다.** MetaTrader5 파이썬 패키지는 스레드 안전하지 않다.
웹 서버는 여러 스레드에서 요청을 처리하므로, 락 없이 부르면 값이 섞이거나
터미널이 응답을 멈춘다. 이 파일의 모든 호출이 하나의 락을 지난다.

**차트는 여기서 그리지 않는다.** 화면의 차트는 트레이딩뷰 위젯이 담당한다.
MT5 는 신호 계산과 주문에만 쓴다. 다만 두 시세는 브로커가 달라 미세하게
다르므로, 화면에 그 사실을 표시한다 (symbols.py 참고).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# 우리 인터벌 표기 → MT5 상수. 지연 임포트라 여기서는 문자열로만 둔다.
TIMEFRAMES = {
    "1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5", "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30", "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}

_lock = threading.RLock()
_initialised = False


class MT5Error(RuntimeError):
    """터미널이 없거나, 로그인이 안 됐거나, 호출이 거부됐다."""


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:                    # pragma: no cover
        raise MT5Error(
            "MetaTrader5 패키지가 없습니다. pip install MetaTrader5") from exc
    return mt5


def initialise() -> bool:
    """
    터미널에 붙는다. 이미 붙어 있으면 아무것도 하지 않는다.

    MT5 는 **터미널이 실행 중이고 계정에 로그인돼 있어야** 한다. 그렇지 않으면
    여기서 바로 알려 준다 — 조회가 조용히 빈 값을 돌려주는 것보다 낫다.
    """
    global _initialised
    with _lock:
        if _initialised:
            return True
        mt5 = _mt5()
        if not mt5.initialize():
            code, msg = mt5.last_error()
            raise MT5Error(
                f"MT5 터미널에 붙지 못했습니다 [{code}] {msg}\n"
                "  · MetaTrader 5 를 실행하고 계정에 로그인하세요\n"
                "  · 도구 > 옵션 > Expert Advisors 에서 자동매매 허용을 켜세요")
        _initialised = True
        return True


def shutdown() -> None:
    global _initialised
    with _lock:
        if _initialised:
            _mt5().shutdown()
            _initialised = False


@dataclass
class Account:
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin_free: float
    is_demo: bool
    trade_allowed: bool
    leverage: int

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def account() -> Account:
    with _lock:
        initialise()
        mt5 = _mt5()
        a = mt5.account_info()
        t = mt5.terminal_info()
        if a is None:
            raise MT5Error("계정 정보를 읽지 못했습니다 — 로그인 상태를 확인하세요")
        return Account(
            login=a.login, server=a.server, currency=a.currency,
            balance=a.balance, equity=a.equity, margin_free=a.margin_free,
            # trade_mode 0 = 데모. 실계좌를 데모로 착각하면 안 되므로 명시적으로 본다.
            is_demo=(a.trade_mode == 0),
            trade_allowed=bool(t and t.trade_allowed),
            leverage=a.leverage,
        )


@dataclass
class SymbolInfo:
    name: str
    digits: int
    point: float
    spread_points: int
    spread_price: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    bid: float
    ask: float

    @property
    def spread_pips(self) -> float:
        """5자리/3자리 호가에서 1핍 = 10포인트."""
        return self.spread_points / (10 if self.digits in (3, 5) else 1)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d["spread_pips"] = round(self.spread_pips, 2)
        return d


def symbol_info(name: str) -> SymbolInfo:
    with _lock:
        initialise()
        mt5 = _mt5()
        if not mt5.symbol_select(name, True):
            raise MT5Error(f"{name} 을 Market Watch 에 추가하지 못했습니다")
        s = mt5.symbol_info(name)
        if s is None:
            raise MT5Error(f"{name} 정보를 읽지 못했습니다")
        return SymbolInfo(
            name=s.name, digits=s.digits, point=s.point,
            spread_points=s.spread, spread_price=s.spread * s.point,
            volume_min=s.volume_min, volume_max=s.volume_max,
            volume_step=s.volume_step, contract_size=s.trade_contract_size,
            bid=s.bid, ask=s.ask,
        )


def available_symbols(only_forex: bool = True) -> list[str]:
    """
    거래 가능한 종목. FX 만 걸러 낸다 — 이 앱은 FX 전용이다.
    브로커마다 접미사가 다르므로(EURUSD, EURUSD., EURUSDm) 경로로 판단한다.
    """
    with _lock:
        initialise()
        out = []
        for s in _mt5().symbols_get():
            if only_forex and not s.path.lower().startswith("forex"):
                continue
            out.append(s.name)
        return sorted(out)


def bars(symbol: str, interval: str = "1h", count: int = 1500) -> pd.DataFrame:
    """
    최근 봉. **마감된 봉만 돌려준다** — 진행 중인 봉을 섞으면 신호가 매초 흔들리고
    백테스트 규칙(마감봉 기준)과 어긋난다.

    반환 열은 백테스트가 쓰는 것과 같은 이름으로 맞춘다.
    """
    tf_name = TIMEFRAMES.get(interval)
    if tf_name is None:
        raise MT5Error(f"지원하지 않는 봉: {interval} (가능: {', '.join(TIMEFRAMES)})")

    with _lock:
        initialise()
        mt5 = _mt5()
        if not mt5.symbol_select(symbol, True):
            raise MT5Error(f"{symbol} 을 Market Watch 에 추가하지 못했습니다")
        # +1 을 받아 마지막(진행 중) 봉을 버린다
        raw = mt5.copy_rates_from_pos(symbol, getattr(mt5, tf_name), 0, count + 1)

    if raw is None or len(raw) == 0:
        raise MT5Error(f"{symbol} {interval} 데이터를 받지 못했습니다")

    df = pd.DataFrame(raw)
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume", "real_volume": "real_volume"})
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume", "spread"]]
    # 마지막 봉은 아직 진행 중이다. 버린다.
    return df.iloc[:-1].astype({"open": "float64", "high": "float64",
                                "low": "float64", "close": "float64",
                                "volume": "float64"})


def tick(symbol: str) -> dict[str, Any]:
    """현재 호가. 화면의 '지금 값'에만 쓴다 — 신호 계산에는 쓰지 않는다."""
    with _lock:
        initialise()
        mt5 = _mt5()
        mt5.symbol_select(symbol, True)
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5Error(f"{symbol} 호가를 읽지 못했습니다")
        return {"symbol": symbol, "bid": t.bid, "ask": t.ask,
                "spread": round(t.ask - t.bid, 8),
                "at": datetime.fromtimestamp(t.time, timezone.utc).isoformat()}


def positions() -> dict[str, dict]:
    """
    **열린 포지션. 이것이 진실이다.** 내부 기록과 다르면 이쪽을 믿는다.
    같은 종목에 여러 포지션이 있을 수 있으므로 티켓 단위로 돌려준다.
    """
    with _lock:
        initialise()
        rows = _mt5().positions_get()
    out = {}
    for p in (rows or []):
        out[str(p.ticket)] = {
            "ticket": p.ticket, "symbol": p.symbol,
            "side": "LONG" if p.type == 0 else "SHORT",
            "volume": p.volume, "entry_price": p.price_open,
            "current_price": p.price_current,
            "stop_loss": p.sl or None, "take_profit": p.tp or None,
            "profit": p.profit, "swap": p.swap,
            # magic 으로 우리가 연 포지션을 구분한다. 사람이 손으로 연 것을
            # 우리 것으로 착각해 청산하면 그건 사고다.
            "magic": getattr(p, "magic", 0),
            "opened_at": datetime.fromtimestamp(p.time, timezone.utc).isoformat(),
            "comment": p.comment,
        }
    return out


def status() -> dict:
    """화면이 물어보는 한 덩어리. 실패해도 예외를 올리지 않고 이유를 담는다."""
    try:
        acct = account()
    except MT5Error as exc:
        return {"connected": False, "message": str(exc)}
    try:
        pos = positions()
    except MT5Error:
        pos = {}
    return {
        "connected": True,
        "account": acct.as_dict(),
        "positions": list(pos.values()),
        "message": "",
    }
