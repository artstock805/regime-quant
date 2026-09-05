"""
FX 전용 계층 — 데이터·주문은 MT5, 차트는 트레이딩뷰.

이 앱은 암호화폐를 다루지 않는다. 그쪽 코드(src/execution, src/data)는
백테스트·복기용으로 남아 있지만 자동매매 경로에서는 쓰지 않는다.
"""

from .broker import (
    MAGIC, Fill, LiveAccountBlocked, OrderRejected, assert_tradable,
    close_all, close_position, lots_for_risk, move_stop, open_position,
    ours, reconcile, tag,
)
from .mt5 import (
    Account, MT5Error, SymbolInfo, account, available_symbols, bars,
    initialise, positions, shutdown, status, symbol_info, tick,
)
from .symbols import (
    CROSSES, DEFAULT_PROVIDER, MAJORS, PROVIDERS, Pair, catalogue, parse,
    resolve, to_pips,
)

__all__ = [
    "MAGIC", "Fill", "LiveAccountBlocked", "OrderRejected", "assert_tradable",
    "close_all", "close_position", "lots_for_risk", "move_stop",
    "open_position", "ours", "reconcile", "tag",
    "Account", "MT5Error", "SymbolInfo", "account", "available_symbols", "bars",
    "initialise", "positions", "shutdown", "status", "symbol_info", "tick",
    "CROSSES", "DEFAULT_PROVIDER", "MAJORS", "PROVIDERS", "Pair", "catalogue",
    "parse", "resolve", "to_pips",
]
