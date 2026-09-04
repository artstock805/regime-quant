"""
실행 계층 — 실제로 주문을 낸다.

**기본은 테스트넷이다.** 실전으로 바꾸려면 config 의 execution.testnet 을
false 로 두고 실전 키를 따로 넣어야 한다. 기본값이 실전이면 실수 한 번이 돈이 된다.
"""

from .exchange import (
    BinanceFutures, ExchangeError, MissingCredentials, SymbolRules,
    load_credentials,
)
from .guards import Decision, RiskGuard, RiskLimits
from .journal import Journal
from .trader import Trader, TraderConfig, client_id

__all__ = [
    "BinanceFutures", "ExchangeError", "MissingCredentials", "SymbolRules",
    "load_credentials", "Decision", "RiskGuard", "RiskLimits", "Journal",
    "Trader", "TraderConfig", "client_id",
]
