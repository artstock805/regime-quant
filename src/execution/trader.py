"""
자동매매 루프 — 신호를 받아 실제로 진입·청산한다.

**기본은 테스트넷이다.** 실전은 설정에서 명시적으로 켜야 한다.

한 바퀴에 하는 일
    1) 거래소에서 **실제 포지션**을 읽는다 (내 기록을 믿지 않는다)
    2) 내 기록과 어긋나면 기록을 고치고 장부에 남긴다
    3) 신호를 계산한다 (백테스트와 같은 규칙 — 마감봉 기준)
    4) 청산 신호가 있으면 먼저 닫는다
    5) 진입 신호는 리스크 가드를 통과해야 나간다
    6) 진입 직후 **손절·목표를 거래소에 건다**

6번이 특히 중요하다. 프로그램이 죽거나 인터넷이 끊겨도 포지션이 맨몸으로
남지 않는다. 스톱을 로컬에서 관리하는 봇은 꺼지는 순간 무방비가 된다.

신호 출처는 두 가지이고 같은 경로로 합류한다.
    · 우리 엔진 (src/live/signals.py) — 봉 마감마다
    · 트레이딩뷰 Pine 알림 → 웹훅 (src/server/webhook.py)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..config import Config, load_config
from ..data.binance import interval_timedelta
from ..live import scan
from .exchange import BinanceFutures, ExchangeError, MissingCredentials
from .guards import RiskGuard, RiskLimits
from .journal import Journal

OPPOSITE = {"LONG": "SELL", "SHORT": "BUY"}
SIDE_OF = {"롱진입": "LONG", "숏진입": "SHORT"}


def client_id(symbol: str, bar_time: str, tag: str) -> str:
    """
    **멱등성 열쇠.** 같은 종목·같은 봉·같은 목적이면 항상 같은 id 가 나온다.
    루프가 두 번 돌거나 재시작해도 주문이 두 번 나가지 않는다.
    바이낸스 clientOrderId 는 36자 제한이라 짧게 만든다.
    """
    stamp = bar_time.replace("-", "").replace(":", "")[2:13]   # YYMMDDTHHMM
    sym = symbol.replace("USDT", "")[:8]
    return f"rq{tag}{sym}{stamp}"[:36]


@dataclass
class TraderConfig:
    testnet: bool = True
    interval: str = "1h"
    symbols: list[str] = field(default_factory=list)
    use_derivatives: bool = True
    # 봉 마감 후 몇 초 뒤에 판단할지. 거래소가 봉을 확정하는 데 시간이 걸린다.
    settle_seconds: int = 20
    # 웹훅 신호도 받을지
    accept_webhook: bool = True


class Trader:
    """자동매매 엔진. 켜기 전까지는 아무것도 하지 않는다."""

    def __init__(self, cfg: Config | None = None,
                 tcfg: TraderConfig | None = None,
                 exchange: BinanceFutures | None = None,
                 journal: Journal | None = None,
                 guard: RiskGuard | None = None) -> None:
        self.cfg = cfg or load_config()
        self.tcfg = tcfg or self._default_tcfg()
        self.journal = journal or Journal()
        self.guard = guard or RiskGuard(self._limits())
        self._exchange = exchange
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.last_cycle: str | None = None
        self.next_cycle: float | None = None
        self.cycles = 0
        self.last_error: str | None = None
        self.positions: dict[str, dict] = {}
        self.balance: dict[str, float] = {"wallet": 0.0, "available": 0.0}
        # 봉시각별로 이미 처리했는지. 같은 봉을 두 번 처리하지 않는다.
        self._handled: dict[str, str] = {}

    def _default_tcfg(self) -> TraderConfig:
        live = self.cfg.get("live", {}) or {}
        return TraderConfig(
            testnet=bool(self.cfg.get("execution.testnet", True)),
            interval=str(live.get("interval", "1h")),
            symbols=list(live.get("symbols") or self.cfg.get("universe") or []),
        )

    def _limits(self) -> RiskLimits:
        r = self.cfg.get("risk", {}) or {}
        e = self.cfg.get("execution", {}) or {}
        return RiskLimits(
            risk_per_trade=float(r.get("risk_per_trade", 0.01)),
            max_positions=int(e.get("max_positions", 3)),
            daily_loss_limit=float(r.get("daily_loss_limit", 0.03)),
            max_notional_ratio=float(e.get("max_notional_ratio", 1.0)),
            max_orders_per_minute=int(e.get("max_orders_per_minute", 6)),
            min_balance=float(e.get("min_balance", 20.0)),
        )

    @property
    def exchange(self) -> BinanceFutures:
        if self._exchange is None:
            self._exchange = BinanceFutures(testnet=self.tcfg.testnet)
        return self._exchange

    # ─── 제어 ────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def running(self) -> bool:
        """끄라고 한 순간 꺼진 것으로 본다 (감시 토글과 같은 규칙)."""
        return self.alive and not self._stop.is_set()

    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self.running:
                return False, "이미 실행 중입니다."
            if self.alive:
                self._thread.join(timeout=5)
                if self.alive:
                    return False, "직전 실행을 정리하는 중입니다."
            # 키가 없으면 시작하지 않는다. 돌다가 첫 주문에서 실패하는 것보다
            # 시작 자체를 거절하는 편이 낫다.
            try:
                self.exchange.sync_time()
                self.balance = self.exchange.balance()
            except MissingCredentials as exc:
                return False, str(exc)
            except ExchangeError as exc:
                return False, f"거래소에 연결하지 못했습니다: {exc}"

            self.guard.release_kill()
            self._stop.clear()
            self.last_error = None
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        mode = "테스트넷" if self.tcfg.testnet else "실전"
        self.journal.write("engine", event="start", mode=mode,
                           symbols=self.tcfg.symbols, interval=self.tcfg.interval)
        return True, f"{mode}에서 자동매매를 시작했습니다."

    def stop(self, flatten: bool = False) -> tuple[bool, str]:
        if not self.alive:
            return False, "실행 중이 아닙니다."
        self._stop.set()
        self.next_cycle = None
        self.journal.write("engine", event="stop", flatten=flatten)
        if flatten:
            closed = self.flatten_all("사용자 요청")
            return True, f"중지했습니다. 포지션 {closed}건을 정리했습니다."
        return True, "중지했습니다. 열린 포지션은 그대로 둡니다."

    def kill(self, reason: str = "사용자 요청") -> dict:
        """
        **킬 스위치.** 새 진입을 즉시 막는다. 엔진이 돌고 있어도 막힌다.
        포지션 정리는 따로 눌러야 한다 — 시장가로 던지는 것도 손실이기 때문이다.
        """
        self.guard.engage_kill(reason)
        self.journal.write("engine", event="kill", reason=reason)
        return self.guard.status(self.balance.get("wallet", 0.0))

    def flatten_all(self, reason: str = "") -> int:
        """열린 포지션을 전부 시장가로 닫는다. 스톱 주문도 함께 취소한다."""
        n = 0
        try:
            positions = self.exchange.positions()
        except ExchangeError as exc:
            self.journal.write("error", where="flatten", message=str(exc))
            return 0
        for sym, pos in positions.items():
            try:
                self.exchange.cancel_all(sym)
                self.exchange.place_market(
                    sym, OPPOSITE[pos["side"]], pos["qty"],
                    client_id=client_id(sym, datetime.now(timezone.utc).isoformat(),
                                        "F"),
                    reduce_only=True)
                self.journal.write("close", symbol=sym, qty=pos["qty"],
                                   side=pos["side"], reason=reason or "정리",
                                   realised=pos.get("unrealised", 0.0))
                n += 1
            except ExchangeError as exc:
                self.journal.write("error", where="flatten", symbol=sym,
                                   message=str(exc))
        return n

    # ─── 루프 ────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:                     # 한 바퀴 실패로 죽지 않는다
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.journal.write("error", where="cycle", message=self.last_error)
            finally:
                self.cycles += 1
                self.last_cycle = datetime.now(timezone.utc).isoformat()

            wait = self._seconds_to_next_bar()
            self.next_cycle = datetime.now(timezone.utc).timestamp() + wait
            if self._stop.wait(wait):
                break
        self.next_cycle = None

    def _seconds_to_next_bar(self) -> float:
        """
        다음 봉 마감 + settle_seconds 까지 기다린다.

        고정 간격으로 돌면 봉 경계와 어긋나 같은 봉을 두 번 보거나 건너뛴다.
        백테스트가 '봉 마감 기준'이므로 실행도 봉에 맞춰야 숫자가 일치한다.
        """
        step = interval_timedelta(self.tcfg.interval).total_seconds()
        now = datetime.now(timezone.utc).timestamp()
        next_close = (int(now // step) + 1) * step
        return max(5.0, next_close + self.tcfg.settle_seconds - now)

    # ─── 한 바퀴 ─────────────────────────────────────────────────

    def run_once(self) -> dict:
        """한 바퀴. 테스트와 수동 실행에서도 이걸 부른다."""
        # 1) 거래소가 진실이다
        self.balance = self.exchange.balance()
        actual = self.exchange.positions()
        self._reconcile(actual)
        self.positions = actual

        # 2) 신호
        signals = scan(self.cfg, self.tcfg.symbols,
                       interval=self.tcfg.interval,
                       use_derivatives=self.tcfg.use_derivatives)
        acted = []
        for s in signals:
            # 같은 봉을 두 번 처리하지 않는다
            if self._handled.get(s.symbol) == s.bar_time:
                continue
            self._handled[s.symbol] = s.bar_time
            self.journal.write("signal", symbol=s.symbol, action=s.action,
                               score=s.score, bar_time=s.bar_time,
                               regime=s.regime_korean)
            r = self._act(s, actual)
            if r:
                acted.append(r)
        return {"signals": len(signals), "acted": acted,
                "positions": len(actual), "balance": self.balance}

    def _reconcile(self, actual: dict[str, dict]) -> None:
        """
        내 기록과 거래소가 다르면 **거래소를 믿고** 기록을 고친다.
        다르다는 사실 자체를 장부에 남긴다 — 자주 어긋나면 원인이 있다.
        """
        mine, theirs = set(self.positions), set(actual)
        if mine == theirs:
            return
        self.journal.write("reconcile",
                           mine=sorted(mine), exchange=sorted(theirs),
                           only_mine=sorted(mine - theirs),
                           only_exchange=sorted(theirs - mine))

    def _act(self, sig, actual: dict[str, dict]) -> dict | None:
        pos = actual.get(sig.symbol)

        # ── 청산이 먼저다. 자리를 비워야 새로 들어갈 수 있다.
        if sig.action == "청산" and pos:
            return self._close(sig.symbol, pos, "신호")

        if sig.action not in SIDE_OF:
            return None
        want = SIDE_OF[sig.action]

        # 이미 같은 방향이면 아무것도 하지 않는다
        if pos and pos["side"] == want:
            return None
        # 반대 방향이면 먼저 닫는다
        if pos:
            self._close(sig.symbol, pos, "방향 전환")

        return self._open(sig, want)

    def _open(self, sig, side: str) -> dict | None:
        if sig.stop_price is None or not sig.risk_per_unit:
            return None
        balance = self.balance.get("available", 0.0)
        qty, why = self.guard.position_size(balance=balance, entry=sig.close,
                                            stop=sig.stop_price)
        if qty <= 0:
            self.journal.write("blocked", symbol=sig.symbol, reason=why)
            return None

        rules = self.exchange.rules(sig.symbol)
        qty = rules.round_qty(qty)
        ok, why = rules.is_tradable(qty, sig.close)
        if not ok:
            self.journal.write("blocked", symbol=sig.symbol, reason=why, qty=qty)
            return None

        decision = self.guard.check_entry(
            balance=balance, open_positions=len(self.positions),
            notional=qty * sig.close, symbol=sig.symbol)
        if not decision:
            self.journal.write("blocked", symbol=sig.symbol,
                               reason=decision.reason, qty=qty)
            return None

        entry_side = "BUY" if side == "LONG" else "SELL"
        exit_side = OPPOSITE[side]
        try:
            order = self.exchange.place_market(
                sig.symbol, entry_side, qty,
                client_id=client_id(sig.symbol, sig.bar_time, "E"))
            self.guard.record_order()
            self.journal.write("order", symbol=sig.symbol, side=entry_side,
                               qty=qty, kind_="entry", price=sig.close,
                               client_id=order.get("clientOrderId"),
                               duplicate=order.get("duplicate", False))

            # **진입 직후 손절·목표를 거래소에 건다.**
            # 프로그램이 죽어도 포지션이 맨몸으로 남지 않는다.
            self.exchange.place_stop(
                sig.symbol, exit_side, sig.stop_price,
                client_id=client_id(sig.symbol, sig.bar_time, "S"))
            if sig.target_price:
                self.exchange.place_target(
                    sig.symbol, exit_side, sig.target_price,
                    client_id=client_id(sig.symbol, sig.bar_time, "T"))
        except ExchangeError as exc:
            self.journal.write("reject", symbol=sig.symbol, side=entry_side,
                               qty=qty, message=str(exc))
            return None

        return {"symbol": sig.symbol, "action": sig.action, "qty": qty,
                "stop": sig.stop_price, "target": sig.target_price}

    def _close(self, symbol: str, pos: dict, reason: str) -> dict | None:
        try:
            # 스톱·목표를 먼저 취소한다. 안 그러면 청산 뒤에 남아 반대 포지션을 연다.
            self.exchange.cancel_all(symbol)
            self.exchange.place_market(
                symbol, OPPOSITE[pos["side"]], pos["qty"],
                client_id=client_id(symbol, datetime.now(timezone.utc).isoformat(),
                                    "X"),
                reduce_only=True)
            self.guard.record_order()
        except ExchangeError as exc:
            self.journal.write("reject", symbol=symbol, kind_="close",
                               message=str(exc))
            return None

        realised = float(pos.get("unrealised", 0.0))
        self.guard.record_realised(realised)
        self.journal.write("close", symbol=symbol, side=pos["side"],
                           qty=pos["qty"], reason=reason, realised=realised)
        return {"symbol": symbol, "action": "청산", "qty": pos["qty"],
                "realised": realised}

    # ─── 웹훅에서 들어온 신호 ────────────────────────────────────

    def handle_external(self, symbol: str, action: str, *,
                        source: str = "tradingview") -> dict:
        """
        트레이딩뷰 Pine 알림이 들어왔을 때. **우리 엔진과 같은 관문을 지난다** —
        리스크 가드, 수량 계산, 손절 주문까지 전부 동일하다. 외부 신호라고
        가드를 건너뛰면 그게 가장 위험한 구멍이 된다.
        """
        if not self.tcfg.accept_webhook:
            return {"ok": False, "message": "웹훅 신호를 받지 않도록 설정돼 있습니다"}
        if not self.running:
            return {"ok": False, "message": "자동매매가 꺼져 있습니다"}

        from ..live import signal_for
        sig = signal_for(self.cfg, symbol, interval=self.tcfg.interval,
                         use_derivatives=self.tcfg.use_derivatives)
        if sig is None:
            return {"ok": False, "message": f"{symbol}: 데이터가 부족합니다"}

        # 동작만 외부 신호로 갈아끼운다. 가격·손절·목표는 우리가 계산한 것을 쓴다 —
        # 알림이 보낸 가격을 그대로 믿으면 지연·오타가 그대로 주문이 된다.
        sig.action = action
        sig.direction = {"롱진입": 1, "숏진입": -1, "청산": 0}.get(action, 0)
        self.journal.write("signal", symbol=symbol, action=action,
                           source=source, bar_time=sig.bar_time)

        actual = self.exchange.positions()
        self.positions = actual
        self.balance = self.exchange.balance()
        result = self._act(sig, actual)
        return {"ok": result is not None, "result": result,
                "message": "처리했습니다" if result else "조건에 맞지 않아 실행하지 않았습니다"}

    # ─── 상태 ────────────────────────────────────────────────────

    def status(self) -> dict:
        wallet = self.balance.get("wallet", 0.0)
        return {
            "running": self.running,
            "stopping": self.alive and self._stop.is_set(),
            "mode": "테스트넷" if self.tcfg.testnet else "실전",
            "testnet": self.tcfg.testnet,
            "interval": self.tcfg.interval,
            "symbols": self.tcfg.symbols,
            "cycles": self.cycles,
            "last_cycle": self.last_cycle,
            "next_cycle": self.next_cycle,
            "last_error": self.last_error,
            "balance": self.balance,
            "positions": list(self.positions.values()),
            "guard": self.guard.status(wallet),
            "journal": self.journal.summary(),
            "accept_webhook": self.tcfg.accept_webhook,
        }
