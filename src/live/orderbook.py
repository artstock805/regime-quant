"""
호가창·체결 중계 — **서버가 거래소 연결을 하나 유지하고 상태를 쌓는다.**

왜 서버가 하는가
    브라우저 직결(이전 방식)은 보기에는 충분했지만 두 가지를 못 한다.
      · 탭을 닫으면 누적 델타가 사라진다.
      · 전략이 그 데이터를 읽을 수 없다.
    서버가 쥐고 있으면 브라우저가 몇 개 붙어도 거래소 연결은 하나고, 누적이
    탭과 무관하게 이어지며, 엔진이 같은 상태를 읽는다.

**여기서 만드는 것이 '진짜' 풋프린트다.** 앞서 봉 단위로 합산한 테이커
불균형은 우위가 없었는데(7쌍 IC -0.021, 5.9년 검증 1/4 관문), 봉으로 뭉개는
순간 실무자들이 보는 구조 — **가격대별 델타** — 가 사라진다. 여기서는 체결을
가격 눈금마다 따로 세므로 그 구조가 남는다. 그것이 값을 낼지는 아직 모르고,
재 보려면 이 상태를 먼저 쌓아야 한다.

스레드 모델
    이 프로젝트는 대체로 동기 코드다(MT5 가 동기이고 락으로 직렬화한다).
    그래서 거래소 연결만 **전용 데몬 스레드의 asyncio 루프**에서 돌리고,
    바깥에서는 평범한 동기 함수로 읽는다. FastAPI 의 동기 엔드포인트든 엔진
    스레드든 `subscribe()` 를 그냥 부를 수 있게 하려는 것이다. 상태 변경과
    읽기는 종목별 락으로 감싼다.

메모리
    무한히 도는 연결이라 상한을 반드시 둔다. 체결 테이프는 최근 500건,
    풋프린트는 현재가 근처 눈금만 남긴다(먼 것부터 버린다). 상한이 없으면
    며칠 돌린 뒤 서버가 죽는다.

@aggTrade 를 안 쓰는 이유
    브라우저에서도 서버에서도 8~10초에 **0건**이 왔다(연결은 열리고 오류도
    없다). 같은 조건에서 @trade 는 173건, @depth20 은 78건이 정상으로 왔고
    REST 로 확인한 체결은 초당 여러 건이었다. 즉 데이터가 없는 게 아니라 그
    스트림만 안 온다. @trade 는 체결을 하나씩 주므로 오히려 세밀하고 m 필드가
    똑같이 있어 잃는 것이 없다.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

WS_BASE = "wss://fstream.binance.com/stream?streams="
LEVELS = 20                 # 호가 단수. 부분 스냅샷이라 시퀀스 관리가 필요 없다.
TAPE_MAX = 500              # 체결 테이프 상한
FOOT_MAX = 600              # 풋프린트 눈금 상한 (먼 것부터 버린다)
RECONNECT_MAX = 30.0        # 재연결 대기 상한(초)

# 우리가 아는 종목만 연다. 문자열을 그대로 URL 에 넣으면 남이 임의의 스트림을
# 열게 만들 수 있고, 오타 하나로 조용히 빈 화면이 된다.
ALLOWED = ("btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt", "dogeusdt",
           "adausdt", "linkusdt", "avaxusdt")


class UnknownSymbol(ValueError):
    """허용 목록에 없는 종목."""


def _now() -> float:
    return time.time()


class Stream:
    """한 종목의 호가창·체결 누적 상태."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._lock = threading.Lock()

        self.bids: list[list[str]] = []
        self.asks: list[list[str]] = []
        self.tape: deque[dict] = deque(maxlen=TAPE_MAX)
        # 가격 눈금 -> [공격매수, 공격매도]
        self.footprint: dict[float, list[float]] = {}

        self.cvd = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.n_trades = 0

        # 호가 눈금. depth 레벨 간격의 **최솟값**으로 추정한다. 레벨이 비어
        # 있으면 간격이 눈금의 배수로 보일 수 있어서, 계속 최솟값으로 줄인다.
        #
        # **문자열로 계산한다.** 거래소는 "80219.30" 처럼 정확한 십진수를 주는데
        # float 로 바꿔 빼면 0.1 이 0.09999999999126885 로 나온다. 그 값으로
        # 가격을 눈금에 맞추면 키가 미세하게 갈라져 같은 가격이 여러 칸이 된다.
        self.tick: float | None = None
        self._tick_digits = 0
        self._tick_dec: Decimal | None = None

        self.started_at = _now()
        self.last_msg_at: float | None = None
        self.connected = False
        self.error = ""
        self.reconnects = 0
        self._stop = False

    # ─── 상태 갱신 (거래소 스레드에서만 호출된다) ──────────────

    def _infer_tick(self) -> None:
        prices: list[Decimal] = []
        for side in (self.bids, self.asks):
            for level in side[:LEVELS]:
                try:
                    prices.append(Decimal(str(level[0])))
                except (InvalidOperation, IndexError, TypeError):
                    return
        prices.sort()
        gaps = [b - a for a, b in zip(prices, prices[1:]) if b > a]
        if not gaps:
            return
        smallest = min(gaps)
        if self._tick_dec is None or smallest < self._tick_dec:
            self._tick_dec = smallest
            self.tick = float(smallest)
            # 눈금의 소수 자릿수. exponent 는 Decimal("0.1") 이면 -1 이다.
            exp = smallest.as_tuple().exponent
            self._tick_digits = max(0, -int(exp)) if isinstance(exp, int) else 0

    def apply_depth(self, d: dict) -> None:
        with self._lock:
            b, a = d.get("b"), d.get("a")
            if isinstance(b, list):
                self.bids = b[:LEVELS]
            if isinstance(a, list):
                self.asks = a[:LEVELS]
            self._infer_tick()
            self.last_msg_at = _now()

    def apply_trade(self, d: dict) -> None:
        try:
            px = float(d["p"])
            qty = float(d["q"])
        except (KeyError, TypeError, ValueError):
            return
        # **m=true 는 "매수자가 메이커"다 → 공격자는 매도자.**
        # 이 부호를 뒤집으면 누적 델타가 정반대로 쌓인다.
        buy = d.get("m") is False

        with self._lock:
            self.n_trades += 1
            if buy:
                self.buy_vol += qty
                self.cvd += qty
            else:
                self.sell_vol += qty
                self.cvd -= qty

            self.tape.appendleft({
                "at": int(d.get("T") or 0), "px": px, "qty": qty, "buy": buy,
            })

            bucket = self._bucket(px)
            cell = self.footprint.get(bucket)
            if cell is None:
                cell = [0.0, 0.0]
                self.footprint[bucket] = cell
            cell[0 if buy else 1] += qty
            self._prune_footprint()
            self.last_msg_at = _now()

    def _bucket(self, px: float) -> float:
        """가격을 호가 눈금에 맞춘다. 눈금을 모르면 가격 그대로 쓴다."""
        t = self.tick
        if not t or t <= 0 or not math.isfinite(t):
            return px
        # 눈금 자릿수로 반올림해 키가 미세하게 갈라지는 것을 막는다.
        # 눈금이 문자열에서 나온 정확한 값이라 자릿수도 정확하다.
        return round(round(px / t) * t, self._tick_digits)

    def _prune_footprint(self) -> None:
        """상한을 넘으면 **현재가에서 먼 눈금부터** 버린다."""
        if len(self.footprint) <= FOOT_MAX:
            return
        mid = self._mid_locked()
        if mid is None:
            # 중간가를 모르면 오래된 것을 버릴 수단이 없다. 임의로 자르기보다
            # 눈금 개수만 맞춘다 (키 정렬 기준으로 양끝을 버린다).
            keys = sorted(self.footprint)
            drop = len(keys) - FOOT_MAX
            for k in keys[:drop]:
                self.footprint.pop(k, None)
            return
        far = sorted(self.footprint, key=lambda k: abs(k - mid), reverse=True)
        for k in far[:len(self.footprint) - FOOT_MAX]:
            self.footprint.pop(k, None)

    def _mid_locked(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        try:
            return (float(self.bids[0][0]) + float(self.asks[0][0])) / 2.0
        except (IndexError, TypeError, ValueError):
            return None

    def reset(self) -> None:
        """누적만 지운다. 호가창과 연결은 그대로 둔다."""
        with self._lock:
            self.cvd = 0.0
            self.buy_vol = 0.0
            self.sell_vol = 0.0
            self.n_trades = 0
            self.tape.clear()
            self.footprint.clear()
            self.started_at = _now()

    # ─── 읽기 (아무 스레드에서나) ────────────────────────────────

    def snapshot(self, *, tape: int = 60, foot: int = 40) -> dict[str, Any]:
        """
        브라우저와 엔진이 읽는 한 덩어리.

        락 안에서 **복사해서** 내보낸다. 참조를 넘기면 거래소 스레드가 같은
        리스트를 고치는 중에 직렬화되어 값이 섞인다.
        """
        with self._lock:
            mid = self._mid_locked()
            rows = []
            if mid is not None and self.footprint:
                near = sorted(self.footprint, key=lambda k: abs(k - mid))[:foot]
                for px in sorted(near, reverse=True):
                    b, s = self.footprint[px]
                    rows.append({"px": px, "buy": round(b, 6),
                                 "sell": round(s, 6), "delta": round(b - s, 6)})
            return {
                "symbol": self.symbol,
                "connected": self.connected,
                "error": self.error,
                "reconnects": self.reconnects,
                "bids": [[l[0], l[1]] for l in self.bids[:LEVELS]],
                "asks": [[l[0], l[1]] for l in self.asks[:LEVELS]],
                "tick": self.tick,
                "cvd": round(self.cvd, 6),
                "buy_vol": round(self.buy_vol, 6),
                "sell_vol": round(self.sell_vol, 6),
                "n_trades": self.n_trades,
                "tape": list(self.tape)[:tape],
                "footprint": rows,
                "footprint_levels": len(self.footprint),
                "started_at": datetime.fromtimestamp(
                    self.started_at, timezone.utc).isoformat(),
                "uptime_s": round(_now() - self.started_at, 1),
                "stale_s": (None if self.last_msg_at is None
                            else round(_now() - self.last_msg_at, 2)),
            }


# ─── 거래소 연결 (전용 스레드의 asyncio 루프) ────────────────────

_streams: dict[str, Stream] = {}
_reg_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_tasks: dict[str, Any] = {}


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """거래소 연결을 돌릴 데몬 스레드와 루프. 처음 필요할 때 만든다."""
    global _loop, _thread
    if _loop is not None and _loop.is_running():
        return _loop
    ready = threading.Event()

    def run() -> None:
        global _loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        ready.set()
        loop.run_forever()

    _thread = threading.Thread(target=run, name="orderbook", daemon=True)
    _thread.start()
    ready.wait(5.0)
    if _loop is None:            # pragma: no cover
        raise RuntimeError("호가창 중계 루프를 시작하지 못했습니다")
    return _loop


async def _reader(stream: Stream) -> None:
    """
    한 종목의 거래소 연결. 끊기면 지수 백오프로 다시 붙는다.

    **끊긴 것을 조용히 넘기면 안 된다.** 화면은 마지막 호가를 그대로 들고
    있어서 살아 있는 것처럼 보인다. connected 와 stale_s 를 내보내 화면이
    그 사실을 말할 수 있게 한다.
    """
    import websockets

    sym = stream.symbol
    url = f"{WS_BASE}{sym}@depth{LEVELS}@100ms/{sym}@trade"
    delay = 1.0
    while not stream._stop:
        try:
            async with websockets.connect(url, open_timeout=20,
                                          ping_interval=20) as ws:
                stream.connected = True
                stream.error = ""
                delay = 1.0
                async for raw in ws:
                    if stream._stop:
                        break
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    d = msg.get("data")
                    if not isinstance(d, dict):
                        continue
                    # **분기는 반드시 e 로 한다.** 필드 존재로 판단하면 체결의
                    # a(집계 체결 ID)를 호가 배열로 오인한다.
                    kind = d.get("e")
                    if kind == "depthUpdate":
                        stream.apply_depth(d)
                    elif kind == "trade":
                        stream.apply_trade(d)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stream.error = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            stream.connected = False

        if stream._stop:
            break
        stream.reconnects += 1
        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX)


def subscribe(symbol: str) -> Stream:
    """
    종목 중계를 보장한다. 이미 돌고 있으면 그 상태를 그대로 돌려준다.

    **동기 함수다** — FastAPI 의 동기 엔드포인트에서도, 엔진 스레드에서도
    그냥 부를 수 있어야 한다. 실제 연결은 전용 스레드에서 돈다.
    """
    sym = symbol.strip().lower()
    if sym not in ALLOWED:
        raise UnknownSymbol(
            f"모르는 종목: {symbol} (가능: {', '.join(ALLOWED)})")

    with _reg_lock:
        stream = _streams.get(sym)
        if stream is not None and not stream._stop:
            return stream
        stream = Stream(sym)
        _streams[sym] = stream
        loop = _ensure_loop()
        _tasks[sym] = asyncio.run_coroutine_threadsafe(
            _reader(stream), loop)
        return stream


def get(symbol: str) -> Stream | None:
    """돌고 있는 중계만 돌려준다. 없으면 None (새로 열지 않는다)."""
    return _streams.get(symbol.strip().lower())


def stop(symbol: str) -> bool:
    """중계를 멈춘다. 누적 상태는 버린다."""
    sym = symbol.strip().lower()
    with _reg_lock:
        stream = _streams.pop(sym, None)
        fut = _tasks.pop(sym, None)
    if stream is None:
        return False
    stream._stop = True
    if fut is not None:
        fut.cancel()
    return True


def status() -> dict[str, Any]:
    """
    무엇이 돌고 있나. **잊힌 연결을 볼 수 있어야 한다** — 한 번 열면 명시적으로
    멈출 때까지 계속 도는 구조라, 보이지 않으면 며칠 뒤에도 붙어 있다.
    """
    with _reg_lock:
        streams = list(_streams.values())
    return {
        "running": [{
            "symbol": s.symbol, "connected": s.connected,
            "uptime_s": round(_now() - s.started_at, 1),
            "n_trades": s.n_trades, "reconnects": s.reconnects,
            "footprint_levels": len(s.footprint),
            "stale_s": (None if s.last_msg_at is None
                        else round(_now() - s.last_msg_at, 2)),
            "error": s.error,
        } for s in streams],
        "allowed": list(ALLOWED),
    }
