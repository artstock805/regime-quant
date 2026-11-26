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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from . import sources

# 우리 인터벌 표기 → MT5 상수. 지연 임포트라 여기서는 문자열로만 둔다.
TIMEFRAMES = {
    "1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5", "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30", "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}

_lock = threading.RLock()
_initialised = False

# ─── 브로커 서버 시간 ────────────────────────────────────────────
#
# **MT5 는 서버 시간을 UTC 에폭처럼 돌려준다.** 브로커가 GMT+2/+3 이면
# `datetime.fromtimestamp(t.time, timezone.utc)` 가 실제보다 2~3시간 앞선
# 시각을 만든다. 실측(XM, 2026-09-07): 실제 UTC 06:08 인데 틱이 09:08 로,
# 가장 최근 거래가 **미래 시각**으로 찍혔다. 마감된 봉이 미래에 있는 셈이다.
#
# 그대로 두면 화면의 청산시각이 3시간 틀리고, 일일 손익 경계가 어긋나며,
# 봉 마감에 맞춰 도는 스케줄러가 3시간 빗나간다.
#
# 오프셋을 직접 물어볼 방법은 없어서 **틱 시각과 실제 UTC 의 차이**로 잰다.
# 브로커 오프셋은 항상 30분 배수이므로 격자에 붙이고, 격자에서 5분 이상
# 벗어나면 시장이 닫혀 틱이 낡은 것으로 보고 **이전 값을 유지한다** —
# 주말의 낡은 틱으로 오프셋을 갱신하면 엉뚱한 값이 박힌다.
_GRID = 1800.0          # 오프셋 격자(초). 30분.
_TOLERANCE = 300.0      # 격자에서 이만큼 벗어나면 낡은 틱으로 본다.
_REFRESH = 1800.0       # 이 주기로 다시 잰다 (서머타임 전환 대응).
_MAX_OFFSET = 14 * 3600.0   # 이 밖의 오프셋은 존재하지 않는다 = 낡은 틱이다.
_offset: float | None = None
_offset_at = 0.0
# 바뀐 오프셋 후보 [값, 관측 횟수]. 두 번 연속 같아야 채택한다.
_pending: list = []


def server_offset(*, symbol: str = "EURUSD") -> float:
    """
    브로커 서버 시간 - 실제 UTC (초). 모르면 0.0.

    한 번 알아낸 값은 캐시하고 30분마다 다시 잰다. 시장이 닫혀 있으면
    갱신하지 않고 이전 값을 쓴다.
    """
    global _offset, _offset_at
    now = time.time()
    if _offset is not None and (now - _offset_at) < _REFRESH:
        return _offset

    try:
        with _lock:
            initialise()
            mt5 = _mt5()
            mt5.symbol_select(symbol, True)
            t = mt5.symbol_info_tick(symbol)
        if t is None:
            return _offset or 0.0

        raw = float(t.time) - time.time()
        snapped = round(raw / _GRID) * _GRID
        # 격자에서 많이 벗어나면 틱이 낡았다 — 갱신하지 않는다.
        if abs(raw - snapped) > _TOLERANCE:
            return _offset or 0.0
        # 세상의 어떤 브로커도 UTC±14시간 밖에 있지 않다. 벗어나면 낡은 틱이다.
        if abs(snapped) > _MAX_OFFSET:
            return _offset or 0.0

        _offset_at = now
        if _offset is None:                # 처음이면 그대로 받는다
            _offset = snapped
            _pending[:] = [snapped, 1]
            return _offset
        if snapped == _offset:             # 그대로면 확인만 하고 끝
            _pending[:] = [snapped, 1]
            return _offset

        # **바뀐 값은 두 번 연속 같아야 받는다.**
        #
        # 낡은 틱과 서머타임 전환을 구분하려면 이 방법밖에 없다. 처음엔
        # "아는 오프셋으로 틱 나이를 재서 낡으면 무시"하게 했는데, 그러면
        # 정상적인 전환(3시간→2시간)도 "1시간 낡은 틱"으로 보여 영원히
        # 갱신되지 않았다(테스트가 잡았다).
        #
        # 시장이 닫혀 있으면 틱은 멈춰 있으므로 잰 오프셋이 잴 때마다
        # 갱신 주기만큼 줄어든다 — 두 번 연속 같은 값이 나오지 않는다.
        # 전환은 그 반대로 매번 같은 값이 나온다. 그래서 채택까지
        # 갱신 주기 하나(30분)가 걸리고, 그 정도는 감수할 만하다.
        if _pending and _pending[0] == snapped:
            _offset = snapped
            _pending[:] = [snapped, 2]
        else:
            _pending[:] = [snapped, 1]
    except Exception:
        return _offset or 0.0
    return _offset


def offset_known() -> bool:
    """오프셋을 실제로 알아냈나. 모른 채 0 을 쓰고 있는지 화면이 알아야 한다."""
    return _offset is not None


def to_utc(server_epoch: float) -> datetime:
    """MT5 가 준 시각을 **실제 UTC** 로 바꾼다."""
    return datetime.fromtimestamp(float(server_epoch) - server_offset(),
                                  timezone.utc)

# 우리 주문임을 표시하는 번호와 태그 접두사. **거래 내역을 읽는 쪽과 주문을
# 내는 쪽이 같은 값을 봐야 한다.** 양쪽에 따로 적어 두면 한쪽만 바뀌었을 때
# 과거 거래가 조용히 "남의 것"으로 분류되고, 성적이 틀리게 나온다.
#
# 정의는 sources.py 에 있다 — 그 파일이 이 계좌에 주문을 내는 **모든 앱**의
# 번호를 한자리에 모아 둔 곳이다. 여기서는 이름만 다시 내보낸다.
MAGIC = sources.MAGIC
PREFIX = sources.PREFIX


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
    # 1틱의 크기와, 1랏이 1틱 움직일 때의 손익(**계좌 통화**). 호가 통화를 계좌
    # 통화로 바꾸는 데 쓴다. 0 이면 모른다는 뜻이다.
    tick_size: float = 0.0
    tick_value: float = 0.0

    def notional(self, lots: float, price: float) -> float:
        """
        이 수량의 명목가를 **계좌 통화**로.

        `랏 × 계약 크기 × 가격` 은 호가 통화 단위다. 달러 계좌에서 그대로 쓰면
        USD/JPY 0.1랏(1만 달러어치)이 159만으로 나온다 — 실제로 그 값이 명목가
        상한에 걸려 엔 통화쌍이 한 번도 진입하지 못했다. CAD·NZD·CHF 호가도 정도만
        다를 뿐 같은 식으로 틀린다.

        틱 가치가 그 환산을 담고 있다: 가격 전체는 (가격 / 틱 크기)틱이고, 1틱의
        계좌 통화 가치가 tick_value 다. 브로커가 쓰는 환율 그대로다.
        틱 정보가 없으면 호가 통화가 곧 계좌 통화라고 본다(예전 계산).
        """
        if self.tick_size > 0 and self.tick_value > 0:
            return lots * (price / self.tick_size) * self.tick_value
        return lots * self.contract_size * price

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
            tick_size=float(getattr(s, "trade_tick_size", 0.0) or 0.0),
            tick_value=float(getattr(s, "trade_tick_value", 0.0) or 0.0),
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
    # **서버 시간을 실제 UTC 로 옮긴다.** 안 옮기면 마감된 봉이 미래 시각으로
    # 찍히고(실측 +3시간), 봉 마감에 맞춰 도는 스케줄러가 그만큼 빗나간다.
    df["open_time"] = pd.to_datetime(
        df["time"] - int(server_offset()), unit="s", utc=True)
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
                "at": to_utc(t.time).isoformat()}


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
            "opened_at": to_utc(p.time).isoformat(),
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


# ─── 거래 내역 ───────────────────────────────────────────────────
#
# **이것이 실현손익의 진실이다.** 우리 장부(Journal)는 앱이 켜져 있던 동안만
# 안다. 손절이 새벽에 브로커에서 터졌거나 사용자가 MT5 창에서 직접 닫았으면
# 장부에는 아무것도 없고 여기에만 남는다. 화면은 이쪽을 보여 준다.

# MT5 의 deal.entry — 이 거래가 포지션을 열었나 닫았나.
_ENTRY_IN, _ENTRY_OUT, _ENTRY_INOUT, _ENTRY_OUT_BY = 0, 1, 2, 3

# deal.type — 2 부터는 매매가 아니라 입출금·크레딧 같은 잔고 조작이다.
_TYPE_BUY, _TYPE_SELL, _TYPE_BALANCE = 0, 1, 2


def deals(days: int = 90) -> list[dict]:
    """
    기간 내 거래(deal) 원본을 정규화해서 돌려준다.

    **시간대 주의.** MT5 는 서버 시간으로 조회 구간을 해석하는데 브로커
    서버가 UTC 가 아니다(XM 은 GMT+2/+3). 그래서 앞뒤로 하루씩 넉넉히
    요청하고 **받은 뒤에 실제 시각으로 잘라낸다.** 안 그러면 경계에 걸친
    거래가 조용히 빠진다.
    """
    now = datetime.now(timezone.utc)
    frm = now - timedelta(days=days)
    with _lock:
        initialise()
        rows = _mt5().history_deals_get(frm - timedelta(days=1),
                                        now + timedelta(days=1))
    out: list[dict] = []
    for d in (rows or []):
        at = to_utc(d.time)
        if at < frm:
            continue
        out.append({
            "deal": int(d.ticket),
            "order": int(getattr(d, "order", 0) or 0),
            "position_id": int(getattr(d, "position_id", 0) or 0),
            "at": at.isoformat(),
            "epoch": int(d.time),
            "symbol": str(getattr(d, "symbol", "") or ""),
            "type": int(d.type),
            "entry": int(getattr(d, "entry", _ENTRY_IN)),
            "volume": float(getattr(d, "volume", 0.0) or 0.0),
            "price": float(getattr(d, "price", 0.0) or 0.0),
            "profit": float(getattr(d, "profit", 0.0) or 0.0),
            "commission": float(getattr(d, "commission", 0.0) or 0.0),
            "swap": float(getattr(d, "swap", 0.0) or 0.0),
            "fee": float(getattr(d, "fee", 0.0) or 0.0),
            "magic": int(getattr(d, "magic", 0) or 0),
            "comment": str(getattr(d, "comment", "") or ""),
        })
    out.sort(key=lambda r: r["epoch"])
    return out


def _weighted(rows: list[dict], key: str) -> float:
    vol = sum(r["volume"] for r in rows)
    if vol <= 0:
        return rows[0][key] if rows else 0.0
    return sum(r[key] * r["volume"] for r in rows) / vol


def closed_trades(days: int = 90, raw: list[dict] | None = None) -> list[dict]:
    """
    거래들을 **왕복 한 건**으로 묶는다. 진입 체결과 청산 체결은 따로 오므로
    position_id 로 모아야 "언제 들어가서 언제 나왔고 얼마 남았나"가 나온다.

    아직 열려 있는 포지션(청산 체결이 없는 것)은 제외한다 — 그건 positions()
    가 보여 준다. 손익은 수수료와 스왑까지 **다 뺀 순액**이다. 수수료를 빼지
    않으면 화면 숫자가 계좌 잔고와 어긋나고, 그건 신뢰를 깨뜨린다.
    """
    src = raw if raw is not None else deals(days)
    groups: dict[int, list[dict]] = {}
    for d in src:
        if d["type"] > _TYPE_SELL:          # 입출금은 매매가 아니다
            continue
        groups.setdefault(d["position_id"], []).append(d)

    trades: list[dict] = []
    for pid, g in groups.items():
        ins = [d for d in g if d["entry"] in (_ENTRY_IN, _ENTRY_INOUT)]
        outs = [d for d in g if d["entry"] in (_ENTRY_OUT, _ENTRY_OUT_BY,
                                               _ENTRY_INOUT)]
        if not ins or not outs:
            continue                        # 아직 열려 있다
        entry_at = min(d["epoch"] for d in ins)
        exit_at = max(d["epoch"] for d in outs)
        net = sum(d["profit"] + d["commission"] + d["swap"] + d["fee"] for d in g)
        # **연 쪽이 주인이다.** 한 포지션을 다른 앱이 닫는 일이 실제로
        # 있다(실측 2건). 체결 순서로 아무거나 고르면 같은 계좌를 두 번
        # 재면서 답이 달라진다.
        src_of = sources.of_deals(g, ins)
        out_of = sources.of_deals(outs)
        trades.append({
            "position_id": pid,
            "symbol": ins[0]["symbol"],
            # 진입 체결이 매수였으면 롱이다. 청산 체결은 반대 방향이므로 보지 않는다.
            "side": "LONG" if ins[0]["type"] == _TYPE_BUY else "SHORT",
            "volume": round(sum(d["volume"] for d in ins), 4),
            "entry_price": _weighted(ins, "price"),
            "exit_price": _weighted(outs, "price"),
            "opened_at": to_utc(entry_at).isoformat(),
            "closed_at": to_utc(exit_at).isoformat(),
            "held_minutes": round((exit_at - entry_at) / 60, 1),
            "profit": round(net, 2),
            "gross": round(sum(d["profit"] for d in g), 2),
            "commission": round(sum(d["commission"] + d["fee"] for d in g), 2),
            "swap": round(sum(d["swap"] for d in g), 2),
            # **누가 낸 거래인가.** 이 계좌에는 여러 앱이 동시에 주문을
            # 낸다. '우리 것/아닌 것' 두 칸으로만 가르면 남의 봇 거래가
            # 전부 "수동"으로 묶여서 성적을 잘못 읽는다 — 실제로 그렇게
            # 읽고 틀린 진단을 냈다. 어느 앱인지까지 싣는다(sources.py).
            "source": src_of.key,
            "source_label": src_of.label,
            "magic": next((d["magic"] for d in ins if d["magic"]), 0),
            # 연 앱과 닫은 앱이 다르면 밝힌다. 앱들이 서로의 포지션을
            # 건드리고 있다는 뜻이고, 그건 조용히 지나갈 일이 아니다.
            "closed_by": (out_of.label if out_of.key not in
                          (src_of.key, sources.HAND.key) else None),
            "ours": src_of.key == "engine",
            "comment": next((d["comment"] for d in ins if d["comment"]), ""),
        })
    trades.sort(key=lambda t: t["closed_at"], reverse=True)
    return trades


def cash_flows(days: int = 3650, raw: list[dict] | None = None) -> dict:
    """입출금 합계. 수익률을 재려면 '얼마를 넣었는지'를 알아야 한다."""
    src = raw if raw is not None else deals(days)
    dep = sum(d["profit"] for d in src
              if d["type"] == _TYPE_BALANCE and d["profit"] > 0)
    wit = sum(d["profit"] for d in src
              if d["type"] == _TYPE_BALANCE and d["profit"] < 0)
    return {"deposit": round(dep, 2), "withdrawal": round(wit, 2),
            "net_funding": round(dep + wit, 2)}
