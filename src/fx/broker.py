"""
MT5 주문 계층 — 진입과 청산.

이 파일이 계좌에 실제로 손을 대는 유일한 곳이다. 그래서 규칙이 몇 개 있다.

**1. 기본은 데모다.** 실계좌에는 설정을 명시적으로 바꾸지 않으면 주문이
나가지 않는다. 사용자의 MT5 터미널에는 실계좌가 붙어 있을 수 있고, 검증되지
않은 전략이 거기로 흘러 들어가는 것이 이 프로젝트에서 가장 큰 사고다.

**2. 손절은 진입과 같은 요청에 붙는다.** 나중에 따로 걸지 않는다. 프로그램이
죽거나 네트워크가 끊겨도 브로커 쪽에 손절이 남아 있어야 포지션이 맨몸으로
남지 않는다. 손절 없이 들어가는 길은 하나뿐이다 — 부르는 쪽이 손절에 None 을
명시했을 때(역행 대응의 최종 손절을 꺼 둔 설정).

**3. 계산한 랏이 최소 랏보다 작으면 주문하지 않는다.** 올려서 넣으면 의도한
1R 보다 큰 위험을 지게 된다 — 조용히 리스크가 커지는 쪽이 훨씬 나쁘다.

**4. 같은 봉에 같은 종목은 한 번만.** MT5 에는 클라이언트 주문 ID 가 없어서
magic 번호와 comment 태그로 대신한다. 루프가 두 번 돌아도 두 번 들어가지 않는다.

**5. MT5 가 진실이다.** 내부 기록과 다르면 이쪽을 믿는다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .mt5 import (
    MAGIC, MT5Error, PREFIX, _lock, _mt5, account, initialise, positions,
    symbol_info,
)

# MAGIC(우리 주문 표시 번호)과 PREFIX(태그 접두사)는 mt5.py 에 있다.
# 거래 내역을 읽는 쪽도 같은 값으로 "우리 거래"를 골라내야 하기 때문이다.
# 여기서 이름을 다시 내보내므로 broker.MAGIC 으로 쓰던 곳은 그대로 동작한다.
__all__ = ["MAGIC", "PREFIX"]

# 브로커가 허용하는 체결 방식은 종목마다 다르다. 안 맞으면 10030 으로 거부된다.
_FILL_FOK, _FILL_IOC = 1, 2


class OrderRejected(MT5Error):
    """브로커가 거부했다. retcode 와 사유를 담는다."""

    def __init__(self, message: str, retcode: int | None = None) -> None:
        super().__init__(message)
        self.retcode = retcode


class LiveAccountBlocked(MT5Error):
    """실계좌인데 허용 설정이 없다. 기본값이 사람을 지켜야 한다."""


# 단기 투표 군이 연 주문의 표식. 태그에서 PREFIX 바로 뒤에 온다.
# 심볼은 대문자로 쓰므로 소문자 s 와 점은 심볼 이름과 겹치지 않는다.
SHORT_CODE = "s."


def tag(symbol: str, bar_time: datetime | str, group_code: str = "") -> str:
    """
    같은 봉에 같은 종목으로 두 번 들어가지 않게 하는 표식.

    MT5 에는 거래소 쪽 중복 방지 ID 가 없으므로 comment 에 실어 보내고,
    진입 전에 열린 포지션의 comment 를 확인한다. 루프가 두 번 돌아도
    한 번만 나간다. comment 는 길이 제한이 빡빡하므로(대개 31자) 짧게 만든다.

    group_code 는 **어느 투표 군이 연 주문인지**를 남긴다. 두 군이 한 계좌에서
    따로 진입·청산하므로, 재시작한 뒤에도 포지션의 주인을 comment 에서 읽을 수
    있어야 한다. 기억에만 두면 재시작 순간 남의 군 포지션을 내 것으로 센다.
    """
    if isinstance(bar_time, str):
        bar_time = datetime.fromisoformat(bar_time)
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=timezone.utc)
    core = symbol.upper()[:6]
    stamp = bar_time.astimezone(timezone.utc).strftime("%y%m%d%H%M")
    return f"{PREFIX}{group_code}{core}{stamp}"


def group_of(pos: dict) -> str:
    """
    이 포지션을 연 투표 군 ("main" 또는 "short").

    표식이 없으면 기본 군이다 — 군을 나누기 전에 연 포지션과, 브로커가 comment 를
    바꿔 버린 포지션이 여기 해당한다. 주인 없는 포지션으로 두면 아무도 청산
    판단을 하지 않는다.
    """
    comment = str(pos.get("comment", "")).strip()
    return "short" if comment.startswith(PREFIX + SHORT_CODE) else "main"


@dataclass
class Fill:
    ticket: int
    symbol: str
    side: str
    volume: float
    price: float
    stop_loss: float | None        # None = 손절 없이 열었다
    take_profit: float | None
    tag: str
    retcode: int
    comment: str

    def as_dict(self) -> dict:
        return asdict(self)


# ─── 실계좌 관문 ─────────────────────────────────────────────────

def assert_tradable(allow_live: bool = False) -> dict:
    """
    주문을 내도 되는 계좌인지 본다. **모든 주문이 여기를 먼저 지난다.**

    실계좌에서 돌리려면 설정에서 명시적으로 켜야 한다. 기본값이 '허용'이면
    언젠가 실수로 실계좌에 붙은 채 켜게 되고, 그때는 늦다.
    """
    a = account()
    if not a.trade_allowed:
        raise MT5Error(
            "MT5 에서 자동매매가 꺼져 있습니다 — "
            "도구 > 옵션 > Expert Advisors > 자동매매 허용을 켜세요")
    if not a.is_demo and not allow_live:
        raise LiveAccountBlocked(
            f"실계좌입니다 (로그인 {a.login}, {a.server}). "
            "이 앱은 기본적으로 데모에서만 주문합니다.\n"
            "  · 실계좌에서 돌리려면 config.yaml 의 fx.allow_live 를 true 로 바꾸세요\n"
            "  · 그 전에 데모에서 충분히 돌려 보세요 — "
            "이 전략은 FX 에서 아직 검증되지 않았습니다")
    return {"login": a.login, "server": a.server, "is_demo": a.is_demo,
            "balance": a.balance, "equity": a.equity}


# ─── 랏 계산 ─────────────────────────────────────────────────────

def _round_step(volume: float, step: float) -> float:
    """랏 단위로 **내림**한다. 올리면 의도한 1R 보다 큰 위험을 지게 된다."""
    if step <= 0:
        return volume
    return math.floor(volume / step + 1e-9) * step


def lots_for_risk(symbol: str, side: str, entry: float, stop: float,
                  risk_money: float) -> tuple[float, str]:
    """
    손절에 맞았을 때 잃는 돈이 risk_money 가 되도록 랏을 정한다.

    통화쌍마다 1핍의 돈 가치가 다르고(EUR/USD 와 USD/JPY 는 계좌통화 환산이
    다르다) 계좌통화도 다르다. 직접 계산하면 크로스 환율에서 틀리므로
    MT5 의 order_calc_profit 에 맡긴다 — 브로커의 실제 환산을 그대로 쓴다.

    돌려주는 두 번째 값은 **거부 사유**다. 비어 있으면 통과.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"방향은 LONG/SHORT 여야 합니다: {side}")
    if risk_money <= 0:
        return 0.0, "위험 금액이 0 입니다"

    si = symbol_info(symbol)
    if abs(entry - stop) <= 0:
        return 0.0, "손절 거리가 0 입니다"

    with _lock:
        initialise()
        mt5 = _mt5()
        kind = mt5.ORDER_TYPE_BUY if side == "LONG" else mt5.ORDER_TYPE_SELL
        # 1랏으로 손절까지 갔을 때의 손익(음수)을 브로커에게 물어본다
        loss_one_lot = mt5.order_calc_profit(kind, symbol, 1.0, entry, stop)

    if not loss_one_lot:
        return 0.0, f"{symbol} 손실 환산에 실패했습니다 — 랏을 계산할 수 없습니다"

    raw = risk_money / abs(loss_one_lot)
    lots = _round_step(raw, si.volume_step)

    if lots < si.volume_min:
        return 0.0, (
            f"계산된 랏 {raw:.4f} 이 최소 {si.volume_min} 보다 작습니다 — "
            f"올려서 넣으면 의도한 위험({risk_money:.2f})보다 커지므로 건너뜁니다")
    if lots > si.volume_max:
        lots = _round_step(si.volume_max, si.volume_step)
    return round(lots, 8), ""


# ─── 중복 방지 ───────────────────────────────────────────────────

def ours(pos: dict) -> bool:
    """
    우리가 연 포지션인가.

    사용자의 계좌에는 사람이 손으로 연 포지션(금·지수 등)이 섞여 있다.
    그것을 우리 것으로 착각해 청산하면 그건 사고다.
    """
    if pos.get("magic") == MAGIC:
        return True
    return str(pos.get("comment", "")).strip().startswith(PREFIX)


def already_open(tag_value: str) -> dict | None:
    """이 태그로 이미 들어가 있는가. 루프가 두 번 돌아도 두 번 들어가지 않는다."""
    if not tag_value:
        return None
    for p in positions().values():
        if str(p.get("comment", "")).strip() == tag_value:
            return p
    return None


# ─── 체결 방식 ───────────────────────────────────────────────────

def _filling(symbol: str) -> int:
    """
    브로커가 이 종목에 허용하는 체결 방식. 틀리면 10030 (Unsupported filling
    mode) 으로 거부되는데, 문구만 보면 원인을 짐작하기 어렵다.
    """
    with _lock:
        initialise()
        mt5 = _mt5()
        s = mt5.symbol_info(symbol)
        modes = getattr(s, "filling_mode", 0) if s else 0
        if modes & _FILL_FOK:
            return mt5.ORDER_FILLING_FOK
        if modes & _FILL_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN


# ─── 진입 ────────────────────────────────────────────────────────

def open_position(symbol: str, side: str, lots: float, stop_loss: float | None,
                  take_profit: float | None = None, *,
                  tag_value: str = "", deviation: int = 20,
                  allow_live: bool = False) -> Fill:
    """
    시장가 진입. **손절이 같은 요청에 붙는다.**

    나중에 따로 거는 방식은 그 사이에 프로그램이 죽으면 포지션이 맨몸으로
    남는다. 브로커 쪽에 손절이 있으면 우리가 꺼져 있어도 지켜진다.

    stop_loss 가 None 이면 **손절 없이** 들어간다. 실수로 빠뜨릴 수 없게 인자는
    여전히 필수다 — 손절 없는 진입은 부르는 쪽이 None 을 명시해야만 나간다
    (역행 대응에서 최종 손절을 꺼 둔 경우, src/fx/defense.py).
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"방향은 LONG/SHORT 여야 합니다: {side}")
    assert_tradable(allow_live)

    if already_open(tag_value):
        raise OrderRejected(f"{tag_value} 로 이미 포지션이 열려 있습니다 — 중복 진입 방지")

    si = symbol_info(symbol)
    if lots < si.volume_min:
        raise OrderRejected(f"랏 {lots} 이 최소 {si.volume_min} 보다 작습니다")

    with _lock:
        initialise()
        mt5 = _mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5Error(f"{symbol} 호가를 읽지 못했습니다")

        if side == "LONG":
            kind, price = mt5.ORDER_TYPE_BUY, t.ask
            if stop_loss is not None and stop_loss >= price:
                raise OrderRejected(
                    f"롱인데 손절({stop_loss})이 진입가({price}) 위에 있습니다")
        else:
            kind, price = mt5.ORDER_TYPE_SELL, t.bid
            if stop_loss is not None and stop_loss <= price:
                raise OrderRejected(
                    f"숏인데 손절({stop_loss})이 진입가({price}) 아래에 있습니다")

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": kind,
            "price": float(price),
            "deviation": int(deviation),
            "magic": MAGIC,
            "comment": tag_value[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling(symbol),
        }
        if stop_loss is not None:
            req["sl"] = round(float(stop_loss), si.digits)
        if take_profit:
            req["tp"] = round(float(take_profit), si.digits)

        res = mt5.order_send(req)
        done = mt5.TRADE_RETCODE_DONE
        last = mt5.last_error()

    if res is None:
        code, msg = last
        raise OrderRejected(f"{symbol} 주문을 보내지 못했습니다 [{code}] {msg}", code)

    if res.retcode != done:
        raise OrderRejected(
            f"{symbol} 주문 거부 [{res.retcode}] {res.comment}", res.retcode)

    # 체결 후의 포지션 티켓을 찾는다. MT5 는 주문·거래·포지션을 따로 센다.
    ticket = int(getattr(res, "order", 0) or 0)
    opened = already_open(tag_value)
    if opened:
        ticket = int(opened["ticket"])

    return Fill(ticket=ticket, symbol=symbol, side=side, volume=float(res.volume),
                price=float(res.price),
                stop_loss=None if stop_loss is None else float(stop_loss),
                take_profit=take_profit, tag=tag_value,
                retcode=int(res.retcode), comment=str(res.comment))


# ─── 청산 ────────────────────────────────────────────────────────

def close_position(ticket: int, *, deviation: int = 20,
                   allow_live: bool = False, reason: str = "") -> dict:
    """
    포지션 하나를 시장가로 닫는다.

    티켓 단위로 다룬다 — 같은 종목에 여러 포지션이 있을 수 있고, 종목 단위로
    닫으면 우리가 열지 않은 것까지 닫힌다.
    """
    assert_tradable(allow_live)
    pos = positions().get(str(ticket))
    if pos is None:
        # 손절이 이미 체결됐거나 사람이 닫았다. 실패가 아니다.
        return {"ticket": ticket, "closed": False, "reason": "이미 닫혀 있습니다"}

    symbol = pos["symbol"]
    with _lock:
        initialise()
        mt5 = _mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5Error(f"{symbol} 호가를 읽지 못했습니다")
        if pos["side"] == "LONG":
            kind, price = mt5.ORDER_TYPE_SELL, t.bid
        else:
            kind, price = mt5.ORDER_TYPE_BUY, t.ask

        res = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos["volume"]),
            "type": kind,
            "position": int(ticket),
            "price": float(price),
            "deviation": int(deviation),
            "magic": MAGIC,
            "comment": (reason or "rqclose")[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _filling(symbol),
        })
        done = mt5.TRADE_RETCODE_DONE
        last = mt5.last_error()

    if res is None:
        code, msg = last
        raise OrderRejected(f"{symbol} 청산을 보내지 못했습니다 [{code}] {msg}", code)
    if res.retcode != done:
        raise OrderRejected(
            f"{symbol} 청산 거부 [{res.retcode}] {res.comment}", res.retcode)

    return {"ticket": ticket, "closed": True, "symbol": symbol,
            "side": pos["side"], "volume": float(res.volume),
            "price": float(res.price), "profit": pos.get("profit"),
            "reason": reason, "retcode": int(res.retcode)}


def close_all(*, allow_live: bool = False, reason: str = "flatten") -> list[dict]:
    """
    **우리가 연 포지션 전부를 닫는다.** 사람이 손으로 연 것은 건드리지 않는다 —
    이 앱이 열지 않은 금·지수 포지션까지 닫아 버리면 그건 사고다.
    """
    out = []
    for ticket, pos in positions().items():
        if not ours(pos):
            continue
        try:
            out.append(close_position(int(ticket), allow_live=allow_live,
                                      reason=reason))
        except MT5Error as exc:
            out.append({"ticket": int(ticket), "closed": False,
                        "reason": str(exc)})
    return out


# ─── 손절 옮기기 ─────────────────────────────────────────────────

def move_stop(ticket: int, stop_loss: float | None,
              take_profit: float | None = None) -> dict:
    """
    손절·목표가를 옮긴다 (본전 이동, 묶음 목표가 당기기). 포지션은 그대로 둔다.

    MT5 는 이 요청에서 손절과 목표가를 **함께** 다시 쓴다. 손절 없는 포지션의
    목표가만 옮길 때는 stop_loss 에 None 을 준다 — 손절 0(없음)으로 보낸다.
    """
    pos = positions().get(str(ticket))
    if pos is None:
        raise MT5Error(f"티켓 {ticket} 포지션이 없습니다")
    si = symbol_info(pos["symbol"])
    with _lock:
        initialise()
        mt5 = _mt5()
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos["symbol"],
            "position": int(ticket),
            "sl": 0.0 if stop_loss is None else round(float(stop_loss), si.digits),
            "magic": MAGIC,
        }
        if take_profit is not None:
            req["tp"] = round(float(take_profit), si.digits)
        res = mt5.order_send(req)
        done = mt5.TRADE_RETCODE_DONE

    if res is None or res.retcode != done:
        code = getattr(res, "retcode", None)
        raise OrderRejected(
            f"손절 이동 거부 [{code}] {getattr(res, 'comment', '')}", code)
    return {"ticket": ticket, "stop_loss": stop_loss, "take_profit": take_profit}


# ─── 대조 ────────────────────────────────────────────────────────

def reconcile(known: dict[str, Any]) -> dict[str, Any]:
    """
    **MT5 가 진실이다.** 내부 기록과 어긋나면 이쪽에 맞춘다.

    어긋나는 경우는 늘 있다. 손절이 브로커 쪽에서 체결됐거나, 사람이 터미널에서
    직접 닫았거나, 우리가 재시작했거나. 조용히 넘기면 이미 닫힌 포지션을
    들고 있다고 착각한 채 동시 보유 한도를 채운다.
    """
    live = positions()
    mine = {k: v for k, v in live.items() if ours(v)}
    return {
        "open": mine,
        # 손절 체결·수동 청산·재시작
        "closed_elsewhere": [k for k in known if k not in live],
        # 우리 태그인데 기록에 없다 (재시작 후 흔하다)
        "unknown_to_us": [k for k in mine if k not in known],
        # 사람이 연 것. 건드리지 않는다
        "not_ours": sorted({v["symbol"] for v in live.values() if not ours(v)}),
        "at": datetime.now(timezone.utc).isoformat(),
    }
