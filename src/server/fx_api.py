"""
FX 상황판 API — 차트는 트레이딩뷰, 데이터·주문은 MT5.

화면의 중심은 **트레이딩뷰 차트**다. 익숙한 인터페이스로 지금 시장을 보고,
그 위에 우리 신호와 실제 포지션을 얹는다. 주문은 MT5 로 나간다.

    GET  /fx                     FX 상황판 (본 화면)
    GET  /api/fx/status          계좌·포지션·연결 상태
    GET  /api/fx/pairs           이 브로커에서 쓸 수 있는 통화쌍 + 트레이딩뷰 심볼
    GET  /api/fx/quotes          관심 통화쌍의 현재 호가·스프레드·비용
    GET  /api/fx/focus           차트가 지금 보고 있는 종목
    POST /api/fx/focus           차트 종목 바꾸기

    GET  /api/fx/engine          자동매매 엔진 상태 + **검증 판정**
    POST /api/fx/engine/start    엔진 켜기 (꺼진 채로 시작한다)
    POST /api/fx/engine/stop     새 진입 중단 (청산은 계속된다)
    POST /api/fx/engine/kill     킬 스위치 — 조건 없이 멈춘다
    POST /api/fx/engine/release  킬 스위치 해제
    POST /api/fx/engine/flatten  우리가 연 포지션 전량 청산
    POST /api/fx/engine/once     한 바퀴 수동 실행
    GET  /api/fx/journal         매매 장부

**두 시세는 다르다.** 트레이딩뷰(FX_IDC)는 은행 간 집계고 MT5 는 우리가 체결하는
브로커 호가다. 차트의 마커와 실제 체결가가 몇 핍 어긋나는 것은 버그가 아니다.
화면이 그 사실을 숨기지 않는다.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import load_config
from ..fx import (
    DEFAULT_PROVIDER, MT5Error, available_symbols, bars, catalogue, resolve,
    status as mt5_status, symbol_info, tick,
)
from ..fx.mt5 import cash_flows, closed_trades, deals as fx_deals
from ..fx.symbols import parse

STATIC = Path(__file__).parent / "static"
router = APIRouter()

# 차트가 보고 있는 종목. 신호나 포지션이 생기면 자동으로 따라가고,
# 사람이 고르면 그쪽이 우선한다 (수동 선택을 자동 전환이 덮으면 짜증난다).
_focus: dict[str, Any] = {"symbol": None, "manual": False}
_focus_lock = threading.Lock()

# 심볼 목록은 자주 안 바뀌므로 한 번만 읽는다. MT5 호출은 느리다.
_pairs_cache: list | None = None


def _pairs(provider: str = DEFAULT_PROVIDER):
    global _pairs_cache
    if _pairs_cache is None:
        _pairs_cache = catalogue(available_symbols(only_forex=True),
                                 provider=provider, majors_only=True)
    return _pairs_cache


def _watchlist(cfg) -> list[str]:
    """설정에 적힌 관심 통화쌍. 없으면 주요 통화쌍 전부."""
    want = cfg.get("fx.symbols") or []
    return [str(s).upper() for s in want]


# 화면 코드가 바뀌었는데 브라우저가 옛 파일을 들고 있으면, 고친 것이 반영되지
# 않은 채 "왜 그대로지"를 디버깅하게 된다. 상황판은 매번 새로 받는다.
NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


@router.get("/fx")
def fx_page() -> FileResponse:
    return FileResponse(STATIC / "fx.html", headers=NO_CACHE)


@router.get("/api/fx/status")
def fx_status() -> dict[str, Any]:
    st = mt5_status()
    st["provider"] = load_config().get("fx.tv_provider", DEFAULT_PROVIDER)
    with _focus_lock:
        st["focus"] = dict(_focus)
    if not st["connected"]:
        return st

    # 포지션에 트레이딩뷰 심볼을 붙여 화면이 차트를 바로 띄울 수 있게 한다
    pairs = {p.mt5: p for p in _pairs(st["provider"])}
    for pos in st["positions"]:
        p = pairs.get(pos["symbol"])
        if p:
            pos["tradingview"] = p.tradingview
            pos["display"] = p.display
    return st


@router.get("/api/fx/pairs")
def fx_pairs() -> dict[str, Any]:
    cfg = load_config()
    provider = cfg.get("fx.tv_provider", DEFAULT_PROVIDER)
    try:
        pairs = _pairs(provider)
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc
    watch = set(_watchlist(cfg))
    return {
        "provider": provider,
        "pairs": [{**p.as_dict(), "watched": (not watch) or p.canonical in watch}
                  for p in pairs],
    }


@router.get("/api/fx/quotes")
def fx_quotes() -> dict[str, Any]:
    """
    관심 통화쌍의 현재 호가와 **비용/1R**.

    비용/1R 을 화면에 계속 띄우는 이유가 있다. 이 프로젝트는 암호화폐에서
    "우위는 실재하는데 비용이 그 59%를 먹어서 순우위가 0" 이라는 결론에 도달했다.
    FX 는 스프레드가 더 넓어 그 비율이 더 크다. 종목을 고를 때 이 숫자를
    안 보면 같은 함정에 다시 빠진다.
    """
    cfg = load_config()
    provider = cfg.get("fx.tv_provider", DEFAULT_PROVIDER)
    interval = cfg.get("fx.interval", "1h")
    stop_mult = float(cfg.get("risk.atr_stop_mult", 4.0))
    watch = set(_watchlist(cfg))

    rows = []
    try:
        pairs = _pairs(provider)
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc

    for p in pairs:
        if watch and p.canonical not in watch:
            continue
        try:
            si = symbol_info(p.mt5)
            t = tick(p.mt5)
            df = bars(p.mt5, interval, 120)
        except MT5Error as exc:
            rows.append({**p.as_dict(), "error": str(exc)})
            continue

        # ATR(14) — 지표 모듈을 그대로 쓰면 무겁다. 여기서는 화면용 근사로 충분하다.
        high, low, close = df["high"], df["low"], df["close"]
        prev = close.shift()
        tr = (high - low).combine((high - prev).abs(), max) \
                         .combine((low - prev).abs(), max)
        atr = float(tr.rolling(14).mean().iloc[-1])
        one_r = stop_mult * atr
        # 왕복 비용 = 스프레드 1회 + 슬리피지 여유 0.5회
        cost = si.spread_price * 1.5
        rows.append({
            **p.as_dict(),
            "bid": t["bid"], "ask": t["ask"],
            "spread_pips": round(si.spread_pips, 1),
            "atr": round(atr, si.digits),
            "one_r": round(one_r, si.digits),
            "cost_over_r_pct": round(cost / one_r * 100, 2) if one_r else None,
            "digits": si.digits,
            "at": t["at"],
        })
    rows.sort(key=lambda r: (r.get("cost_over_r_pct") is None,
                             r.get("cost_over_r_pct", 999)))
    return {"interval": interval, "stop_mult": stop_mult, "quotes": rows}


@router.get("/api/fx/focus")
def get_focus() -> dict[str, Any]:
    with _focus_lock:
        return dict(_focus)


@router.post("/api/fx/focus")
def set_focus(symbol: str, manual: bool = True) -> dict[str, Any]:
    """
    차트 종목을 바꾼다. manual=True 면 자동 전환이 덮지 않는다 —
    사람이 골라 본 것을 다음 신호가 밀어내면 화면을 볼 수가 없다.
    """
    try:
        pairs = _pairs(load_config().get("fx.tv_provider", DEFAULT_PROVIDER))
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc

    target = symbol.upper()
    found = next((p for p in pairs
                  if target in (p.canonical, p.mt5.upper())), None)
    if found is None:
        raise HTTPException(404, f"{symbol} 은 이 브로커에 없습니다")

    with _focus_lock:
        _focus.update(symbol=found.canonical, manual=manual,
                      mt5=found.mt5, tradingview=found.tradingview,
                      display=found.display)
        return dict(_focus)


def follow(symbol: str) -> None:
    """
    엔진이 부른다 — 신호나 진입이 생긴 종목으로 차트를 옮긴다.
    사람이 수동으로 골라 둔 상태면 건드리지 않는다.
    """
    with _focus_lock:
        if _focus.get("manual"):
            return
    try:
        set_focus(symbol, manual=False)
    except HTTPException:
        pass


# ═══════════════════════════════════════════════════════════════
#  자동매매 엔진
#
#  **검증 판정을 모든 응답에 싣는다.** 이 신호는 FX 에서 우위가 확인되지
#  않았고, 화면이 그걸 숨기면 사용자는 검증된 것을 돌리고 있다고 착각한다.
# ═══════════════════════════════════════════════════════════════


def _engine():
    from ..fx.engine import trader
    return trader()


@router.get("/api/fx/engine")
def engine_status() -> dict[str, Any]:
    try:
        return _engine().status()
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/api/fx/engine/start")
def engine_start() -> dict[str, Any]:
    try:
        return _engine().start()
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/api/fx/engine/stop")
def engine_stop() -> dict[str, Any]:
    """새 진입만 막는다. 들고 있는 것은 계속 청산 판단을 받는다 —
    정지 버튼이 포지션을 방치하는 버튼이면 위험을 키운다."""
    return _engine().stop()


@router.post("/api/fx/engine/kill")
def engine_kill(reason: str = "사용자") -> dict[str, Any]:
    return _engine().kill(reason)


@router.post("/api/fx/engine/release")
def engine_release() -> dict[str, Any]:
    return _engine().release()


@router.post("/api/fx/engine/flatten")
def engine_flatten() -> dict[str, Any]:
    try:
        return {"closed": _engine().flatten()}
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/api/fx/engine/once")
def engine_once() -> dict[str, Any]:
    """한 바퀴 수동 실행. 엔진이 꺼져 있으면 신호만 계산하고 주문은 내지 않는다."""
    try:
        return _engine().run_once()
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/api/fx/journal")
def engine_journal(n: int = 100) -> dict[str, Any]:
    """무엇을 왜 했는지. **막힌 기록도 남는다** — 없으면 '왜 진입 안 했지'에
    답할 수 없고, 그러면 사용자가 가드를 꺼 버린다."""
    return {"rows": _engine().journal.tail(n)}


# ─── 매매 내역 · 잔고 현황 ───────────────────────────────────────
#
# **왜 MT5 내역을 진실로 삼는가.** 우리 장부는 앱이 켜져 있던 동안만 안다.
# 손절이 새벽에 브로커에서 터졌거나 사용자가 MT5 창에서 직접 닫으면 장부에는
# 아무 줄도 없다. 그때 장부만 보여 주면 화면은 "거래 없음"이라고 말하는데
# 잔고는 줄어 있다. 그건 최악의 화면이다.
#
# 장부는 **MT5 가 모르는 것**을 채우는 데만 쓴다: 어떤 손절가로 들어갔는지,
# 그때 점수와 국면이 무엇이었는지. 그게 있어야 R배수를 계산할 수 있다.

def _journal_index() -> dict[str, dict]:
    """포지션 티켓 -> 우리가 주문할 때 기록해 둔 것."""
    try:
        rows = _engine().journal.tail(3000, kinds=["order"])
    except Exception:
        return {}
    idx: dict[str, dict] = {}
    for r in rows:
        for key in (str(r.get("ticket") or ""), str(r.get("tag") or "")):
            if key and key not in idx:
                idx[key] = r
    return idx


def _r_multiple(t: dict, order: dict | None) -> float | None:
    """
    이 거래가 몇 R 이었나. 손절 폭을 1R 로 본다.

    돈이 아니라 **가격으로** 잰다. 랏 크기·계약 크기·환율이 끼면 같은 판단이
    종목마다 다른 숫자로 보여서 비교가 안 된다.
    """
    if not order:
        return None
    try:
        entry = float(order.get("price") or t["entry_price"])
        stop = float(order.get("stop_loss") or 0)
    except (TypeError, ValueError):
        return None

    # 손절가 0 은 "손절가 없음"이다. 그걸 값으로 쓰면 1R 이 진입가 전체가 되어
    # 어떤 거래든 0.0x R 로 찍힌다 — 사람은 그걸 "본전"으로 읽는다.
    # 모르는 것은 모른다고 말해야 한다.
    if stop <= 0:
        return None
    risk = abs(entry - stop)
    if not risk:
        return None
    moved = t["exit_price"] - entry
    if t["side"] == "SHORT":
        moved = -moved
    return round(moved / risk, 2)


@router.get("/api/fx/history")
def fx_history(days: int = 30, limit: int = 60,
               only_ours: bool = False) -> dict[str, Any]:
    """
    닫힌 거래 목록. 우리가 연 것과 사람이 손으로 연 것을 **구분해서** 준다.
    섞어서 한 덩어리로 보여 주면 엔진 성적을 잘못 읽는다.
    """
    try:
        raw = fx_deals(days=max(days, 1))
        trades = closed_trades(raw=raw)
        flows = cash_flows(raw=raw)
    except MT5Error as exc:
        raise HTTPException(502, str(exc)) from exc

    idx = _journal_index()
    for t in trades:
        order = idx.get(str(t["position_id"])) or idx.get(t["comment"].strip())
        t["r_multiple"] = _r_multiple(t, order)
        t["score"] = order.get("score") if order else None
        t["regime"] = order.get("regime") if order else None
        t["stop_loss"] = order.get("stop_loss") if order else None
        # parse 는 브로커 심볼을 통화쌍으로 읽고, FX 가 아니면 None 을 준다.
        # 계좌에는 사람이 손으로 연 금·지수 거래가 섞여 있으므로 그 경우
        # 브로커 이름을 그대로 보여 주고 차트 링크는 달지 않는다.
        pair = parse(t["symbol"])
        t["display"] = pair.display if pair else t["symbol"]
        t["tradingview"] = pair.tradingview if pair else None
        t["is_fx"] = pair is not None

    shown = [t for t in trades if t["ours"]] if only_ours else trades
    return {
        "days": days,
        "trades": shown[:limit],
        "truncated": max(0, len(shown) - limit),
        "summary": _history_summary(trades),
        "funding": flows,
    }


def _history_summary(trades: list[dict]) -> dict[str, Any]:
    """
    성적 요약. **우리 거래와 사람 거래를 따로 센다.**

    승률만 보여 주지 않는다. 작게 여러 번 이기고 크게 한 번 지는 전략은
    승률이 높은데 돈을 잃는다 — 이 프로젝트에서 이미 겪었다(평균회귀 승률
    54.8%, 기대R 음수). 그래서 순손익과 평균 R 을 같이 보여 준다.
    """
    def block(rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {"trades": 0, "net": 0.0, "wins": 0, "win_rate": None,
                    "avg_r": None, "best": None, "worst": None, "costs": 0.0}
        wins = [r for r in rows if r["profit"] > 0]
        rs = [r["r_multiple"] for r in rows if r.get("r_multiple") is not None]
        return {
            "trades": len(rows),
            "net": round(sum(r["profit"] for r in rows), 2),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(rows), 3),
            "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
            "r_known": len(rs),
            "best": round(max(r["profit"] for r in rows), 2),
            "worst": round(min(r["profit"] for r in rows), 2),
            # 수수료와 스왑을 따로 보여 준다. 이 프로젝트의 결론이
            # "우위는 있는데 비용이 먹는다"였으므로 이 숫자가 중요하다.
            "costs": round(sum(r["commission"] + r["swap"] for r in rows), 2),
        }

    today = datetime.now(timezone.utc).date().isoformat()
    ours = [t for t in trades if t["ours"]]
    return {
        "ours": block(ours),
        "manual": block([t for t in trades if not t["ours"]]),
        "all": block(trades),
        "realised_today": round(
            sum(t["profit"] for t in trades if t["closed_at"][:10] == today), 2),
        "ours_today": round(
            sum(t["profit"] for t in ours if t["closed_at"][:10] == today), 2),
    }


@router.get("/api/fx/balance")
def fx_balance(days: int = 90) -> dict[str, Any]:
    """
    잔고 현황 — 지금 숫자와 **어디서 왔는지**를 같이 준다.

    잔고만 크게 띄우면 "이게 벌어서 이렇게 된 건지 원래 이랬는지"를 알 수
    없다. 입금액과 실현손익을 같이 보여 줘야 숫자가 뜻을 갖는다.
    """
    st = mt5_status()
    if not st.get("connected"):
        raise HTTPException(502, st.get("message") or "MT5 연결 안 됨")
    a = st["account"]

    try:
        raw = fx_deals(days=max(days, 1))
        trades = closed_trades(raw=raw)
        flows = cash_flows(raw=raw)
    except MT5Error:
        trades, flows = [], {"deposit": 0.0, "withdrawal": 0.0,
                             "net_funding": 0.0}

    unrealised = round(a["equity"] - a["balance"], 2)
    realised = round(sum(t["profit"] for t in trades), 2)
    funded = flows["net_funding"]
    # 곡선은 실현손익 누적으로 그린다. MT5 는 과거 잔고 스냅샷을 주지 않으므로
    # 이것이 우리가 정직하게 그릴 수 있는 유일한 곡선이다 (미실현은 안 들어간다).
    curve, run = [], funded or (a["balance"] - realised)
    for t in sorted(trades, key=lambda r: r["closed_at"]):
        run += t["profit"]
        curve.append({"at": t["closed_at"], "balance": round(run, 2)})

    return {
        "currency": a["currency"], "login": a["login"], "server": a["server"],
        "is_demo": a["is_demo"], "trade_allowed": a["trade_allowed"],
        "leverage": a["leverage"],
        "balance": round(a["balance"], 2),
        "equity": round(a["equity"], 2),
        "margin_free": round(a["margin_free"], 2),
        "margin_used": round(a["equity"] - a["margin_free"], 2),
        "unrealised": unrealised,
        "realised": realised,
        "funding": flows,
        # 입금액 대비 몇 % 인가. 입금 기록이 없으면(데모 초기잔고) 계산하지 않는다.
        "return_pct": (round((a["equity"] - funded) / funded * 100, 2)
                       if funded > 0 else None),
        "open_positions": len(st["positions"]),
        "curve": curve[-200:],
        "days": days,
    }
