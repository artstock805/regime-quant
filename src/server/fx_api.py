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

**두 시세는 다르다.** 트레이딩뷰(FX_IDC)는 은행 간 집계고 MT5 는 우리가 체결하는
브로커 호가다. 차트의 마커와 실제 체결가가 몇 핍 어긋나는 것은 버그가 아니다.
화면이 그 사실을 숨기지 않는다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import load_config
from ..fx import (
    DEFAULT_PROVIDER, MT5Error, available_symbols, bars, catalogue, resolve,
    status as mt5_status, symbol_info, tick,
)

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


@router.get("/fx")
def fx_page() -> FileResponse:
    return FileResponse(STATIC / "fx.html")


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
