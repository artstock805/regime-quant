"""
자동매매 API + 트레이딩뷰 웹훅.

화면 구성이 여기서 갈린다.

    /            자동매매 (본 화면) — 엔진 켜기·포지션·체결·킬 스위치
    /review      복기 — 백테스트·검증·지표 성적표
    /cockpit     복기용 차트 재생

**백테스트를 본 화면에서 뺀 이유**: 자동매매 앱을 켰는데 첫 화면이 백테스트면
지금 무슨 일이 일어나고 있는지 알 수 없다. 백테스트는 '왜 그랬는지 돌아보는' 도구라
복기 쪽에 둔다.

    GET  /api/live/status          엔진·포지션·잔고·가드 상태
    POST /api/live/start           자동매매 시작 (기본 테스트넷)
    POST /api/live/stop            중지 (flatten=true 면 포지션 정리)
    POST /api/live/kill            킬 스위치 — 새 진입 즉시 차단
    POST /api/live/release         킬 스위치 해제
    POST /api/live/flatten         열린 포지션 전부 정리
    POST /api/live/once            한 바퀴만 수동 실행 (테스트용)
    GET  /api/live/journal         매매 장부
    POST /api/webhook/tradingview  Pine 알림 수신
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..config import load_config
from ..execution import ExchangeError, MissingCredentials, Trader
from .home import STATIC

router = APIRouter()

_trader: Trader | None = None

# 웹훅 비밀. 이게 없으면 누구나 우리 서버에 주문을 시킬 수 있다.
# 로컬 전용이어도 열어 둘 이유가 없다.
WEBHOOK_SECRET_ENV = "REGIME_QUANT_WEBHOOK_SECRET"


def trader() -> Trader:
    global _trader
    if _trader is None:
        _trader = Trader(load_config())
    return _trader


# ─── 화면 ────────────────────────────────────────────────────────

@router.get("/review")
def review_page() -> FileResponse:
    return FileResponse(STATIC / "review.html")


# ─── 엔진 ────────────────────────────────────────────────────────

@router.get("/api/live/status")
def live_status() -> dict[str, Any]:
    t = trader()
    st = t.status()
    st["webhook_configured"] = bool(os.environ.get(WEBHOOK_SECRET_ENV))
    st["credentials"] = _credential_state(t)
    return st


def _credential_state(t: Trader) -> dict:
    """키가 있는지만 본다. 값은 절대 돌려주지 않는다."""
    from ..execution.exchange import load_credentials
    try:
        load_credentials(t.tcfg.testnet)
        return {"present": True, "message": ""}
    except MissingCredentials as exc:
        return {"present": False, "message": str(exc)}


@router.post("/api/live/start")
def live_start() -> dict[str, Any]:
    ok, msg = trader().start()
    return {"ok": ok, "message": msg, **trader().status()}


@router.post("/api/live/stop")
def live_stop(flatten: bool = False) -> dict[str, Any]:
    ok, msg = trader().stop(flatten=flatten)
    return {"ok": ok, "message": msg, **trader().status()}


@router.post("/api/live/kill")
def live_kill(reason: str = "사용자 요청") -> dict[str, Any]:
    trader().kill(reason)
    return {"ok": True, "message": "새 진입을 차단했습니다.", **trader().status()}


@router.post("/api/live/release")
def live_release() -> dict[str, Any]:
    trader().guard.release_kill()
    return {"ok": True, "message": "차단을 해제했습니다.", **trader().status()}


@router.post("/api/live/flatten")
def live_flatten() -> dict[str, Any]:
    n = trader().flatten_all("사용자 요청")
    return {"ok": True, "message": f"포지션 {n}건을 정리했습니다.", **trader().status()}


@router.post("/api/live/once")
def live_once() -> dict[str, Any]:
    """
    한 바퀴만 돌린다. 엔진을 켜기 전에 "지금 돌리면 뭘 할까"를 보는 용도다.
    가드와 주문 경로는 실제와 완전히 같다 — 리허설이 아니라 실제 실행이다.
    """
    try:
        return {"ok": True, "result": trader().run_once()}
    except (ExchangeError, MissingCredentials) as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/api/live/journal")
def live_journal(n: int = 100, kinds: str | None = None) -> dict[str, Any]:
    want = [k.strip() for k in kinds.split(",")] if kinds else None
    j = trader().journal
    return {"rows": j.tail(n, kinds=want), "summary": j.summary()}


# ─── 트레이딩뷰 웹훅 ─────────────────────────────────────────────

@router.post("/api/webhook/tradingview")
async def tradingview_webhook(request: Request) -> dict[str, Any]:
    """
    Pine Script 알림을 받는다.

    알림 메시지에 이 JSON 을 넣으세요.

        {"secret": "<비밀>", "symbol": "BTCUSDT", "action": "롱진입"}

    action 은 롱진입 / 숏진입 / 청산.

    **가격은 받지 않는다.** 알림이 보낸 가격을 그대로 믿으면 지연이나 오타가
    그대로 주문이 된다. 종목과 방향만 받고 가격·손절·목표는 우리가 계산한다.
    그리고 우리 엔진 신호와 **같은 리스크 가드**를 지난다 — 외부 신호라고
    가드를 건너뛰면 그게 가장 위험한 구멍이 된다.
    """
    secret = os.environ.get(WEBHOOK_SECRET_ENV)
    if not secret:
        raise HTTPException(
            503, f"웹훅 비밀이 설정돼 있지 않습니다. 환경변수 {WEBHOOK_SECRET_ENV} 를 "
                 "정하고 서버를 다시 켜세요.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "JSON 이 아닙니다")

    got = str(body.get("secret", ""))
    # 상수 시간 비교. 문자열 == 비교는 타이밍으로 비밀을 흘릴 수 있다.
    if not hmac.compare_digest(got, secret):
        raise HTTPException(401, "비밀이 맞지 않습니다")

    symbol = str(body.get("symbol", "")).upper().strip()
    action = str(body.get("action", "")).strip()
    if not symbol or action not in {"롱진입", "숏진입", "청산"}:
        raise HTTPException(
            400, "symbol 과 action(롱진입/숏진입/청산)이 필요합니다")

    return trader().handle_external(symbol, action, source="tradingview")
