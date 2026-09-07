"""
호가창 화면과 그 API. **/fx 와 일부러 분리해 뒀다.**

/fx 는 MT5 로 통화쌍을 거래하는 화면이고, 여기는 바이낸스 코인 호가창이다.
XM MT5 는 호가창을 아예 제공하지 않는다 (실측: EURUSD·USDJPY·GBPUSD·GOLD·
US500Cash·BTCUSD 전부 market_book_add=False). 딜러 시장이라 브로커가 자기
호가 한 단만 준다. 두 화면을 합치면 사람이 서로 무관한 정보를 같은 판단
근거로 읽게 되므로 파일부터 갈라 뒀다.

**서버가 중계한다.** 브라우저가 거래소에 직결하던 이전 방식은 탭을 닫으면
누적이 사라지고 전략이 그 데이터를 못 읽었다. 이제 서버가 연결을 하나 쥐고
상태를 쌓고, 브라우저는 그 스냅샷을 받아 그린다 (src/live/orderbook.py).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..live import orderbook

STATIC = Path(__file__).parent / "static"
router = APIRouter()

# 화면 코드가 바뀌었는데 브라우저가 옛 파일을 들고 있으면, 고친 것이 반영되지
# 않은 채 "왜 그대로지"를 디버깅하게 된다.
NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}

# 브라우저로 스냅샷을 밀어 주는 주기. 거래소는 100ms 로 오지만 그걸 그대로
# 흘리면 브라우저가 못 따라가고 대역만 먹는다. 눈으로 보기에 200ms 면 충분하다.
PUSH_SECONDS = 0.2


@router.get("/book")
def book_page() -> FileResponse:
    return FileResponse(STATIC / "book.html", headers=NO_CACHE)


@router.get("/api/book/snapshot")
def book_snapshot(symbol: str = "btcusdt", tape: int = 60,
                  foot: int = 40) -> dict[str, Any]:
    """
    한 번 읽기. 화면은 WebSocket 을 쓰고, 이 경로는 엔진과 사람이 확인할 때 쓴다.
    없으면 여기서 중계를 시작한다.
    """
    try:
        stream = orderbook.subscribe(symbol)
    except orderbook.UnknownSymbol as exc:
        raise HTTPException(400, str(exc)) from exc
    return stream.snapshot(tape=max(0, tape), foot=max(0, foot))


@router.get("/api/book/status")
def book_status() -> dict[str, Any]:
    """
    무엇이 돌고 있나. **한 번 열면 명시적으로 멈출 때까지 계속 돈다** —
    보이지 않으면 잊힌 연결이 며칠 붙어 있게 된다.
    """
    return orderbook.status()


@router.post("/api/book/reset")
def book_reset(symbol: str = "btcusdt") -> dict[str, Any]:
    """
    누적(델타·풋프린트·테이프)만 지운다. 호가창과 연결은 그대로 둔다.

    **누적이 서버에 있으므로 초기화도 서버가 해야 한다.** 브라우저에서만
    지우면 다른 탭과 엔진은 옛 누적을 계속 본다.
    """
    stream = orderbook.get(symbol)
    if stream is None:
        raise HTTPException(404, f"{symbol} 중계가 돌고 있지 않습니다")
    stream.reset()
    return {"reset": True, "symbol": stream.symbol}


@router.post("/api/book/stop")
def book_stop(symbol: str = "btcusdt") -> dict[str, Any]:
    """중계를 끊는다. 누적도 버린다."""
    return {"stopped": orderbook.stop(symbol), "symbol": symbol.lower()}


@router.websocket("/ws/book")
async def ws_book(ws: WebSocket) -> None:
    """
    화면에 스냅샷을 밀어 준다.

    **거래소 메시지를 그대로 중계하지 않는다.** 서버가 이미 상태를 들고 있으니
    주기마다 그 상태를 보내면 된다. 이렇게 하면 클라이언트가 몇 개든 거래소
    연결은 하나로 유지되고, 늦게 접속한 탭도 첫 프레임에서 곧바로 완성된
    호가창과 누적을 받는다.
    """
    await ws.accept()
    symbol = (ws.query_params.get("symbol") or "btcusdt").lower()
    try:
        stream = orderbook.subscribe(symbol)
    except orderbook.UnknownSymbol as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close()
        return

    try:
        while True:
            # 종목을 바꾸는 명령이 왔으면 갈아탄다. 연결을 다시 맺지 않아도
            # 되므로 화면 전환이 끊기지 않는다.
            try:
                msg = await asyncio.wait_for(ws.receive_text(),
                                             timeout=PUSH_SECONDS)
            except asyncio.TimeoutError:
                msg = None
            if msg:
                try:
                    cmd = __import__("json").loads(msg)
                except ValueError:
                    cmd = {}
                want = str(cmd.get("symbol") or "").lower()
                if cmd.get("cmd") == "symbol" and want and want != stream.symbol:
                    try:
                        stream = orderbook.subscribe(want)
                    except orderbook.UnknownSymbol as exc:
                        await ws.send_json({"type": "error",
                                            "message": str(exc)})
            await ws.send_json({"type": "snapshot", **stream.snapshot()})
    except (WebSocketDisconnect, RuntimeError):
        pass
