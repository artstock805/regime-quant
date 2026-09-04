"""
상황판 서버.

    python -m src.server.run --symbol BTCUSDT --start 2025-09-01

    GET  /                     상황판 화면
    GET  /api/meta             세션 메타 + 백테스트 성적
    GET  /api/candles?lo&hi    캔들 구간
    GET  /api/frame/{i}        봉 i 시점의 상태
    GET  /api/leaderboard/{w}  창 w 의 전략 리더보드 (학습 구간에서 뽑힌 가중치)
    WS   /ws                   재생 스트림 (play/pause/seek/speed)

세션 구성은 walk-forward 완주를 포함해 수십 초가 걸린다. 첫 요청에서 만들고
메모리에 들고 있는다. 재생은 그 결과를 되돌리는 것이므로 즉시 응답한다.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_config
from ..regime import KOREAN, Regime
from .home import router as home_router
from .session import Session, build_session, first_traded_bar

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="레짐 퀀트 상황판", docs_url=None, redoc_url=None)
app.state.session = None
app.state.opts = {"symbol": "BTCUSDT", "interval": "1h", "start": None, "end": None,
                  "entry_threshold": None}


def get_session() -> Session:
    """
    세션은 **처음 필요할 때** 만든다. 서버가 뜰 때 만들면 데이터가 없거나 전략이
    요구하는 열이 없을 때 서버 자체가 죽고, 홈 화면조차 열리지 않는다.
    실패는 502 로 돌려 화면이 이유를 보여 주게 한다.
    """
    if app.state.session is None:
        cfg = load_config()
        o = app.state.opts
        try:
            app.state.session = build_session(
                cfg, o["symbol"], o["interval"], o["start"], o["end"],
                o["entry_threshold"])
        except Exception as exc:
            raise HTTPException(502, str(exc)) from exc
    return app.state.session


# ─── 화면 ────────────────────────────────────────────────────────
# "/" 는 홈(온보딩)이고 상황판은 "/cockpit" 이다. 아이콘을 눌렀을 때 처음 보이는
# 것이 데이터도 없이 실패하는 상황판이면 안 된다.

app.include_router(home_router)


@app.get("/cockpit")
def cockpit() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# ─── API ─────────────────────────────────────────────────────────

@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    s = get_session()
    m = s.meta()
    m["start_bar"] = first_traded_bar(s)
    return m


@app.get("/api/candles")
def api_candles(lo: int = 0, hi: int = 300) -> dict[str, Any]:
    s = get_session()
    lo = max(0, lo)
    hi = min(s.n, max(lo + 1, hi))
    return {"lo": lo, "hi": hi, "candles": s.candles(lo, hi)}


@app.get("/api/frame/{i}")
def api_frame(i: int) -> dict[str, Any]:
    return get_session().frame(i)


@app.get("/api/fills")
def api_fills(lo: int = 0, hi: int = 10**9) -> dict[str, Any]:
    """[lo, hi) 구간의 체결. 차트 마커와 체결 로그를 되감을 때 쓴다."""
    s = get_session()
    out = []
    for i in sorted(s.fills_at):
        if lo <= i < hi:
            for f in s.fills_at[i]:
                out.append({"i": i, **f})
    return {"fills": out}


@app.get("/api/regime-bands")
def api_regime_bands(lo: int = 0, hi: int = 10**9) -> dict[str, Any]:
    """레짐이 연속된 구간들. 차트 배경 띠를 그리는 데 쓴다."""
    s = get_session()
    hi = min(hi, s.n)
    r = s.regimes["regime"].iloc[lo:hi]
    bands, start, cur = [], lo, None
    for k, v in enumerate(r.to_numpy(), start=lo):
        v = v if isinstance(v, str) else None
        if v != cur:
            if cur is not None:
                bands.append({"lo": start, "hi": k, "regime": cur})
            start, cur = k, v
    if cur is not None:
        bands.append({"lo": start, "hi": hi, "regime": cur})
    return {"bands": bands}


@app.get("/api/leaderboard/{w}")
def api_leaderboard(w: int) -> dict[str, Any]:
    """
    창 w 의 전략 리더보드. 가중치는 **학습 구간**에서 뽑힌 값이고,
    성적도 학습 구간 성적이다 — 검증 구간 성적으로 가중치를 정하면 미래 참조다.
    """
    s = get_session()
    res = s.result
    if not 0 <= w < len(res.weights_per_window):
        raise HTTPException(404, f"창 {w} 이 없습니다 (0~{len(res.weights_per_window)-1})")

    weights = res.weights_per_window[w]
    scores = res.train_scores[w] if w < len(res.train_scores) else pd.DataFrame()
    korean = {st.name: st.korean for st in s.strategies}

    rows = []
    for name in weights.columns:
        for regime in [str(r) for r in Regime]:
            sc = scores[(scores["strategy"] == name) & (scores["regime"] == regime)] \
                if not scores.empty else pd.DataFrame()
            wt = float(weights.at[regime, name])
            rows.append({
                "strategy": name, "korean": korean.get(name, name),
                "regime": regime, "regime_korean": KOREAN[Regime(regime)],
                "weight": wt,
                "trades": int(sc.iloc[0]["trades"]) if not sc.empty else 0,
                "expectancy_r": (None if sc.empty or pd.isna(sc.iloc[0]["expectancy_r"])
                                 else float(sc.iloc[0]["expectancy_r"])),
                "win_rate": (None if sc.empty else float(sc.iloc[0]["win_rate"])),
                "status": ("참여" if wt > 0 else
                           ("표본부족" if (not sc.empty and sc.iloc[0]["trades"] < 20)
                            else ("표본없음" if sc.empty else "제외"))),
            })
    win = res.windows[w]
    return {
        "window": w,
        "train": [win.train_start.isoformat(), win.train_end.isoformat()],
        "test": [win.test_start.isoformat(), win.test_end.isoformat()],
        "rows": rows,
    }


@app.get("/api/equity")
def api_equity(step: int = 1) -> dict[str, Any]:
    """자산곡선. 봉이 5만 개라 그대로 보내면 무겁다 — step 으로 솎아 보낸다."""
    s = get_session()
    step = max(1, step)
    idx = s.bars.index[::step]
    vals = s.equity[::step]
    return {"t": [x.isoformat() for x in idx], "equity": [float(v) for v in vals]}


@app.get("/api/windows")
def api_windows() -> dict[str, Any]:
    """창별 검증 성적. 한 창이 전체를 끌고 가는지 화면에서 바로 보이게 한다."""
    s = get_session()
    tr = s.result.oos_trades
    rows = []
    for i, w in enumerate(s.result.windows):
        block = tr[(tr["entry_time"] >= w.test_start) & (tr["entry_time"] < w.test_end)] \
            if not tr.empty else pd.DataFrame()
        rows.append({
            "window": i,
            "period": f"{w.test_start:%Y-%m}~{w.test_end:%Y-%m}",
            "trades": len(block),
            "sum_r": (round(float(block["r_multiple"].sum()), 3) if not block.empty else 0.0),
            "mean_r": (round(float(block["r_multiple"].mean()), 4) if not block.empty else None),
        })
    return {"rows": rows}


# ─── 재생 스트림 ─────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_replay(ws: WebSocket) -> None:
    await ws.accept()
    s = get_session()
    i = first_traded_bar(s)
    playing = False
    speed = 4.0          # 초당 봉 수

    async def send_frame(idx: int) -> None:
        await ws.send_text(json.dumps({"type": "frame", **s.frame(idx)}, default=str))

    await ws.send_text(json.dumps({"type": "ready", "n": s.n, "i": i}))
    await send_frame(i)

    async def reader() -> None:
        nonlocal playing, speed, i
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            cmd = msg.get("cmd")
            if cmd == "play":
                playing = True
            elif cmd == "pause":
                playing = False
            elif cmd == "speed":
                speed = max(0.25, min(120.0, float(msg.get("value", 4))))
            elif cmd == "seek":
                i = max(0, min(s.n - 1, int(msg.get("value", 0))))
                await send_frame(i)
            elif cmd == "step":
                i = max(0, min(s.n - 1, i + int(msg.get("value", 1))))
                await send_frame(i)

    task = asyncio.create_task(reader())
    try:
        while True:
            if playing and i < s.n - 1:
                i += 1
                await send_frame(i)
                await asyncio.sleep(1.0 / speed)
            else:
                if playing and i >= s.n - 1:
                    playing = False
                    await ws.send_text(json.dumps({"type": "end"}))
                await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
