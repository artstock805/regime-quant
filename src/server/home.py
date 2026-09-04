"""
홈 화면 — 아이콘을 누르면 처음 만나는 곳.

터미널을 모르는 사람이 여기서 모든 기능에 닿을 수 있어야 한다. 동시에
**판정(1/5)을 화면 맨 위에 두는 것**이 이 파일의 다른 절반이다. 기능을 먼저
보여 주고 경고를 아래에 숨기면 아무도 읽지 않는다.

    GET  /                       홈
    GET  /api/home/status        데이터·설정·판정 요약
    GET  /api/home/tasks         실행 가능한 작업 목록
    POST /api/home/run/{name}    작업 시작 (목록에 있는 것만)
    POST /api/home/stop          실행 중인 작업 중단
    GET  /api/home/log/{name}    작업 로그 (since 이후 줄만)
    GET  /api/home/signals       마지막 신호 조회 결과 (runs/signals.json)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import load_config
from .tasks import catalog, runner

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC = Path(__file__).parent / "static"

router = APIRouter()

# 검증 결과. 화면에 박아 두는 값이라 한 곳에서만 고친다.
VERDICT = {
    "passed": 1, "total": 5,
    "trades": 7496, "symbols": 129, "period": "2021-12 ~ 2026-08",
    "gross_r": 0.0260, "gross_t": 3.52, "gross_p": 0.0004,
    "net_r": 0.0106, "net_t": 1.42, "net_p": 0.154,
    "cost_share": 0.59, "dsr": 0.865, "dsr_threshold": 0.95,
    "order_execution": False,
}


def _dir_size(path: Path, limit: int = 4000) -> tuple[int, int]:
    """(파일 수, 바이트). 파일이 많으면 세다 만다 — 화면 하나 때문에 디스크를 훑지 않는다."""
    n = total = 0
    if not path.exists():
        return 0, 0
    for p in path.rglob("*.parquet"):
        n += 1
        try:
            total += p.stat().st_size
        except OSError:
            pass
        if n >= limit:
            break
    return n, total


@router.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "home.html")


@router.get("/api/home/status")
def status() -> dict[str, Any]:
    cfg = load_config()
    kd = cfg.path("paths.klines_dir")
    dd = cfg.path("paths.derivatives_dir")
    k_n, k_bytes = _dir_size(kd)
    d_n, d_bytes = _dir_size(dd)

    symbols = sorted({p.name.split("_")[0] for p in kd.glob("**/*.parquet")}) if kd.exists() else []
    deriv_symbols = sorted({p.name.split("-")[0] for p in dd.glob("**/*metrics.parquet")}) \
        if dd.exists() else []

    cur = runner.current
    return {
        "now": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "config": {
            "market": cfg.get("market"),
            "strategies": cfg.get("strategies"),
            "interval": cfg.get("live.interval", "1h"),
            "fee_rate": cfg.get("backtest.fee_rate"),
            "fee_kind": ("메이커" if float(cfg.get("backtest.fee_rate", 0)) <= 0.0002
                         else "테이커"),
            "entry_threshold": cfg.get("risk.entry_threshold"),
            "atr_stop_mult": cfg.get("risk.atr_stop_mult"),
            "reward_ratio": cfg.get("risk.reward_ratio"),
        },
        "data": {
            "klines_dir": str(kd), "klines_files": k_n, "klines_mb": round(k_bytes / 1e6, 1),
            "deriv_dir": str(dd), "deriv_files": d_n, "deriv_mb": round(d_bytes / 1e6, 1),
            "symbols": symbols[:200], "symbol_count": len(symbols),
            "deriv_symbol_count": len(deriv_symbols),
            "ready": k_n > 0,
            "deriv_ready": d_n > 0,
        },
        "running": None if not (cur and cur.running) else
                   {"name": cur.name, "label": cur.label},
        "reports": {
            "final": (ROOT / "runs" / "final" / "report.html").exists(),
            "signals": (ROOT / "runs" / "signals.json").exists(),
        },
    }


@router.get("/api/home/tasks")
def tasks() -> dict[str, Any]:
    cur = runner.current
    return {
        "tasks": [{"name": k, **v} for k, v in catalog().items()],
        "running": None if not (cur and cur.running) else
                   {"name": cur.name, "label": cur.label},
    }


@router.post("/api/home/run/{name}")
def run_task(name: str) -> dict[str, Any]:
    spec = catalog().get(name)
    if spec is None:
        # 목록에 없는 것은 실행하지 않는다. 임의 인자를 받으면 원격 코드 실행이 된다.
        raise HTTPException(404, f"그런 작업이 없습니다: {name}")
    ok, msg = runner.start(name, spec["label"], spec["argv"])
    return {"ok": ok, "message": msg, "name": name}


@router.post("/api/home/stop")
def stop_task() -> dict[str, Any]:
    return {"ok": runner.stop()}


@router.get("/api/home/log/{name}")
def task_log(name: str, since: int = 0) -> dict[str, Any]:
    t = runner.get(name)
    if t is None:
        return {"name": name, "running": False, "lines": [], "total_lines": 0,
                "returncode": None, "error": None, "never_run": True}
    return t.as_dict(since=since)


@router.get("/api/home/signals")
def last_signals() -> dict[str, Any]:
    p = ROOT / "runs" / "signals.json"
    if not p.exists():
        return {"available": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    sigs = data.get("signals", [])
    return {
        "available": True,
        "created": data.get("created"),
        "interval": data.get("interval"),
        "count": len(sigs),
        "actionable": [s for s in sigs
                       if s.get("action") in {"롱진입", "숏진입", "청산"}],
        "signals": sigs,
    }
