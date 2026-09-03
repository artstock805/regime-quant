"""
상황판 실행.

    python -m src.server.run                                  # BTCUSDT, 설정 기간 전체
    python -m src.server.run --symbol ETHUSDT --start 2025-09-01
    python -m src.server.run --entry-threshold 0.79           # 선별도 상향 설정으로

세션 구성에 walk-forward 완주가 포함되므로 첫 화면이 뜨기까지 수십 초 걸린다.
기간을 좁히면(--start) 훨씬 빨라진다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import uvicorn

    from src.config import load_config
    from src.server.app import app

    cfg = load_config()
    ap = argparse.ArgumentParser(description="레짐 퀀트 상황판")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default=None, help="기본값은 config 의 backtest.start")
    ap.add_argument("--end", default=None)
    ap.add_argument("--entry-threshold", type=float, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--prewarm", action="store_true",
                    help="서버 시작 전에 세션을 미리 만든다 (첫 화면이 즉시 뜬다)")
    args = ap.parse_args()

    app.state.opts = {
        "symbol": args.symbol, "interval": args.interval,
        "start": args.start, "end": args.end,
        "entry_threshold": args.entry_threshold,
    }

    print(f"종목 {args.symbol} · {args.interval} · "
          f"{args.start or cfg.require('backtest.start')} → {args.end or cfg.require('backtest.end')}")
    if args.entry_threshold is not None:
        print(f"진입 문턱 {args.entry_threshold}")

    if args.prewarm:
        from src.server.app import get_session
        print("세션 구성 중 (walk-forward 완주)...", flush=True)
        s = get_session()
        m = s.meta()["metrics"]
        print(f"완료: {s.n:,}봉 · 거래 {m['trades']}건 · "
              f"수익률 {m['total_return']*100:+.2f}% · 샤프 {m['sharpe']:+.2f}")

    print(f"\n  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
