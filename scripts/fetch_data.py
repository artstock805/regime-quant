"""
캔들 데이터 수집 CLI.

    python scripts/fetch_data.py                     # 설정의 신호 타임프레임 전부
    python scripts/fetch_data.py --intervals 1m      # 체결 판정용 1분봉만
    python scripts/fetch_data.py --symbols BTCUSDT --start 2026-01-01
    python scripts/fetch_data.py --summary           # 캐시 현황만 출력

종목×인터벌 조합을 스레드로 병렬 처리한다. data.binance.vision 은 정적 파일
호스팅이라 병렬로 받아도 레이트리밋에 걸리지 않는다. 조합마다 각자의 parquet에
쓰기 때문에 경합도 없다.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.data.cache import cache_summary, get_klines      # noqa: E402

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def fetch_one(symbol: str, interval: str, start: str, end: str, market: str,
              klines_dir: Path, refresh: bool) -> dict:
    t0 = time.time()
    try:
        df = get_klines(
            symbol, interval, start, end,
            market=market, klines_dir=klines_dir, refresh=refresh,
        )
        return {
            "symbol": symbol, "interval": interval, "bars": len(df),
            "start": df.index[0] if len(df) else None,
            "end": df.index[-1] if len(df) else None,
            "secs": round(time.time() - t0, 1), "error": None,
        }
    except Exception as exc:                              # noqa: BLE001
        return {
            "symbol": symbol, "interval": interval, "bars": 0,
            "start": None, "end": None,
            "secs": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="바이낸스 캔들 수집")
    ap.add_argument("--symbols", nargs="*", default=cfg.require("universe"))
    ap.add_argument("--intervals", nargs="*", default=cfg.require("timeframes.signal"))
    ap.add_argument("--start", default=cfg.require("backtest.start"))
    ap.add_argument("--end", default=cfg.require("backtest.end"))
    ap.add_argument("--market", default=cfg.require("market"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 다시 받는다")
    ap.add_argument("--summary", action="store_true", help="캐시 현황만 출력하고 종료")
    args = ap.parse_args()

    klines_dir = cfg.path("paths.klines_dir")

    if args.summary:
        table = cache_summary(klines_dir, args.market)
        print(table.to_string(index=False) if not table.empty else "캐시가 비어 있습니다.")
        return 0

    jobs = [(s, i) for s in args.symbols for i in args.intervals]
    print(f"수집 대상 {len(jobs)}건 · {args.market} · {args.start} → {args.end}")
    print(f"종목 {len(args.symbols)} × 인터벌 {len(args.intervals)} · 워커 {args.workers}\n")

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, s, i, args.start, args.end,
                        args.market, klines_dir, args.refresh): (s, i)
            for s, i in jobs
        }
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = "실패" if r["error"] else f"{r['bars']:>9,}봉"
            _log(f"[{n:>3}/{len(jobs)}] {r['symbol']:<10} {r['interval']:<4} {tag}  {r['secs']:>6.1f}s"
                 + (f"  {r['error']}" if r["error"] else ""))

    elapsed = time.time() - t0
    ok = [r for r in results if not r["error"]]
    bad = [r for r in results if r["error"]]
    total_bars = sum(r["bars"] for r in ok)

    print(f"\n완료 {len(ok)}/{len(jobs)} · 총 {total_bars:,}봉 · {elapsed/60:.1f}분")
    if bad:
        print(f"\n실패 {len(bad)}건:")
        for r in bad:
            print(f"  {r['symbol']} {r['interval']}: {r['error']}")

    table = cache_summary(klines_dir, args.market)
    if not table.empty:
        print("\n캐시 현황")
        print(table.to_string(index=False))

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
