"""
파생 데이터 내려받기 — 미결제약정·롱숏비율·펀딩.

    python scripts/fetch_derivatives.py
    python scripts/fetch_derivatives.py --symbols BTCUSDT --start 2024-01-01

REST 는 30일치만 주므로 벌크 아카이브를 쓴다. 일 단위 ZIP 이라 종목당 2천 장쯤
받는다. 캐시가 있으면 빠진 구간만 채운다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                   # noqa: E402
from src.data.cache import get_klines                                # noqa: E402
from src.data.derivatives import (                                   # noqa: E402
    METRICS_START, align_to_bars, coverage, get_derivatives,
)

pd.set_option("display.width", 200)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="파생 데이터 내려받기")
    ap.add_argument("--symbols", nargs="*", default=list(cfg.require("universe")))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--deriv-dir", default=None)
    args = ap.parse_args()
    if args.deriv_dir is None:
        args.deriv_dir = cfg.path("paths.derivatives_dir")

    start = args.start or cfg.require("backtest.start")
    end = args.end or cfg.require("backtest.end")
    market = cfg.require("market")

    print(f"파생 데이터 · {len(args.symbols)}종목 · {start} ~ {end}")
    print(f"metrics 아카이브 시작일 {METRICS_START} — 그 이전은 존재하지 않습니다\n")

    rows = []
    for sym in args.symbols:
        t0 = time.time()
        d = get_derivatives(sym, start, end, market=market,
                            deriv_dir=args.deriv_dir, log=print)
        bars = get_klines(sym, args.interval, start, end, market=market,
                          klines_dir=cfg.path("paths.klines_dir"))
        joined = align_to_bars(bars, d, args.interval)
        cov = coverage(joined)
        rows.append({"종목": sym, "파생행": len(d), "봉": len(bars),
                     **{r["필드"]: r["커버리지_%"] for _, r in cov.iterrows()}})
        print(f"  {sym:<9} 파생 {len(d):>7,}행 · 봉 {len(bars):,} · "
              f"{time.time() - t0:.0f}초")

    print("\n" + "=" * 100)
    print("봉에 붙인 뒤 커버리지 (%)")
    print("=" * 100)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n2021-01 이전 봉은 metrics 가 없어 비어 있는 것이 정상입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
