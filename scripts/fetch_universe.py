"""
확장 유니버스 내려받기 — 생존편향 없는 종목 집합.

    python scripts/fetch_universe.py --workers 5
    python scripts/fetch_universe.py --dry-run

**왜 상장폐지 종목까지 받는가.** 지금 상장된 종목만 쓰면 생존편향이 들어간다.
아카이브에 2021-12 데이터가 있는 USDT 무기한 134개 중 **44개(33%)가 이미
상장폐지**됐다 — LUNA, FTM, SRM, BZRXUSDT 처럼 무너진 것들이다. 이들을 빼면
백테스트가 "살아남은 종목만 골라 거래한" 결과가 되고, 그건 실전에서 재현할 수 없다.

종목 선정 규칙은 사전에 적용 가능한 것만 쓴다 — **2021-12-01 에 파생 아카이브가
존재했는가**. 오늘의 거래대금으로 고르면 그것 자체가 미래 정보다.

받는 것: 1시간봉(신호) · 1분봉(봉내부 판정) · 파생(OI·펀딩·롱숏)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                   # noqa: E402
from src.data.cache import get_klines                                # noqa: E402
from src.data.derivatives import get_derivatives                     # noqa: E402

CANDIDATES = ROOT / "runs" / "universe_candidates.json"


def fetch_one(sym: str, cfg, start: str, end: str, skip_minute: bool) -> dict:
    kd, dd = cfg.path("paths.klines_dir"), cfg.path("paths.derivatives_dir")
    market = cfg.require("market")
    t0 = time.time()
    row = {"종목": sym}
    try:
        h1 = get_klines(sym, "1h", start, end, market=market, klines_dir=kd)
        row["1h봉"] = len(h1)
        row["시작"] = f"{h1.index.min():%Y-%m}" if len(h1) else None
        row["끝"] = f"{h1.index.max():%Y-%m}" if len(h1) else None
        if not skip_minute:
            row["1m봉"] = len(get_klines(sym, "1m", start, end,
                                         market=market, klines_dir=kd))
        d = get_derivatives(sym, start, end, market=market, deriv_dir=dd)
        row["파생행"] = len(d)
        row["초"] = round(time.time() - t0)
    except Exception as exc:                       # 한 종목이 죽어도 나머지는 계속
        row["오류"] = f"{type(exc).__name__}: {exc}"[:120]
        row["초"] = round(time.time() - t0)
    return row


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="확장 유니버스 내려받기")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--start", default="2021-12-01",
                    help="파생 아카이브가 전 종목에 존재하는 시작일")
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=5,
                    help="동시에 받을 종목 수. 종목 하나가 이미 8스레드를 쓴다")
    ap.add_argument("--skip-minute", action="store_true",
                    help="1분봉을 건너뛴다. 봉내부 판정이 보수적으로 바뀌므로 "
                         "기존 종목과 비교가 어긋난다 — 급할 때만")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.symbols:
        symbols = list(args.symbols)
    elif CANDIDATES.exists():
        symbols = json.loads(CANDIDATES.read_text())
    else:
        print(f"후보 목록이 없습니다: {CANDIDATES}")
        return 1

    end = args.end or cfg.require("backtest.end")
    print(f"유니버스 {len(symbols)}종목 · {args.start} ~ {end} · 동시 {args.workers}")
    print(f"저장 위치: {cfg.path('paths.klines_dir')}")
    if args.dry_run:
        print("\n".join(f"  {s}" for s in symbols))
        return 0

    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, s, cfg, args.start, end, args.skip_minute): s
                   for s in symbols}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            rate = (time.time() - t0) / done
            eta = rate * (len(symbols) - done) / 60
            status = row.get("오류") or f"1h {row.get('1h봉', 0):>6,} · 파생 {row.get('파생행', 0):>7,}"
            print(f"  [{done:>3}/{len(symbols)}] {row['종목']:<14} {status}  "
                  f"({row['초']}초, 남은 예상 {eta:.0f}분)", flush=True)

    tbl = pd.DataFrame(rows).sort_values("종목")
    out = ROOT / "runs" / "universe_fetch.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    bad = tbl[tbl.get("오류").notna()] if "오류" in tbl.columns else tbl.iloc[:0]
    print(f"\n완료 {len(tbl)}종목 · 실패 {len(bad)}개 · {(time.time() - t0) / 60:.0f}분")
    if not bad.empty:
        print(bad[["종목", "오류"]].to_string(index=False))
    print(f"요약 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
