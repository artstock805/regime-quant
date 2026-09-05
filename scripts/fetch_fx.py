"""
MT5 에서 FX 과거 봉을 받아 캐시에 넣는다.

암호화폐와 달리 FX 는 무료 벌크 아카이브가 없다. 브로커마다 호가가 다르고,
**우리가 주문을 낼 브로커의 호가로 검증해야** 백테스트와 실전이 어긋나지 않는다.
그래서 데이터 출처가 MT5 하나다.

**봉마다 그때의 스프레드가 함께 온다.** 이게 중요하다. 지금 스프레드로 과거를
평가하면 비용을 과소평가한다 — 뉴스 시간과 아시아 새벽에는 스프레드가 몇 배로
벌어지고, 전략이 그 시간에 진입한다면 실제 비용은 훨씬 크다. 이 프로젝트는
암호화폐에서 "총우위는 실재하는데 비용이 그 59%를 먹어 순우위가 0" 이라는
결론에 도달했다. 비용을 낙관적으로 잡으면 그 결론 자체를 놓친다.

    python scripts/fetch_fx.py                    주요 통화쌍, 1시간봉
    python scripts/fetch_fx.py --interval 4h
    python scripts/fetch_fx.py --symbols EURUSD USDJPY --bars 50000

MetaTrader 5 터미널이 실행 중이고 로그인돼 있어야 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fx import (  # noqa: E402
    CROSSES, MAJORS, MT5Error, available_symbols, bars, catalogue, resolve,
    symbol_info,
)

CACHE = ROOT / "data" / "fx"


def cache_path(canonical: str, interval: str) -> Path:
    return CACHE / f"{canonical}_{interval}.parquet"


def fetch(canonical: str, broker_symbol: str, interval: str,
          count: int) -> pd.DataFrame:
    df = bars(broker_symbol, interval, count)
    si = symbol_info(broker_symbol)

    # spread 는 포인트 단위다. 가격 단위와 비율로 바꿔 둔다 — 백테스터가 쓰는
    # 비용은 비율이고, 여기서 바꿔 두지 않으면 매번 자릿수를 다시 따져야 한다.
    df = df.copy()
    df["spread_price"] = df["spread"] * si.point
    df["spread_rate"] = df["spread_price"] / df["close"]
    df.attrs["symbol"] = canonical
    df.attrs["broker_symbol"] = broker_symbol
    df.attrs["digits"] = si.digits
    df.attrs["point"] = si.point
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="MT5 에서 FX 과거 봉 받기")
    ap.add_argument("--interval", default="1h",
                    help="봉 간격 (1m/5m/15m/30m/1h/4h/1d). 기본 1h")
    ap.add_argument("--bars", type=int, default=30_000,
                    help="종목당 봉 개수. 기본 30000 (1시간봉 약 4.8년)")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="표준 이름으로 지정. 비우면 주요 통화쌍 전부")
    ap.add_argument("--crosses", action="store_true",
                    help="크로스 통화쌍도 포함")
    args = ap.parse_args()

    try:
        available = available_symbols(only_forex=True)
    except MT5Error as exc:
        print(f"MT5 에 붙지 못했습니다.\n{exc}")
        return 1

    if args.symbols:
        wanted = [s.upper() for s in args.symbols]
    else:
        wanted = list(MAJORS) + (list(CROSSES) if args.crosses else [])

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"브로커 FX 종목 {len(available)}개 · 받을 통화쌍 {len(wanted)}개 "
          f"· {args.interval} · 최대 {args.bars:,}봉\n")

    rows, failed = [], []
    for name in wanted:
        pair = resolve(name, available)
        if pair is None:
            failed.append((name, "이 브로커에 없습니다"))
            print(f"  {name:8s} 건너뜀 — 이 브로커에 없습니다")
            continue
        try:
            df = fetch(pair.canonical, pair.mt5, args.interval, args.bars)
        except MT5Error as exc:
            failed.append((name, str(exc)))
            print(f"  {name:8s} 실패 — {exc}")
            continue

        path = cache_path(pair.canonical, args.interval)
        df.to_parquet(path)
        spread_bps = df["spread_rate"].median() * 10_000
        rows.append({
            "종목": pair.canonical,
            "브로커심볼": pair.mt5,
            "봉": len(df),
            "시작": df.index[0].date(),
            "끝": df.index[-1].date(),
            "스프레드(bp, 중앙값)": round(spread_bps, 2),
            "스프레드(bp, 90분위)": round(
                df["spread_rate"].quantile(0.9) * 10_000, 2),
        })
        print(f"  {pair.canonical:8s} {len(df):>7,}봉  "
              f"{df.index[0].date()} ~ {df.index[-1].date()}  "
              f"스프레드 {spread_bps:.2f}bp")

    if not rows:
        print("\n받은 것이 없습니다.")
        return 1

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    print(f"\n캐시: {CACHE}")

    # 스프레드가 넓은 통화쌍은 우위가 있어도 비용이 먹는다. 미리 경고한다.
    wide = table[table["스프레드(bp, 중앙값)"] > 2.0]
    if not wide.empty:
        print("\n스프레드가 넓은 통화쌍 — 우위가 있어도 비용이 먹을 수 있습니다:")
        for _, r in wide.iterrows():
            print(f"  {r['종목']}  {r['스프레드(bp, 중앙값)']}bp "
                  f"(90분위 {r['스프레드(bp, 90분위)']}bp)")

    if failed:
        print(f"\n실패 {len(failed)}건:")
        for name, why in failed:
            print(f"  {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
