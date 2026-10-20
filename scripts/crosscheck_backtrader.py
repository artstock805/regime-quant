"""
엔진의 거래 목록을 backtrader 로 재현해 대조한다 — 배선 버그를 잡는 용도.

**왜 이 스크립트가 있나.** 이 프로젝트에서 숫자를 망친 것은 규칙이 아니라
배선이었다. stop_loss=0 을 진짜 손절로 받아 1R 이 진입가가 된 적이 있고,
spread==0 을 "비용 없음"으로 읽어 에너지 계열이 유일한 양수라는 거짓 결과가
나온 적이 있다. 둘 다 규칙은 맞았고 배선이 틀렸다. 같은 코드를 다시 읽는
것보다 **다른 구현이 같은 답을 내는지** 보는 편이 낫다.

**일치가 뜻하는 것.** 봉 정렬·신호 지연·ATR·손절·추적손절·비용 산술이
독립 구현과 같다는 뜻이다. **규칙 자체가 옳다는 뜻은 아니다** — 둘 다 같은
명세를 구현하므로, 명세가 틀렸으면 사이좋게 같이 틀린다.

**어긋나면 첫 번째만 본다.** 거래 하나가 어긋나면 현금이 달라지고, 그러면
그 뒤 모든 거래의 수량이 달라져 폭포처럼 번진다. 보고서 맨 위의 '첫 어긋남'
하나를 고치고 다시 돌리는 것이 맞다.

사용법
    python scripts/crosscheck_backtrader.py
    python scripts/crosscheck_backtrader.py --symbols EURUSD,GBPUSD --bars 8000
    python scripts/crosscheck_backtrader.py --strategy vixfix_reversal --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import ExecParams                                 # noqa: E402
from src.backtest.crosscheck import WARMUP_BARS, crosscheck         # noqa: E402
from src.config import Costs, load_config                           # noqa: E402
from src.fx.mt5 import bars, symbol_info                            # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.indicators.core import atr as atr_of                       # noqa: E402
from src.strategies import build_all                                # noqa: E402

DEFAULT_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF")

# FX 에서 돌지 않는 전략(파생 데이터를 요구한다)은 애초에 대조 대상이 아니다.
DERIVATIVE_PREFIXES = ("perp_", "funding", "oi_", "basis")


def fx_strategies(only: str | None):
    out = []
    for s in build_all():
        if any(str(n).startswith(DERIVATIVE_PREFIXES) for n in getattr(s, "needs", ())):
            continue
        if only and s.name != only:
            continue
        out.append(s)
    return out


def half_spread(df, si) -> float:
    """
    편도 비용. **spread==0 인 봉을 뺀다** — MT5 옛 이력에는 스프레드가 없고
    0 으로 온다. "비용 없음" 이 아니라 "모름" 이다.
    """
    import numpy as np
    if "spread" not in df.columns:
        return 0.0
    rel = (df["spread"] * si.point / df["close"]).replace([np.inf, -np.inf], np.nan)
    nz = rel[(df["spread"] > 0) & rel.notna()]
    return float(nz.median()) / 2.0 if len(nz) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=4000)
    ap.add_argument("--strategy", default=None, help="하나만 대조한다")
    ap.add_argument("--warmup", type=int, default=WARMUP_BARS)
    ap.add_argument("--verbose", action="store_true",
                    help="어긋난 항목을 전부 나열한다")
    args = ap.parse_args()

    cfg = load_config()
    ep = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    strats = fx_strategies(args.strategy)
    if not strats:
        print(f"전략을 못 찾았다: {args.strategy}")
        return 1

    names = [x.strip() for x in args.symbols.split(",") if x.strip()]
    total = mismatched = skipped = 0
    failures: list[str] = []

    print(f"손절 {ep.atr_stop_mult}xATR · 워밍업 {args.warmup}봉 · "
          f"{args.interval} {args.bars}봉 · 전략 {len(strats)}개")
    print("=" * 78)

    for sym in names:
        df = bars(sym, args.interval, args.bars)
        if len(df) <= args.warmup + 100:
            print(f"{sym:10s} 봉 부족 ({len(df)}) — 건너뜀")
            continue
        feats = compute_all(df)
        atr = atr_of(df, 14)
        costs = Costs(fee_rate=half_spread(df, symbol_info(sym)), slippage_bps=0.5)

        for s in strats:
            try:
                sig = s.generate(df, feats)
            except KeyError:
                skipped += 1
                continue
            xc = crosscheck(df, sig, atr, interval=args.interval, costs=costs,
                            params=ep, warmup=args.warmup, symbol=sym,
                            strategy=s.name)
            total += 1
            tag = "일치" if xc.agrees else "어긋남"
            print(f"{sym:10s} {s.name:24s} 거래 {xc.compared:>4} · {tag}")
            if not xc.agrees:
                mismatched += 1
                failures.append(f"{sym} / {s.name}\n" +
                                "\n".join("  " + l for l in xc.report().split("\n")))
                if args.verbose:
                    for d in xc.divergences[:20]:
                        print(f"      {d}")

    print("=" * 78)
    print(f"대조 {total}쌍 · 어긋남 {mismatched}쌍"
          + (f" · 특징열 없어 건너뜀 {skipped}" if skipped else ""))
    if failures:
        print("\n── 어긋난 것들 ──")
        for f in failures:
            print(f)
        print("\n첫 어긋남 하나를 고치고 다시 돌리세요. 그 아래는 대개")
        print("현금이 달라져 번진 그림자라 따로 고칠 것이 없습니다.")
        return 1

    print("\n엔진의 배선이 독립 구현과 같습니다.")
    print("**규칙이 옳다는 뜻은 아닙니다** — 둘 다 같은 명세를 구현합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
