"""
유니버스 확장이 실제로 정보를 더하는가.

    python scripts/universe_scaling.py --trades runs/<run>/trades.parquet
    python scripts/universe_scaling.py --run deriv-taker-134

거래 수를 16배로 늘리면 순진한 t 는 √16 = 4배로 뛴다. 그런데 암호화폐 알트는
서로 0.8 이상 상관돼 있어서, 같은 주에 여러 종목에서 열린 거래는 **같은 시장
충격을 여러 번 센 것**에 가깝다. 늘어난 t 중 어디까지가 정보이고 어디부터가
중복인지를 여기서 가른다.

재는 것
  1) 종목 수별 유효표본 — 종목을 2배 늘리면 독립 정보는 몇 배가 되는가
  2) 그 배증계수의 감쇠 — 확장의 수익체감 곡선
  3) 목표 t 에 닿으려면 종목이 몇 개 필요한가 (외삽)

유효표본이 종목 수에 비례해 늘면 확장이 통한다. 제자리면 같은 것을 다시 센 것이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validation.stats import clustered_mean_test, time_clusters   # noqa: E402

pd.set_option("display.width", 200)


def scaling_curve(tr: pd.DataFrame, col: str, freq: str, draws: int,
                  seed: int = 0) -> pd.DataFrame:
    syms = list(tr["symbol"].unique())
    rng = np.random.default_rng(seed)
    ks, k = [], 1
    while k <= len(syms):
        ks.append(k)
        k *= 2
    if ks[-1] != len(syms):
        ks.append(len(syms))

    rows = []
    for k in ks:
        acc = []
        reps = 1 if k == len(syms) else draws
        for _ in range(reps):
            pick = rng.choice(syms, size=k, replace=False)
            sub = tr[tr["symbol"].isin(pick)]
            if len(sub) < 100:
                continue
            r = clustered_mean_test(sub[col].to_numpy(),
                                    time_clusters(sub["entry_time"], freq))
            if np.isfinite(r.t):
                acc.append((r.n, r.t_naive, r.t, r.inflation, r.effective_n))
        if not acc:
            continue
        a = np.array(acc)
        rows.append({"종목": k, "거래": a[:, 0].mean(), "순진t": a[:, 1].mean(),
                     "군집t": a[:, 2].mean(), "팽창": a[:, 3].mean(),
                     "유효표본": a[:, 4].mean()})
    out = pd.DataFrame(rows)
    out["배증계수"] = out["유효표본"] / out["유효표본"].shift(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="유니버스 확장 효과")
    ap.add_argument("--trades", default="runs/symbol_selection_trades.parquet")
    ap.add_argument("--column", default="r_multiple",
                    choices=["r_multiple", "gross_r"])
    ap.add_argument("--freq", default="W", help="군집 주기 (W=주, D=일, M=월)")
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--target-t", type=float, default=1.96)
    args = ap.parse_args()

    path = Path(args.trades)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"거래 파일이 없습니다: {path}")
        return 1
    tr = pd.read_parquet(path)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    if "symbol" not in tr.columns:
        print("symbol 열이 없습니다 — 종목별로 나눌 수 없습니다")
        return 1

    print(f"{len(tr):,} 거래 · {tr['symbol'].nunique()}종목 · "
          f"{tr['entry_time'].min():%Y-%m} ~ {tr['entry_time'].max():%Y-%m}")
    print(f"지표 {args.column} · 군집 {args.freq}\n")

    full = clustered_mean_test(tr[args.column].to_numpy(),
                               time_clusters(tr["entry_time"], args.freq))
    print("=" * 74)
    print("전체 표본")
    print("=" * 74)
    print(f"  기대값 {full.mean:+.4f}R")
    print(f"  순진한 t {full.t_naive:+.3f}  →  군집 t {full.t:+.3f}  (p={full.p:.4f})")
    print(f"  SE 팽창 {full.inflation:.2f}배 · 군집 {full.n_clusters}개")
    print(f"  유효표본 {full.effective_n:,.0f} / {full.n:,} "
          f"= 거래의 {full.effective_n / full.n * 100:.0f}% 만 독립 정보")

    print("\n" + "=" * 74)
    print("종목 수를 늘릴 때 독립 정보가 함께 느는가")
    print("=" * 74)
    curve = scaling_curve(tr, args.column, args.freq, args.draws)
    print(curve.round({"거래": 0, "순진t": 3, "군집t": 3, "팽창": 2,
                       "유효표본": 0, "배증계수": 2}).to_string(index=False))

    gains = curve["배증계수"].dropna()
    if len(gains) >= 2:
        print(f"\n  배증계수가 {gains.iloc[0]:.2f} → {gains.iloc[-1]:.2f} 로 줄어든다"
              f" — 종목을 늘릴수록 새 정보의 비중이 떨어진다.")

    # 목표 t 까지 필요한 유효표본
    if np.isfinite(full.t) and full.t != 0:
        need = (args.target_t / full.t) ** 2
        print(f"\n  t={args.target_t} 에 닿으려면 유효표본이 지금의 {need:.1f}배 필요하다"
              f" ({full.effective_n * need:,.0f})")
        if len(gains) >= 2 and gains.iloc[-1] > 1.0:
            doublings = np.log(need) / np.log(gains.iloc[-1])
            print(f"  현재 배증계수 {gains.iloc[-1]:.2f} 가 유지된다면 종목 "
                  f"{curve['종목'].iloc[-1] * 2 ** doublings:,.0f}개 — "
                  f"{'도달 가능' if doublings < 4 else '현실적으로 도달 불가'}")
            print("  (배증계수는 실제로는 계속 줄어드므로 이 수치는 낙관적인 하한이다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
