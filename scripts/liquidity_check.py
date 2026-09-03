"""
우위가 유동성에 어떻게 분포하는가 — 슬리피지 과소평가인지 가른다.

    python scripts/liquidity_check.py --run runs/taker-129

비용 모델은 전 종목에 **같은 슬리피지(bps)** 를 쓴다. 대형 종목에는 맞지만
소형 알트에는 낙관적이다. 그래서 확장 결과에서 반드시 물어야 하는 질문이 있다.

    우위가 유동성 낮은 종목에 몰려 있는가?

몰려 있다면 그 우위는 실재하는 게 아니라 **비용을 덜 매긴 결과**다. 유동성
구간별로 고르게 나오면 모델의 결함이 결론을 만든 것은 아니라고 말할 수 있다.

유동성은 백테스트 구간의 **평균 체결대금**으로 잰다. 오늘의 거래대금을 쓰면
그것 자체가 미래 정보다(오늘 큰 종목은 그동안 살아남고 커진 종목이다).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                      # noqa: E402
from src.data.cache import cache_path, load_cached                      # noqa: E402
from src.validation.stats import clustered_mean_test, time_clusters     # noqa: E402

pd.set_option("display.width", 200)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="유동성별 우위 분포")
    ap.add_argument("--run", default="runs/taker-129")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--groups", type=int, default=3)
    args = ap.parse_args()

    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    tp = run / "trades.parquet"
    if not tp.exists():
        print(f"거래 파일이 없습니다: {tp}")
        return 1
    tr = pd.read_parquet(tp)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)

    # ── 유동성: 백테스트 구간 평균 시간당 체결대금
    kd = cfg.path("paths.klines_dir")
    liq = {}
    for sym in tr["symbol"].unique():
        df = load_cached(cache_path(kd, cfg.require("market"), sym, "1h"))
        if df.empty or "quote_volume" not in df.columns:
            continue
        liq[sym] = float(df["quote_volume"].median())
    liq = pd.Series(liq, name="시간당체결대금")
    print(f"{len(liq)}종목 유동성 측정 (구간 중위 시간당 체결대금)")
    print(f"  최소 {liq.min():,.0f}  중위 {liq.median():,.0f}  최대 {liq.max():,.0f} USDT\n")

    per = tr.groupby("symbol").agg(
        거래=("r_multiple", "size"),
        순기대R=("r_multiple", "mean"),
        총기대R=("gross_r", "mean"),
        비용R=("cost_r", "mean"),
    )
    per = per[per["거래"] >= args.min_trades].join(liq, how="inner").dropna()
    per["유동성분위"] = pd.qcut(per["시간당체결대금"], args.groups,
                              labels=[f"{i+1}분위" for i in range(args.groups)])

    print("=" * 92)
    print(f"유동성 {args.groups}분위별 (거래 {args.min_trades}건 이상 {len(per)}종목)")
    print("=" * 92)
    grp = per.groupby("유동성분위", observed=True).agg(
        종목=("거래", "size"), 거래=("거래", "sum"),
        중위체결대금=("시간당체결대금", "median"),
        순기대R=("순기대R", "mean"), 총기대R=("총기대R", "mean"),
        비용R=("비용R", "mean"), 양수종목=("순기대R", lambda s: int((s > 0).sum())),
    )
    grp["양수비율_%"] = (grp["양수종목"] / grp["종목"] * 100).round(0)
    print(grp.round({"중위체결대금": 0, "순기대R": 4, "총기대R": 4, "비용R": 4}).to_string())

    # 거래 단위로 군집 검정 — 분위별
    print("\n" + "=" * 92)
    print("분위별 거래 단위 군집 검정 (총기대R — 우위 자체의 유무)")
    print("=" * 92)
    tr2 = tr.merge(per[["유동성분위"]], left_on="symbol", right_index=True, how="inner")
    rows = []
    for q, g in tr2.groupby("유동성분위", observed=True):
        r = clustered_mean_test(g["gross_r"].to_numpy(),
                                time_clusters(g["entry_time"], "W"))
        rn = clustered_mean_test(g["r_multiple"].to_numpy(),
                                 time_clusters(g["entry_time"], "W"))
        rows.append({"분위": q, "거래": r.n, "총기대R": round(r.mean, 4),
                     "총_군집t": round(r.t, 2) if np.isfinite(r.t) else None,
                     "순기대R": round(rn.mean, 4),
                     "순_군집t": round(rn.t, 2) if np.isfinite(rn.t) else None})
    qt = pd.DataFrame(rows)
    print(qt.to_string(index=False))

    print("\n" + "=" * 92)
    print("판단")
    print("=" * 92)
    lo, hi = qt.iloc[0], qt.iloc[-1]
    spread = abs(hi["총기대R"] - lo["총기대R"])
    print(f"  최저 유동성 분위 총기대R {lo['총기대R']:+.4f} vs "
          f"최고 유동성 분위 {hi['총기대R']:+.4f}")
    if lo["총기대R"] > hi["총기대R"] * 1.5 and lo["총기대R"] > 0:
        print("  ⚠ 우위가 유동성 낮은 쪽에 몰려 있다. 같은 슬리피지를 매긴 것이")
        print("    결론을 만들었을 가능성이 크다 — 소형 종목 실슬리피지는 훨씬 크다.")
    elif spread < abs(qt["총기대R"]).mean() * 0.6:
        print("  우위가 유동성 구간에 고르게 퍼져 있다. 슬리피지 과소평가가")
        print("  결론을 만든 것은 아니라고 볼 수 있다 (모델 결함이 없다는 뜻은 아니다).")
    else:
        print("  분위 간 차이가 있으나 방향이 단조롭지 않다 — 표를 직접 확인할 것.")
    print("\n  주의: 이 검사는 '분포가 이상하지 않은가'만 본다. 소형 종목의 실제")
    print("        슬리피지를 재려면 호가 데이터가 필요하고, 그건 아카이브에 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
