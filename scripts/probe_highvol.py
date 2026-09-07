"""
VixFix 반등이 고변동 국면에서 정말 우위가 있나 — 홀드아웃으로 검정한다.

**가설이 어떻게 좁혀졌나.**
    일봉 측정에서 국면별 기대R 중위가 high_vol 만 양수였다(+0.0834).
    그런데 11개 전략 중 high_vol 을 선언한 것은 `vixfix_reversal` **하나뿐**이다.
    즉 "고변동이 양수" 는 실은 "**VixFix 반등이 고변동에서 양수**" 라는
    단일 전략 주장이다. 14종목 중 11종목이 양수였고 기대R 이 +0.02~+0.42 였다.

**왜 같은 종목으로 다시 재면 안 되나.**
    고변동을 파기로 한 근거가 그 14종목의 결과다. 같은 데이터로 확인하면
    순환논리이고, 우연히 좋았던 것을 확인으로 착각한다. 그래서 종목을 둘로
    나눈다.

        기존군  이미 잰 14종목 — 가설을 만든 데이터. **참고용이다.**
        홀드아웃 아직 안 잰 종목 — 이 가설에 대해 **처음 보는 데이터**.

    판단은 홀드아웃으로 한다. 기존군이 좋고 홀드아웃이 나쁘면 그건 우연이었다.

**대조군을 같이 본다.**
    vixfix_reversal 은 high_vol 과 range 를 선언한다. 고변동에서만 양수면
    "고변동에서 통한다" 는 구체적 주장이 되고, 두 국면 모두 양수면 그냥
    "이 전략이 통한다" 다. 둘은 다른 결론이므로 나눠 봐야 한다.

사용법
    python scripts/probe_highvol.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import ExecParams                                 # noqa: E402
from src.backtest.walkforward import run_standalone                 # noqa: E402
from src.config import Costs                                        # noqa: E402
from src.fx.engine import trader                                    # noqa: E402
from src.fx.mt5 import bars, symbol_info                            # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.regime import classify                                     # noqa: E402

TARGET = "vixfix_reversal"

# 가설을 만든 종목들. 이걸로 확인하면 순환논리다 — 참고로만 낸다.
SEEN = ("GOLD", "US100Cash", "JP225Cash", "XAUEUR", "GER40Cash", "US500Cash",
        "US30Cash", "FRA40Cash", "OILCash", "BRENTCash", "SILVER",
        "EURUSD", "USDJPY", "NZDUSD")

# 이 가설에 대해 처음 보는 종목들. **판단은 여기서 한다.**
# 만기물은 넣지 않는다(만기에 심볼이 사라진다). NGASCash·XPDUSD·XPTUSD 는
# 비용이 1R 의 17~34% 라 애초에 거래 대상이 아니지만, **비용이 큰 곳에서
# 무너지는지 보는 것도 정보**이므로 넣고 따로 표시한다.
HOLDOUT = (
    # 지수 (아직 안 잰 것)
    "AUS200Cash", "EU50Cash", "HK50Cash", "IT40Cash", "NETH25Cash",
    "SWI20Cash", "UK100Cash", "CA60Cash", "ChinaHCash", "US2000Cash",
    "GerMid50Cash", "GerTech30Cash",
    # 금속·에너지 (아직 안 잰 것)
    "XPTUSD", "XPDUSD", "NGASCash",
    # FX 크로스 (메이저와 상관이 낮아 독립 관측에 가깝다)
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "CADJPY", "CHFJPY",
    "AUDNZD", "EURCHF", "GBPAUD",
)

HIGH_COST = {"NGASCash", "XPDUSD", "XPTUSD"}


def half_spread(df: pd.DataFrame, si) -> float:
    """
    편도 비용. **spread==0 인 봉을 뺀다** — MT5 옛 이력에는 스프레드가 기록돼
    있지 않고 0 으로 오는데 그건 "비용 없음"이 아니라 "모름"이다. 안 빼면
    공짜로 거래하는 셈이 되어 거짓 양수가 나온다(실제로 한 번 겪었다).
    """
    if "spread" not in df.columns:
        return 0.0
    rel = (df["spread"] * si.point / df["close"]).replace(
        [np.inf, -np.inf], np.nan)
    nz = rel[(df["spread"] > 0) & rel.notna()]
    return float(nz.median()) / 2.0 if len(nz) else 0.0


def trades_of(sym: str, t, interval: str, nbars: int) -> pd.DataFrame:
    """VixFix 반등의 거래 원본. 국면 라벨이 붙어 있다."""
    df = bars(sym, interval, nbars)
    if len(df) < 800:
        return pd.DataFrame()
    feats = compute_all(df)
    si = symbol_info(sym)
    half = half_spread(df, si)
    if half <= 0:
        return pd.DataFrame()
    regs = classify(feats)["regime"]
    strat = [s for s in t.strategies() if s.name == TARGET]
    if not strat:
        raise SystemExit(f"{TARGET} 전략을 못 찾았다")
    res = run_standalone(df, feats, strat, interval=interval,
                         costs=Costs(fee_rate=half, slippage_bps=0.5),
                         params=ExecParams(), initial_equity=10_000,
                         minute_df=None, regimes=regs, symbol=sym)
    tr = res[TARGET].trades
    if tr.empty:
        return tr
    tr = tr.copy()
    tr["symbol"] = sym
    tr["cost_bp"] = round(half * 2 * 1e4, 2)
    tr["bars"] = len(df)
    return tr


def summarise(tr: pd.DataFrame, regime: str) -> dict:
    """
    한 국면의 거래를 모아 t검정한다.

    **거래 단위로 t검정을 해야 한다.** 종목별 기대R 을 평균하면 거래 수가
    다른 종목이 같은 무게를 갖고, 표본 30개인 종목이 300개인 종목과 동등해진다.
    """
    g = tr[tr["regime"] == regime]
    if g.empty:
        return {"거래": 0}
    r = g["r_multiple"].astype("float64")
    n = len(r)
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    tstat = float(r.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    from scipy import stats as st
    p = float(2 * (1 - st.t.cdf(abs(tstat), n - 1))) if np.isfinite(tstat) else float("nan")
    return {
        "거래": n,
        "기대R": round(float(r.mean()), 4),
        "중위R": round(float(r.median()), 4),
        "t": round(tstat, 2) if np.isfinite(tstat) else None,
        "p": round(p, 4) if np.isfinite(p) else None,
        "승률": round(float((r > 0).mean()), 3),
        "비용R": round(float(g["cost_r"].mean()), 4),
    }


def by_symbol(tr: pd.DataFrame, regime: str) -> pd.DataFrame:
    g = tr[tr["regime"] == regime]
    if g.empty:
        return pd.DataFrame()
    out = g.groupby("symbol").agg(
        거래=("r_multiple", "size"),
        기대R=("r_multiple", "mean"),
        승률=("r_multiple", lambda x: (x > 0).mean()),
        비용R=("cost_r", "mean"),
        비용bp=("cost_bp", "first"))
    return out.sort_values("기대R", ascending=False)


def sign_test(per_sym: pd.DataFrame) -> str:
    if per_sym.empty:
        return "종목 없음"
    from scipy import stats as st
    pos = int((per_sym["기대R"] > 0).sum())
    n = len(per_sym)
    p = st.binomtest(pos, n, 0.5).pvalue
    return (f"{pos}/{n} 종목 양수 ({pos/n:.0%}) · p={p:.4f} "
            f"{'유의' if p < 0.05 else '유의하지 않음'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--bars", type=int, default=8000)
    args = ap.parse_args()

    t = trader()
    groups = {"기존군(가설을 만든 데이터)": SEEN,
              "홀드아웃(처음 보는 데이터)": HOLDOUT}
    collected: dict[str, pd.DataFrame] = {}

    for label, names in groups.items():
        rows = []
        print(f"\n[{label}]", flush=True)
        for sym in names:
            try:
                tr = trades_of(sym, t, args.interval, args.bars)
            except Exception as exc:
                print(f"  {sym:14s} 실패: {type(exc).__name__}", flush=True)
                continue
            if tr.empty:
                print(f"  {sym:14s} 건너뜀 (이력·스프레드 없음)", flush=True)
                continue
            hv = int((tr["regime"] == "high_vol").sum())
            rows.append(tr)
            print(f"  {sym:14s} 거래 {len(tr):>4} (고변동 {hv:>3}) "
                  f"· {int(tr['bars'].iloc[0])}봉", flush=True)
        collected[label] = (pd.concat(rows, ignore_index=True) if rows
                            else pd.DataFrame())

    pd.set_option("display.width", 200)
    print()
    print("=" * 78)
    print(f"{TARGET} · {args.interval} · 고변동 국면 검정")
    print("=" * 78)

    for label, tr in collected.items():
        if tr.empty:
            print(f"\n[{label}] 데이터 없음")
            continue
        print(f"\n{'='*70}\n[{label}]  거래 {len(tr):,}건 "
              f"· 종목 {tr['symbol'].nunique()}개")
        for reg, tag in (("high_vol", "고변동 ← 가설"),
                         ("range", "횡보 ← 대조군")):
            s = summarise(tr, reg)
            if not s.get("거래"):
                print(f"  {tag:18s} 거래 없음")
                continue
            print(f"  {tag:18s} " + " · ".join(
                f"{k} {v}" for k, v in s.items()))
        ps = by_symbol(tr, "high_vol")
        print(f"  고변동 부호검정: {sign_test(ps)}")
        if not ps.empty:
            ps2 = ps.copy()
            ps2["비용많음"] = [s in HIGH_COST for s in ps2.index]
            print(ps2.round(4).to_string())

    print()
    print("판단 기준")
    print("  · **홀드아웃**에서 t검정 p<0.05 이고 부호검정도 유의해야 한다.")
    print("  · 고변동만 양수여야 '고변동에서 통한다'는 주장이 성립한다.")
    print("    횡보도 같이 양수면 그냥 '이 전략이 통한다'는 다른 주장이다.")
    print("  · 기존군만 좋고 홀드아웃이 나쁘면 앞선 결과는 우연이었다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
