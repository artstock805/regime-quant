"""
코인 제외 선물성 종목(금·지수·에너지)에서 기존 전략이 사는지 재는 스크립트.

**왜 이걸 먼저 재는가.** 이 프로젝트에서 전략을 죽인 것은 언제나 비용이었다.
코인 총우위 t=3.5 가 비용을 빼면 t=1.4 였고, FX 는 우위 자체가 없었다
(config 주석: "어떤 값도 이 신호를 수익성 있게 만들지 못한다").

그런데 실측해 보니 비용이 훨씬 싼 시장이 있다 — 손절 4×ATR 기준 왕복비용이
1R 대비:

    GOLD                1.0%      ← FX 의 1/11
    US100Cash JP225Cash 1.5%
    GER40Cash US500Cash 2.2~2.3%
    US30Cash            2.5%
    FX (기준)           10.9%
    NGASCash            33.6%     ← 거래 불가

같은 전략이라도 비용이 1/10 이면 결과가 달라질 수 있다. **엔진을 확장하기
전에 그것부터 확인한다** — 재 보지도 않고 다중자산 엔진을 만들면, 안 되는
것을 위해 큰 구조를 짓는 셈이다.

**FX 심볼 레이어를 타지 않는다.** `symbols.parse()` 는 FX 가 아니면 None 을
주므로 금·지수는 통과하지 못한다. 여기서는 봉 → 지표 → 국면 → 단독 백테스트만
직접 엮어서, 엔진을 건드리지 않고 답을 얻는다.

사용법
    python scripts/probe_nonfx.py
    python scripts/probe_nonfx.py --symbols GOLD,US500Cash --months 6
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
from src.backtest.ensemble import score_by_regime                   # noqa: E402
from src.backtest.walkforward import run_standalone                 # noqa: E402
from src.config import Costs                                        # noqa: E402
from src.fx.engine import trader                                    # noqa: E402
from src.fx.mt5 import bars, symbol_info                            # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.regime import classify                                     # noqa: E402

# 비용이 FX(10.9%)보다 확실히 싼 것들만. NGASCash(33.6%)·XPDUSD(25.2%)·
# XPTUSD(16.9%) 는 비용이 우위를 먹을 것이 뻔하므로 아예 넣지 않는다.
# US400Cash 는 중위 0.0% 인데 90분위가 24.7% 로 튀어 유동성을 의심한다 — 제외.
DEFAULT = ("GOLD", "US100Cash", "JP225Cash", "XAUEUR", "XAUJPY",
           "GER40Cash", "US500Cash", "US30Cash", "FRA40Cash",
           "OILCash", "BRENTCash", "SILVER")

# 비교군. **같은 조건에서 나란히 재야** "비용이 싼 쪽이 정말 나은가"를 말할 수
# 있다. 따로 잰 숫자를 나중에 나란히 놓으면 봉 수도 기간도 달라 비교가 안 된다.
FX_GROUP = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
            "NZDUSD")

# 종목 성격별로 묶어 본다. 금·지수·에너지가 다르게 나오면 그 자체가 정보다.
FAMILY = {
    "GOLD": "금", "XAUEUR": "금", "XAUJPY": "금", "SILVER": "은",
    "OILCash": "에너지", "BRENTCash": "에너지",
}


def half_spread(df: pd.DataFrame, si) -> tuple[float, float, float]:
    """
    편도 비용(비율)과 진단값. (편도, 전체0제외bp, 최근bp) 를 돌려준다.

    **spread==0 인 봉을 반드시 빼야 한다.** MT5 옛 이력에는 스프레드가 기록돼
    있지 않고 0 으로 온다. 그건 "비용 없음"이 아니라 **"모름"** 이다. 안 빼면
    중위값이 0 이 되어 비용을 공짜로 넣게 된다.

    실측(4시간봉 17,000봉): OILCash 68.0% · BRENTCash 60.9% ·
    GER40Cash 55.5% · XAUEUR 61.0% 가 0 이었다. 그 탓에 앞선 측정에서
    "에너지가 유일하게 기대R 양수(+0.028)" 라는 **거짓 결과**가 나왔다 —
    실제 비용은 4~5bp 다. GOLD·EURUSD·SILVER 는 0 이 0.1% 미만이라 무사했다.

    전 구간에 하나의 비용을 적용하는 구조이므로 **0 을 뺀 전 구간 중위**를
    쓴다. 최근 구간이 더 싼 종목도 있지만(SILVER 15.1→7.8bp) 그쪽을 쓰면
    과거 수익에 오늘 비용을 붙이는 셈이라 결과가 유리해진다.
    """
    if "spread" not in df.columns:
        return 0.0, 0.0, 0.0
    rel = (df["spread"] * si.point / df["close"]).replace(
        [np.inf, -np.inf], np.nan)
    nz = rel[(df["spread"] > 0) & rel.notna()]
    if not len(nz):
        return 0.0, 0.0, 0.0
    full = float(nz.median())
    tail = nz.tail(2000)
    recent = float(tail.median()) if len(tail) else full
    return full / 2.0, full * 1e4, recent * 1e4


def measure(sym: str, t, interval: str, nbars: int) -> pd.DataFrame:
    df = bars(sym, interval, nbars)
    if len(df) < 1500:
        print(f"  {sym} 이력 부족 ({len(df)}봉)", flush=True)
        return pd.DataFrame()
    feats = compute_all(df)
    si = symbol_info(sym)
    half, cost_bp, recent_bp = half_spread(df, si)
    if half <= 0:
        print(f"  {sym} 스프레드 기록이 없어 비용을 알 수 없다 — 건너뜀",
              flush=True)
        return pd.DataFrame()
    zero_share = float((df["spread"] == 0).mean())
    regs = classify(feats)["regime"]

    res = run_standalone(df, feats, t.strategies(), interval=interval,
                         costs=Costs(fee_rate=half, slippage_bps=0.5),
                         params=ExecParams(), initial_equity=10_000,
                         minute_df=None, regimes=regs, symbol=sym)
    frames = [r.trades for r in res.values() if not r.trades.empty]
    if not frames:
        print(f"  {sym} 거래 없음", flush=True)
        return pd.DataFrame()
    sc = score_by_regime(pd.concat(frames, ignore_index=True),
                         t.weight_params())
    sc["symbol"] = sym
    sc["cost_bp"] = round(cost_bp, 2)
    sc["recent_bp"] = round(recent_bp, 2)
    sc["zero_share"] = round(zero_share, 3)
    sc["bars"] = len(df)
    sc["family"] = FAMILY.get(sym, "지수" if "Cash" in sym else "FX")
    per_year = {"1h": 24 * 365, "4h": 6 * 365, "1d": 252}.get(
        interval, 365)
    yrs = len(df) / per_year
    print(f"  {sym} 완료 · {len(df)}봉({yrs:.1f}년) · "
          f"왕복 {cost_bp:.2f}bp(최근 {recent_bp:.2f}) · "
          f"spread=0 {zero_share:.0%}", flush=True)
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT))
    ap.add_argument("--interval", default="1h")
    # MT5 는 요청하면 서버에서 더 받아온다. 6000 만 요청했다가 "이력이 250일뿐"
    # 이라고 잘못 결론 낸 적이 있다 — 1시간봉 2.3년, 4시간봉 7.9년이 나온다.
    ap.add_argument("--bars", type=int, default=20000)
    ap.add_argument("--with-fx", action="store_true",
                    help="FX 메이저를 같은 조건에서 같이 재서 비교한다")
    args = ap.parse_args()

    t = trader()
    wp = t.weight_params()
    declared = {s.name: {str(r) for r in s.regimes} for s in t.strategies()}

    names = [x.strip() for x in args.symbols.split(",") if x.strip()]
    if args.with_fx:
        names += list(FX_GROUP)

    frames = []
    for sym in names:
        try:
            r = measure(sym, t, args.interval, args.bars)
            if not r.empty:
                frames.append(r)
        except Exception as exc:
            print(f"  {sym} 실패: {type(exc).__name__}: {exc}", flush=True)
    if not frames:
        return 1

    all_ = pd.concat(frames, ignore_index=True)
    # 전략이 선언하지 않은 국면은 후보가 아니다 — 세면 성적이 부풀려진다.
    all_ = all_[[r in declared.get(s, set())
                 for s, r in zip(all_["strategy"], all_["regime"])]]

    pd.set_option("display.width", 200)
    print()
    print("=" * 78)
    print(f"코인 제외 종목 · 기존 {len(declared)}개 전략 · min_trades="
          f"{wp.min_trades}")
    print("=" * 78)

    print("\n── 종목별 요약 ──────────────────────────────")
    g = all_.groupby("symbol").agg(
        칸=("strategy", "count"),
        자격=("eligible", "sum"),
        거래=("trades", "sum"),
        기대R평균=("expectancy_r", "mean"),
        기대R최고=("expectancy_r", "max"),
        비용bp=("cost_bp", "first"))
    print(g.sort_values("자격", ascending=False).round(4).to_string())

    print("\n── 자격을 얻은 칸 (기대R>0 이고 거래 충분) ──────────────")
    ok = all_[all_["eligible"]].sort_values("expectancy_r", ascending=False)
    cols = ["symbol", "regime", "strategy", "trades", "expectancy_r",
            "win_rate", "cost_bp"]
    if ok.empty:
        print("  없음")
    else:
        print(ok[cols].round(4).to_string(index=False))

    print("\n── 성격별 (비교가 요점이다) ──")
    fam = all_.groupby("family").agg(
        종목=("symbol", "nunique"), 칸=("strategy", "count"),
        자격=("eligible", "sum"), 거래=("trades", "sum"),
        기대R평균=("expectancy_r", "mean"),
        기대R중위=("expectancy_r", "median"),
        양수비율=("expectancy_r", lambda x: (x > 0).mean()),
        비용bp=("cost_bp", "mean"), 최근bp=("recent_bp", "mean"))
    fam["자격비율"] = (fam["자격"] / fam["칸"]).round(3)
    print(fam.sort_values("기대R평균", ascending=False).round(4).to_string())

    print("\n── 국면별 (전 종목) ──────────────────────────────")
    r = all_.groupby("regime").agg(
        칸=("strategy", "count"), 자격=("eligible", "sum"),
        거래=("trades", "sum"), 기대R중위=("expectancy_r", "median"))
    print(r.round(4).to_string())

    print()
    tot = all_["expectancy_r"]
    print(f"전체 {len(all_)}칸 · 기대R 양수 {int((tot>0).sum())}칸 "
          f"({(tot>0).mean():.0%}) · 자격 {int(all_['eligible'].sum())}칸")
    print(f"기대R 평균 {tot.mean():+.4f} · 중위 {tot.median():+.4f}")
    print()
    print("판단: 양수비율이 50% 를 넘어야 계통적 우위다. 50% 미만이면 비용이 이긴다.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
