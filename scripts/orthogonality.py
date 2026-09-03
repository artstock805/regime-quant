"""
증분 정보 검정 — 파생 특징이 캔들이 모르는 것을 아는가.

    python scripts/orthogonality.py
    python scripts/orthogonality.py --symbols BTCUSDT --horizon 4

단독 IC 비교는 잘못된 질문일 수 있다. 캔들 오실레이터들(RSI·스토캐스틱·CCI·
Williams %R·볼린저 %B)은 **사실상 같은 것**이다. 상관이 0.9 를 넘는 특징 다섯 개를
모아 놓고 "최고 IC 가 0.044" 라고 말해 봐야, 정보량은 하나짜리와 같다.

그러니 물어야 할 것은 "IC 가 더 높은가"가 아니라
**"기존 특징을 다 알고 난 뒤에도 추가로 알려 주는 게 있는가"** 이다.

재는 방법
  1) 상관 구조 — 파생 특징이 캔들 특징과 얼마나 겹치는가
  2) 부분 IC  — 캔들 최고 특징을 회귀로 걷어낸 잔차와 미래수익률의 상관
  3) 증분 R²  — 캔들만 쓴 회귀 vs 파생을 더한 회귀
  4) 유효 차원 — 상관행렬 고유값으로 센 '실질적으로 몇 개인가'

부분 IC 가 단독 IC 와 비슷하게 유지되면 그 특징은 새 정보다.
0 으로 무너지면 이미 캔들이 아는 것을 다르게 부른 것뿐이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                  # noqa: E402
from src.data.cache import get_klines                               # noqa: E402
from src.data.derivatives import align_to_bars, get_derivatives     # noqa: E402
from src.indicators import all_specs, by_category, compute_all      # noqa: E402

pd.set_option("display.width", 220)

# 가격 수준에 가까워 IC 가 자동으로 높게 나오는 열 — screen_features 와 같은 기준
LEVEL_LIKE = {
    "dc_upper", "dc_lower", "dc_mid", "kc_upper", "kc_lower",
    "bb_upper", "bb_lower", "st_line", "psar", "tenkan", "kijun",
    "obv", "atr", "momentum", "macd", "macd_signal", "awesome_osc", "kst",
}


def rank(s: pd.Series) -> pd.Series:
    """순위로 바꾼 뒤 표준화. 스피어만 상관을 선형대수로 다루기 위한 준비."""
    r = s.rank()
    return (r - r.mean()) / r.std(ddof=0)


def residualise(target: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """controls 로 target 을 회귀하고 남은 잔차. 최소제곱, 절편 포함."""
    x = np.column_stack([np.ones(len(controls)), controls.to_numpy()])
    y = target.to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.Series(y - x @ beta, index=target.index)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="증분 정보 검정")
    ap.add_argument("--symbols", nargs="*",
                    default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--n-controls", type=int, default=8,
                    help="통제로 쓸 캔들 특징 수 (IC 상위부터)")
    ap.add_argument("--deriv-dir", default=None)
    args = ap.parse_args()
    if args.deriv_dir is None:
        args.deriv_dir = cfg.path("paths.derivatives_dir")

    market = cfg.require("market")
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")

    deriv_names = by_category().get("derivatives", [])
    deriv_cols = {c for n in deriv_names for c in all_specs()[n].output_names}

    parts = []
    for sym in args.symbols:
        bars = get_klines(sym, args.interval, start, end, market=market, klines_dir=kd)
        d = get_derivatives(sym, start, end, market=market, deriv_dir=args.deriv_dir)
        bars = align_to_bars(bars, d, args.interval)
        bars = bars[bars["open_interest"].notna()]
        feats = compute_all(bars).drop(columns=list(LEVEL_LIKE), errors="ignore")
        feats["__fwd"] = bars["close"].shift(-args.horizon) / bars["close"] - 1.0
        parts.append(feats)
        print(f"  {sym:<9} {len(bars):,}봉", flush=True)

    data = pd.concat(parts).dropna()
    fwd = rank(data["__fwd"])
    cols = [c for c in data.columns if c != "__fwd"]
    kline_cols = [c for c in cols if c not in deriv_cols]
    dcols = [c for c in cols if c in deriv_cols]

    print(f"\n표본 {len(data):,}행 · 캔들 특징 {len(kline_cols)}개 · "
          f"파생 특징 {len(dcols)}개 · 지평 {args.horizon}봉\n")

    ranked = pd.DataFrame({c: rank(data[c]) for c in cols})
    ic = ranked.apply(lambda s: float(np.corrcoef(s, fwd)[0, 1]))

    controls = list(ic[kline_cols].abs().sort_values(ascending=False)
                    .head(args.n_controls).index)
    print("=" * 78)
    print(f"통제로 쓸 캔들 특징 {len(controls)}개 (단독 |IC| 상위)")
    print("=" * 78)
    print(ic[controls].round(4).to_string())

    # 통제 특징들이 서로 얼마나 같은 것인가
    cmat = ranked[controls].corr()
    off = cmat.to_numpy()[~np.eye(len(controls), dtype=bool)]
    eig = np.linalg.eigvalsh(cmat.to_numpy())
    eff = float(eig.sum() ** 2 / (eig ** 2).sum())
    print(f"\n  서로 간 평균 |상관| {np.abs(off).mean():.3f} · "
          f"최대 {np.abs(off).max():.3f}")
    print(f"  유효 차원 {eff:.2f} / {len(controls)} — "
          f"{'사실상 한 가지 정보다' if eff < 2.5 else '서로 다른 정보를 담고 있다'}")

    # ── 부분 IC
    print("\n" + "=" * 78)
    print("파생 특징의 단독 IC vs 부분 IC (캔들 상위 특징을 걷어낸 뒤)")
    print("=" * 78)
    ctrl_df = ranked[controls]
    fwd_resid = residualise(fwd, ctrl_df)
    rows = []
    for c in dcols:
        resid = residualise(ranked[c], ctrl_df)
        sd = resid.std(ddof=0)
        partial = (float(np.corrcoef(resid, fwd_resid)[0, 1]) if sd > 1e-12 else np.nan)
        with np.errstate(invalid="ignore"):
            overlap = float(np.sqrt(max(0.0, 1.0 - (sd ** 2) / max(ranked[c].var(ddof=0), 1e-12))))
        rows.append({"특징": c, "단독IC": round(float(ic[c]), 4),
                     "부분IC": None if pd.isna(partial) else round(partial, 4),
                     "캔들과겹침_R": round(overlap, 3),
                     "유지율_%": (None if pd.isna(partial) or abs(ic[c]) < 1e-9
                                  else round(abs(partial) / abs(ic[c]) * 100, 1))})
    tbl = pd.DataFrame(rows).sort_values("부분IC", key=lambda s: s.abs(), ascending=False)
    print(tbl.to_string(index=False))

    # 비교 기준 — 캔들 특징도 같은 방식으로 재면 얼마나 남는가
    krows = []
    for c in [x for x in kline_cols if x not in controls][:200]:
        resid = residualise(ranked[c], ctrl_df)
        if resid.std(ddof=0) <= 1e-12:
            continue
        krows.append(abs(float(np.corrcoef(resid, fwd_resid)[0, 1])))
    if krows:
        print(f"\n  비교 — 통제에 안 들어간 캔들 특징 {len(krows)}개의 부분 |IC|: "
              f"중위 {np.median(krows):.4f}, 최대 {max(krows):.4f}")

    # ── 증분 R²
    print("\n" + "=" * 78)
    print("증분 설명력 — 캔들만 vs 캔들+파생")
    print("=" * 78)

    def r2(x_cols: list[str]) -> float:
        x = np.column_stack([np.ones(len(data)), ranked[x_cols].to_numpy()])
        beta, *_ = np.linalg.lstsq(x, fwd.to_numpy(), rcond=None)
        pred = x @ beta
        ss_res = float(((fwd.to_numpy() - pred) ** 2).sum())
        ss_tot = float(((fwd.to_numpy() - fwd.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    r2_k = r2(controls)
    r2_kd = r2(controls + dcols)
    print(f"  캔들 {len(controls)}개        R² = {r2_k * 100:.4f}%")
    print(f"  + 파생 {len(dcols)}개        R² = {r2_kd * 100:.4f}%")
    print(f"  증분                 ΔR² = {(r2_kd - r2_k) * 100:.4f}%p")
    # 무작위 특징을 같은 수만큼 더하면 얼마나 오르는가 — 과적합 기준선
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(
        {f"__n{i}": rng.standard_normal(len(data)) for i in range(len(dcols))},
        index=data.index)
    ranked_with_noise = pd.concat([ranked[controls], noise], axis=1)
    x = np.column_stack([np.ones(len(data)), ranked_with_noise.to_numpy()])
    beta, *_ = np.linalg.lstsq(x, fwd.to_numpy(), rcond=None)
    ss_res = float(((fwd.to_numpy() - x @ beta) ** 2).sum())
    ss_tot = float(((fwd.to_numpy() - fwd.mean()) ** 2).sum())
    r2_noise = 1.0 - ss_res / ss_tot
    print(f"  무작위 {len(dcols)}개 더했을 때 ΔR² = {(r2_noise - r2_k) * 100:.4f}%p "
          f"(과적합 기준선)")
    verdict = ("파생이 실제 정보를 더한다"
               if (r2_kd - r2_k) > 3 * max(r2_noise - r2_k, 1e-12)
               else "무작위를 더한 것과 구분되지 않는다")
    print(f"  → {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
