"""
검증 스위트 — 전 과정을 한 번에 돌리고 리포트로 고정한다.

`scripts/validate.py` 가 이걸 부른다. 오케스트레이션을 스크립트에 두지 않고
라이브러리에 둔 이유는 테스트할 수 있어야 하기 때문이다.

판정 항목(check)은 다섯 개다. 통과 개수가 리포트 맨 위에 뜬다.
    통합 순기대R > 0        전 종목 거래를 합친 기대값이 0보다 유의하게 큰가
    부호 검정               종목 대부분에서 양수인가
    다중검정 보정 후 유의    Holm 보정을 통과한 종목이 하나라도 있는가
    창별 양수 과반          검증 창의 절반 이상이 양수인가
    순열 검정               신호 순서를 흩어도 같은 성과가 나오지는 않는가

기준을 못 넘으면 기준을 낮추는 게 아니라 전략을 고친다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..backtest import ExecParams, WeightParams, walk_forward
from ..backtest.walkforward import WalkForwardResult
from ..config import Config, Costs
from ..data.cache import get_klines
from ..data.derivatives import ALL_FIELDS, align_to_bars, get_derivatives
from ..indicators import compute_all
from ..regime import Regime, classify
from ..strategies import build_all
from .audit import audit_bars, audit_summary
from .efficiency import walk_forward_efficiency
from .permutation import permutation_test
from .report import ValidationReport
from .sensitivity import cost_sensitivity, parameter_plateau
from .stats import (
    benjamini_hochberg, clustered_mean_test, deflated_sharpe, gain_concentration,
    holm, sign_test, t_test, time_clusters,
)

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass
class Loaded:
    symbol: str
    bars: pd.DataFrame
    feats: pd.DataFrame
    regimes: pd.Series
    minute: pd.DataFrame | None


@dataclass
class SuiteConfig:
    symbols: list[str]
    interval: str = "1h"
    entry_threshold: float | None = None
    fee_rate: float | None = None
    # 무거운 항목들 — 기본은 켜고, 빠른 확인이 필요하면 끈다
    run_permutation: bool = True
    run_sensitivity: bool = True
    run_efficiency: bool = True
    # 파생 시장 데이터(미결제약정·펀딩·롱숏비율)를 봉에 붙일지.
    # 기본을 False 로 둔 이유: 아카이브가 2021-01 부터라 그 이전 구간이 잘린다.
    # 켜면 표본이 짧아지는 대신 파생 지표 9종이 살아난다.
    use_derivatives: bool = False
    # 부호검정·다중검정에서 한 표를 받으려면 최소 몇 거래가 있어야 하는가.
    # 거래 3건짜리 종목의 기대값 부호는 동전던지기다.
    min_trades_per_symbol: int = 30
    # 확신도를 백분위로 캘리브레이션할지. 켜면 결과가 바뀌므로 새로운 시험 한 번이다.
    calibrate_strength: bool = False
    # None 이면 config 의 paths.derivatives_dir 를 쓴다.
    # 경로를 두 곳에 적으면 언젠가 갈라진다.
    deriv_dir: str | None = None
    n_permutations: int = 200
    permutation_symbols: int = 3
    # Deflated Sharpe 에 넣을 시험 횟수 후보.
    # 이 세션에서 실제로 시험한 조합이 20개를 넘으므로 N>=50 쪽을 봐야 정직하다.
    dsr_trials: tuple[int, ...] = (8, 20, 50, 120)


def load_symbol(cfg: Config, sc: SuiteConfig, sym: str, log: Log = _noop) -> Loaded:
    """종목 하나를 메모리에 올린다."""
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")
    market = cfg.require("market")

    bars = get_klines(sym, sc.interval, start, end, market=market, klines_dir=kd)
    note = ""
    if sc.use_derivatives:
        deriv = get_derivatives(
            sym, start, end, market=market,
            deriv_dir=sc.deriv_dir or cfg.path("paths.derivatives_dir"))
        bars = align_to_bars(bars, deriv, sc.interval)
        # 파생이 없는 앞쪽 구간을 남기면 지표가 NaN 인 채로 백테스트에 들어가
        # 전략이 조용히 발언권을 잃는다. 잘라서 표본을 정직하게 만든다.
        fields = [c for c in ALL_FIELDS if c in bars.columns]
        if fields:
            have = bars[fields].notna().all(axis=1)
            kept = bars[have]
            note = (f" · 파생 {len(kept):,}봉 "
                    f"({kept.index[0]:%Y-%m} 부터)" if not kept.empty else " · 파생 없음")
            bars = kept if not kept.empty else bars
    feats = compute_all(bars)
    minute = get_klines(sym, "1m", start, end, market=market, klines_dir=kd)
    log(f"  {sym:<9} {len(bars):,}봉 로드{note}")
    return Loaded(sym, bars, feats, classify(feats)["regime"], minute)


def iter_symbols(cfg: Config, sc: SuiteConfig, log: Log = _noop):
    """
    종목을 **하나씩** 내놓는다. 전부 리스트로 들고 있으면 안 된다.

    1분봉이 종목당 메모리에서 ~200MB 다. 8종목이면 1.6GB 로 버티지만 134종목이면
    27GB 라 그대로 죽는다. 유니버스를 넓히는 순간 이 구조가 병목이 된다.
    소비하는 쪽이 필요한 것(거래 기록)만 남기고 봉은 흘려보내야 한다.
    """
    for sym in sc.symbols:
        yield load_symbol(cfg, sc, sym, log)


def load_symbols(cfg: Config, sc: SuiteConfig, log: Log = _noop) -> list[Loaded]:
    """전 종목을 한꺼번에. 종목이 적을 때만 쓴다 — 메모리 주의는 iter_symbols 참고."""
    return list(iter_symbols(cfg, sc, log))


def _params(cfg: Config) -> ExecParams:
    return ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )


def _costs(cfg: Config, sc: SuiteConfig, scale: float = 1.0) -> Costs:
    base = Costs.from_config(cfg)
    fee = base.fee_rate if sc.fee_rate is None else sc.fee_rate
    return Costs(fee_rate=fee * scale, slippage_bps=base.slippage_bps * scale)


def run_suite(
    cfg: Config, sc: SuiteConfig, log: Log = _noop
) -> tuple[ValidationReport, dict[str, WalkForwardResult]]:
    strat_names = cfg.get("strategies")
    strategies = [s for s in build_all() if not strat_names or s.name in set(strat_names)]
    params = _params(cfg)
    wp = WeightParams(
        **({} if sc.entry_threshold is None else {"entry_threshold": sc.entry_threshold}),
        calibrate_strength=sc.calibrate_strength,
    )
    costs = _costs(cfg, sc)

    report = ValidationReport(
        title=f"레짐 퀀트 검증 · {sc.interval} · {len(sc.symbols)}종목",
        config={
            **{k: cfg.get(k) for k in ("market", "strategies")},
            "interval": sc.interval,
            "period": [cfg.require("backtest.start"), cfg.require("backtest.end")],
            "atr_stop_mult": params.atr_stop_mult,
            "reward_ratio": params.reward_ratio,
            "risk_per_trade": params.risk_per_trade,
            "entry_threshold": wp.entry_threshold,
            "calibrate_strength": wp.calibrate_strength,
            "fee_rate": costs.fee_rate,
            "slippage_bps": costs.slippage_bps,
            "walk_forward": cfg.get("backtest.walk_forward"),
            "acceptance": cfg.get("acceptance"),
        },
        meta={"symbols": sc.symbols,
              "strategies": [s.korean for s in strategies]},
    )

    # ── 1~2) 감사와 walk-forward 를 종목 단위로 **함께** 돈다
    #
    # 전 종목을 리스트로 들고 있으면 안 된다. 1분봉만으로 종목당 ~200MB 라
    # 8종목은 1.6GB 로 버티지만 134종목은 27GB 라 그대로 죽는다.
    # 종목 하나를 올려 감사하고 백테스트한 뒤 봉은 버리고 결과(거래 기록)만 남긴다.
    # 뒤에서 감도·효율·순열에 쓸 앞쪽 몇 종목만 따로 붙잡아 둔다.
    log("데이터 로드 · 감사 · walk-forward")
    rows: list[dict] = []
    results: dict[str, WalkForwardResult] = {}
    frames = []
    keep: list[Loaded] = []
    n_keep = (max(sc.permutation_symbols, 3)
              if (sc.run_permutation or sc.run_sensitivity or sc.run_efficiency) else 0)

    for d in iter_symbols(cfg, sc, log):
        rows.append(audit_bars(d.bars, d.symbol, sc.interval).as_dict())
        if d.minute is not None:
            rows.append(audit_bars(d.minute, d.symbol, "1m").as_dict())

        wf = walk_forward(d.bars, d.feats, d.regimes, strategies, cfg,
                          interval=sc.interval, minute_df=d.minute, symbol=d.symbol,
                          weight_params=wp, exec_params=params, costs=costs)
        results[d.symbol] = wf
        if not wf.oos_trades.empty:
            frames.append(wf.oos_trades)
        m = wf.metrics
        log(f"    거래 {m.trades:>4}  수익률 {m.total_return*100:+7.2f}%  "
            f"샤프 {m.sharpe:+.2f}  MDD {m.max_drawdown*100:5.2f}%  PF {m.profit_factor:.2f}")

        if len(keep) < n_keep:
            keep.append(d)
        # 나머지 종목의 봉은 여기서 참조가 끊긴다 — 다음 반복에서 회수된다

    loaded = keep
    audit_tbl = pd.DataFrame(rows)
    summary = audit_summary(audit_tbl)
    report.add("데이터 감사", {**summary, "rows": rows})
    log("\n[1/7] 데이터 품질 감사")
    log(f"  {summary['total_bars']:,}봉 · 누락 {summary['total_missing']:,} · "
        f"이상치 {summary['total_outliers']} · 최소 커버리지 {summary['min_coverage']:.4f}")
    if not summary["ok"]:
        log(f"  ⚠ 구조적 문제: {', '.join(summary['problems'])}")
    log(f"\n[2/7] walk-forward — {len(results)}종목 완료")

    if not frames:
        report.add("통합 검정", {"error": "거래가 없습니다"})
        report.check("통합 순기대R > 0", False)
        return report, results

    trades = pd.concat(frames, ignore_index=True)
    per_symbol = {s: r.oos_trades["r_multiple"].to_numpy()
                  for s, r in results.items() if not r.oos_trades.empty}

    report.add("종목별 성적", [
        {"종목": s, "거래": r.metrics.trades,
         "수익률_%": round(r.metrics.total_return * 100, 2),
         "샤프": round(r.metrics.sharpe, 2),
         "MDD_%": round(r.metrics.max_drawdown * 100, 2),
         "PF": round(r.metrics.profit_factor, 3) if np.isfinite(r.metrics.profit_factor) else None,
         "승률_%": round(r.metrics.win_rate * 100, 1)}
        for s, r in results.items()
    ])

    # ── 3) 통합 검정 (주 증거)
    log("\n[3/7] 통합 검정")
    net = t_test(trades["r_multiple"].to_numpy())
    gross = t_test(trades["gross_r"].to_numpy())
    cost_r = float(trades["cost_r"].mean())
    report.add("통합 검정", {
        "trades": net.n,
        "net_mean": net.mean, "net_t": net.t, "net_p": net.p,
        "net_ci_lo": net.ci_lo, "net_ci_hi": net.ci_hi, "net_verdict": net.verdict,
        "gross_mean": gross.mean, "gross_t": gross.t, "gross_p": gross.p,
        "gross_verdict": gross.verdict,
        "cost_r": cost_r,
        "cost_share_of_edge": (cost_r / gross.mean) if gross.mean > 0 else None,
        "trades_needed_for_significance": net.trades_needed(),
    })
    report.check("통합 순기대R > 0", net.significant)
    log(f"  순  {net.mean:+.4f}R  t={net.t:+.2f}  p={net.p:.4f}  → {net.verdict}")
    log(f"  총  {gross.mean:+.4f}R  t={gross.t:+.2f}  p={gross.p:.4f}  → {gross.verdict}")
    if gross.mean > 0:
        share = cost_r / gross.mean
        if share >= 1.0:
            log(f"  비용 {cost_r:.4f}R > 총우위 {gross.mean:.4f}R — "
                f"우위를 다 먹고 {(share - 1) * 100:.0f}% 더 나간다")
        else:
            log(f"  비용 {cost_r:.4f}R 이 총우위의 {share * 100:.0f}% 를 먹는다 "
                f"(남는 것 {gross.mean - cost_r:+.4f}R)")
    if net.trades_needed():
        log(f"  유의해지려면 {net.trades_needed():,} 거래 필요 (현재 {net.n:,})")

    # ── 3b) 군집 강건 검정 — 종목을 늘려 만든 가짜 유의성을 막는다
    #
    # 순진한 t검정은 거래 하나하나를 독립 관측으로 센다. 암호화폐는 종목 간 상관이
    # 매우 높아서, 같은 주에 여러 종목에서 열린 거래는 사실상 같은 시장 충격을
    # 여러 번 센 것이다. 유니버스를 8→134 종목으로 늘리면 거래 수가 16배가 되고
    # 순진한 t 는 √16 배로 뛰는데, 그 상승분의 대부분은 정보가 아니라 중복이다.
    # **표본을 늘려 유의성을 얻었다고 착각하는 것을 여기서 막는다.**
    clusters = time_clusters(trades["entry_time"], "W")
    net_cl = clustered_mean_test(trades["r_multiple"].to_numpy(), clusters)
    gross_cl = clustered_mean_test(trades["gross_r"].to_numpy(), clusters)
    report.add("군집 강건 검정", {
        "cluster_freq": "주",
        "n_clusters": net_cl.n_clusters,
        "net": net_cl.as_dict(), "gross": gross_cl.as_dict(),
    })
    report.check("군집 보정 후 순기대R > 0",
                 bool(np.isfinite(net_cl.t) and net_cl.t > 0 and net_cl.p < 0.05))
    log(f"  [군집 {net_cl.n_clusters}주] 순 t {net_cl.t_naive:+.2f} → {net_cl.t:+.2f} "
        f"(p={net_cl.p:.4f}) · SE 팽창 {net_cl.inflation:.2f}배")
    log(f"  [군집 {net_cl.n_clusters}주] 총 t {gross_cl.t_naive:+.2f} → {gross_cl.t:+.2f} "
        f"(p={gross_cl.p:.4f})")
    if np.isfinite(net_cl.effective_n):
        log(f"  유효표본 {net_cl.effective_n:,.0f} / 실제 {net_cl.n:,} "
            f"— 거래의 {net_cl.effective_n / net_cl.n * 100:.0f}% 만이 독립 정보다")

    # ── 4) 부호 검정 + 다중검정 보정
    #
    # 거래가 몇 건뿐인 종목에 한 표를 주면 안 된다. 유니버스를 134종목으로 넓히면
    # 상장폐지로 몇 달치만 남은 종목이 많아지는데, 거래 3건짜리 종목의 기대값 부호는
    # 동전던지기다. 그런 표가 절반을 넘으면 부호검정은 잡음을 세는 꼴이 된다.
    log("\n[4/7] 부호 검정 · 다중검정 보정")
    eligible = {s: r for s, r in per_symbol.items() if len(r) >= sc.min_trades_per_symbol}
    dropped = len(per_symbol) - len(eligible)
    if not eligible:                      # 전부 미달이면 기준을 접고 전체를 쓴다
        eligible, dropped = per_symbol, 0
    signs = sign_test({s: float(r.mean()) for s, r in eligible.items()})
    signs["min_trades"] = sc.min_trades_per_symbol
    signs["symbols_dropped"] = dropped
    report.add("부호 검정", signs)
    report.check("부호 검정", signs["significant"])
    log(f"  양수 {signs['positive']}/{signs['n']} 종목 · p={signs['p']:.4f}"
        + (f"  (거래 {sc.min_trades_per_symbol}건 미만 {dropped}종목 제외)"
           if dropped else ""))

    stats = {s: t_test(r) for s, r in eligible.items()}
    pvals = {s: st.p for s, st in stats.items()}
    hp, bh = holm(pvals), benjamini_hochberg(pvals)
    sig_syms = [s for s, st in stats.items() if st.t > 0 and hp[s] < 0.05]
    report.add("종목별 검정", [
        {"종목": s, "거래": st.n, "기대R": round(st.mean, 5), "t": round(st.t, 3),
         "원p": round(st.p, 4), "Holm_p": round(hp[s], 4), "BH_p": round(bh[s], 4),
         "Holm유의": s in sig_syms}
        for s, st in sorted(stats.items(), key=lambda kv: -kv[1].t)
    ])
    report.check("다중검정 보정 후 유의", bool(sig_syms))
    log(f"  Holm 보정 후 유의한 종목: {len(sig_syms)}개")

    # ── 5) 창별 일관성
    log("\n[5/7] 창별 일관성")
    windows = next(iter(results.values())).windows
    wrows = []
    for i, w in enumerate(windows):
        block = trades[(trades["entry_time"] >= w.test_start)
                       & (trades["entry_time"] < w.test_end)]
        if block.empty:
            continue
        wrows.append({"창": i, "기간": f"{w.test_start:%Y-%m}~{w.test_end:%Y-%m}",
                      "거래": len(block),
                      "합계R": round(float(block["r_multiple"].sum()), 3),
                      "기대R": round(float(block["r_multiple"].mean()), 5)})
    sums = np.array([r["합계R"] for r in wrows]) if wrows else np.array([])
    pos_w = int((sums > 0).sum())
    conc = gain_concentration(sums)
    report.add("창별 일관성", {
        "windows": len(wrows), "positive": pos_w,
        "gain_concentration": conc,
        "rows": wrows,
    })
    report.check("창별 양수 과반", bool(len(wrows) and pos_w > len(wrows) / 2))
    log(f"  양수 {pos_w}/{len(wrows)}창 · 이익의 {conc*100:.0f}%가 최고 창 하나에서"
        if np.isfinite(conc) else f"  양수 {pos_w}/{len(wrows)}창")

    # ── 6) Deflated Sharpe
    log("\n[6/7] Deflated Sharpe")
    trial_sr = np.array([st.mean / st.sd for st in stats.values()
                         if np.isfinite(st.sd) and st.sd > 0])
    best = max(stats, key=lambda s: stats[s].t)
    dsr_rows = [deflated_sharpe(per_symbol[best], n, trial_sr).as_dict()
                for n in sc.dsr_trials]
    report.add("Deflated Sharpe", {"best_symbol": best, "rows": dsr_rows})
    for row in dsr_rows:
        log(f"  시험 {row['n_trials']:>4}회 → DSR {row['dsr']:.3f}"
            if row["dsr"] is not None else f"  시험 {row['n_trials']:>4}회 → 판정불가")

    # ── 7) 순열 검정
    if sc.run_permutation:
        log(f"\n[7/7] 순열 검정 ({sc.n_permutations}회 x {sc.permutation_symbols}종목)")
        perm_rows = []
        for d in loaded[: sc.permutation_symbols]:
            wf = results[d.symbol]
            if wf.oos_trades.empty:
                continue
            sig = _ensemble_signal(d, strategies, wf, wp)
            if sig is None:
                continue
            r = permutation_test(
                d.bars, sig, d.feats["atr"], interval=sc.interval, costs=costs,
                params=params, initial_equity=float(cfg.require("backtest.initial_equity")),
                minute_df=d.minute, regimes=d.regimes,
                n_permutations=sc.n_permutations, seed=0,
                progress=lambda k, n, s=d.symbol: log(f"    {s} {k}/{n}"),
            )
            perm_rows.append({"종목": d.symbol, **{
                k: v for k, v in r.as_dict().items() if k != "null_samples"}})
            log(f"  {d.symbol:<9} 관측 {r.observed:+.4f} vs 귀무 {r.null_mean:+.4f} "
                f"· p={r.p_value:.4f} · {r.verdict}")
        report.add("순열 검정", perm_rows)
        report.check("순열 검정",
                     bool(perm_rows) and all(r["significant"] for r in perm_rows))
    else:
        log("\n[7/7] 순열 검정 — 건너뜀")

    # ── 부가: 민감도 · 효율
    if sc.run_sensitivity:
        log("\n[부가] 민감도")
        _add_sensitivity(report, cfg, sc, loaded[0], strategies, params, wp, log)
    if sc.run_efficiency:
        log("\n[부가] 워크포워드 효율")
        _add_efficiency(report, cfg, sc, loaded, strategies, params, wp, costs, log)

    log(f"\n{'=' * 70}\n{report.verdict}")
    for k, v in report.checks.items():
        log(f"  [{'O' if v else 'X'}] {k}")
    if report.passed < len(report.checks):
        log("  → 기준을 낮추는 게 아니라 전략을 고쳐야 한다.")
    report.trades = trades
    return report, results


def _ensemble_signal(d: Loaded, strategies, wf: WalkForwardResult, wp: WeightParams):
    """
    순열 검정에 넣을 앙상블 신호. 창별 가중치를 각 검증 구간에 적용해 이어붙인다.
    가중치가 학습 구간에서 나왔다는 성질을 유지해야 검정이 의미를 갖는다.
    """
    from ..backtest.ensemble import combine

    pieces = []
    for k, w in enumerate(wf.windows):
        weights = (wf.weights_per_window[k]
                   if k < len(wf.weights_per_window) else None)
        if weights is None:
            continue
        mask = (d.bars.index >= w.test_start) & (d.bars.index < w.test_end)
        if not mask.any():
            continue
        bars, feats = d.bars[mask], d.feats[mask]
        sigs = {}
        for s in strategies:
            sig = s.generate(bars, feats).copy()
            allowed = d.regimes.reindex(bars.index).isin([str(r) for r in s.regimes])
            sig.loc[~allowed, ["direction", "strength"]] = 0.0
            sigs[s.name] = sig
        pieces.append(combine(sigs, d.regimes, weights, wp)[["direction", "strength"]])
    if not pieces:
        return None
    out = pd.concat(pieces).sort_index()
    # 검증 구간만 신호가 있고 나머지는 0 — 전체 인덱스에 맞춘다
    return out.reindex(d.bars.index).fillna(0.0)


def _add_sensitivity(report, cfg, sc, d: Loaded, strategies, params, wp, log: Log) -> None:
    from ..backtest.walkforward import run_standalone

    def trades_for(stop_mult: float, cost_scale: float = 1.0) -> pd.DataFrame:
        from dataclasses import replace
        res = run_standalone(
            d.bars, d.feats, strategies, interval=sc.interval,
            costs=_costs(cfg, sc, cost_scale),
            params=replace(params, atr_stop_mult=stop_mult),
            initial_equity=float(cfg.require("backtest.initial_equity")),
            minute_df=d.minute, regimes=d.regimes, symbol=d.symbol,
        )
        return pd.concat([r.trades for r in res.values()], ignore_index=True)

    plateau = parameter_plateau(
        lambda v: trades_for(v), "atr_stop_mult",
        [1.5, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0], chosen=params.atr_stop_mult,
    )
    report.add("파라미터 고원", plateau)
    log(f"  손절폭 고원 폭 {plateau.plateau_width}/{plateau.n_points} · {plateau.verdict}")

    cost = cost_sensitivity(lambda m: trades_for(params.atr_stop_mult, m))
    report.add("비용 민감도", cost)
    log(f"  {cost.verdict}")


def _add_efficiency(report, cfg, sc, loaded, strategies, params, wp, costs, log: Log) -> None:
    pooled_is, pooled_oos, rows = [], [], []
    for d in loaded[:3]:
        eff = walk_forward_efficiency(
            d.bars, d.feats, d.regimes, strategies, cfg, interval=sc.interval,
            minute_df=d.minute, costs=costs, exec_params=params, weight_params=wp,
        )
        rows.append({"종목": d.symbol, "창": eff.windows,
                     "학습기대R": round(eff.is_expectancy, 5),
                     "검증기대R": round(eff.oos_expectancy, 5),
                     "WFE": round(eff.wfe, 3) if np.isfinite(eff.wfe) else None,
                     "판정": eff.verdict})
        if np.isfinite(eff.is_expectancy):
            pooled_is.append(eff.is_expectancy)
        if np.isfinite(eff.oos_expectancy):
            pooled_oos.append(eff.oos_expectancy)
        log(f"  {d.symbol:<9} 학습 {eff.is_expectancy:+.4f} → 검증 {eff.oos_expectancy:+.4f} "
            f"· WFE {eff.wfe:+.2f} · {eff.verdict}")

    is_m = float(np.mean(pooled_is)) if pooled_is else float("nan")
    oos_m = float(np.mean(pooled_oos)) if pooled_oos else float("nan")
    report.add("워크포워드 효율", {
        "rows": rows,
        "pooled_is": is_m, "pooled_oos": oos_m,
        "pooled_wfe": (oos_m / is_m) if (np.isfinite(is_m) and abs(is_m) > 1e-9) else None,
    })
