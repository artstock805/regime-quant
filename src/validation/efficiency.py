"""
워크포워드 효율(WFE) — 학습 성과가 검증 구간에서 얼마나 살아남는가.

    WFE = 검증 구간 기대R / 학습 구간 기대R

walk-forward 는 검증 성과만 보고한다. 그런데 **학습 성과와의 비율**이 과최적화를
가장 직접적으로 드러낸다.

    WFE ≈ 1.0   학습에서 본 것이 검증에서 그대로 나온다 — 구조를 잡았다
    WFE ≈ 0.5   절반이 사라졌다 — 흔한 수준. 경계.
    WFE ≈ 0     학습 성과가 전부 사라졌다 — 그 성과는 잡음에 맞춘 것이었다
    WFE < 0     부호가 뒤집혔다 — 학습에서 배운 것이 검증에서 해가 된다

같은 가중치를 학습 구간과 검증 구간에 **각각** 적용해 비교한다. 학습 구간 성과는
그 데이터로 가중치를 뽑았으므로 정의상 in-sample 이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..backtest.engine import ExecParams, run
from ..backtest.ensemble import WeightParams, combine, derive_weights, score_by_regime
from ..backtest.walkforward import Window, make_windows, run_standalone
from ..config import Config, Costs
from ..regime import Regime
from ..strategies.base import Strategy


@dataclass(frozen=True)
class EfficiencyResult:
    windows: int
    is_expectancy: float
    oos_expectancy: float
    wfe: float
    per_window: list[dict] = field(default_factory=list)
    positive_is: int = 0
    positive_oos: int = 0

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.wfe):
            return "판정불가 — 학습 기대값이 0 근처"
        if self.wfe < 0:
            return "부호 뒤집힘 — 학습에서 배운 것이 검증에서 해가 됐다"
        if self.wfe < 0.3:
            return "학습 성과가 대부분 사라졌다 — 과최적화"
        if self.wfe < 0.6:
            return "절반가량 사라졌다 — 경계"
        return "학습 성과가 검증에서도 유지된다"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


def _slice(obj, lo, hi):
    return obj.loc[(obj.index >= lo) & (obj.index < hi)]


def _ensemble_trades(
    bars, feats, strategies, regimes, weights, wp, *,
    interval, costs, params, initial_equity, minute_df, gate=True,
) -> pd.DataFrame:
    """주어진 가중치로 앙상블을 돌려 거래 표를 얻는다."""
    if bars.empty:
        return pd.DataFrame()
    sigs = {}
    for s in strategies:
        sig = s.generate(bars, feats).copy()
        if gate:
            allowed = regimes.reindex(bars.index).isin([str(r) for r in s.regimes])
            sig.loc[~allowed, ["direction", "strength"]] = 0.0
        sigs[s.name] = sig
    ens = combine(sigs, regimes, weights, wp)
    res = run(bars, ens[["direction", "strength"]], feats["atr"],
              interval=interval, costs=costs, params=params,
              initial_equity=initial_equity, minute_df=minute_df, regimes=regimes)
    return res.trades


def walk_forward_efficiency(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    regimes: pd.Series,
    strategies: list[Strategy],
    cfg: Config,
    *,
    interval: str,
    minute_df: pd.DataFrame | None = None,
    costs: Costs | None = None,
    exec_params: ExecParams | None = None,
    weight_params: WeightParams | None = None,
    start=None,
    end=None,
    gate: bool = True,
    metric: str = "r_multiple",
    progress=None,
) -> EfficiencyResult:
    costs = costs or Costs.from_config(cfg)
    params = exec_params or ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    wp = weight_params or WeightParams()
    initial = float(cfg.require("backtest.initial_equity"))

    windows: list[Window] = make_windows(
        start if start is not None else cfg.require("backtest.start"),
        end if end is not None else cfg.require("backtest.end"),
        int(cfg.require("backtest.walk_forward.train_months")),
        int(cfg.require("backtest.walk_forward.test_months")),
    )
    declared = ({s.name: s.regimes for s in strategies} if gate
                else {s.name: tuple(Regime) for s in strategies})

    rows, is_all, oos_all = [], [], []
    for k, w in enumerate(windows):
        tr_df = _slice(df, w.train_start, w.train_end)
        te_df = _slice(df, w.test_start, w.test_end)
        if len(tr_df) < 500 or te_df.empty:
            if progress:
                progress(k + 1, len(windows))
            continue
        tr_f, te_f = _slice(feats, w.train_start, w.train_end), _slice(feats, w.test_start, w.test_end)
        tr_m = _slice(minute_df, w.train_start, w.train_end) if minute_df is not None else None
        te_m = _slice(minute_df, w.test_start, w.test_end) if minute_df is not None else None

        # 학습 구간에서 가중치를 뽑는다 (walk_forward 와 동일한 경로)
        standalone = run_standalone(
            tr_df, tr_f, strategies, interval=interval, costs=costs, params=params,
            initial_equity=initial, minute_df=tr_m, regimes=regimes, gate=gate,
        )
        all_tr = pd.concat([r.trades for r in standalone.values()], ignore_index=True)
        weights = derive_weights(score_by_regime(all_tr, wp), declared, wp)

        kw = dict(interval=interval, costs=costs, params=params,
                  initial_equity=initial, gate=gate)
        is_tr = _ensemble_trades(tr_df, tr_f, strategies, regimes, weights, wp,
                                 minute_df=tr_m, **kw)
        oos_tr = _ensemble_trades(te_df, te_f, strategies, regimes, weights, wp,
                                  minute_df=te_m, **kw)

        is_e = float(is_tr[metric].mean()) if not is_tr.empty else np.nan
        oos_e = float(oos_tr[metric].mean()) if not oos_tr.empty else np.nan
        if not is_tr.empty:
            is_all.append(is_tr[metric].to_numpy())
        if not oos_tr.empty:
            oos_all.append(oos_tr[metric].to_numpy())

        rows.append({
            "창": k, "기간": f"{w.test_start:%Y-%m}~{w.test_end:%Y-%m}",
            "학습거래": len(is_tr), "학습기대R": None if np.isnan(is_e) else round(is_e, 5),
            "검증거래": len(oos_tr), "검증기대R": None if np.isnan(oos_e) else round(oos_e, 5),
            "WFE": (None if not (np.isfinite(is_e) and np.isfinite(oos_e) and abs(is_e) > 1e-9)
                    else round(oos_e / is_e, 3)),
        })
        if progress:
            progress(k + 1, len(windows))

    is_pool = np.concatenate(is_all) if is_all else np.array([])
    oos_pool = np.concatenate(oos_all) if oos_all else np.array([])
    is_e = float(is_pool.mean()) if is_pool.size else float("nan")
    oos_e = float(oos_pool.mean()) if oos_pool.size else float("nan")
    wfe = (oos_e / is_e) if (np.isfinite(is_e) and abs(is_e) > 1e-9) else float("nan")

    return EfficiencyResult(
        windows=len(rows), is_expectancy=is_e, oos_expectancy=oos_e, wfe=wfe,
        per_window=rows,
        positive_is=sum(1 for r in rows if (r["학습기대R"] or 0) > 0),
        positive_oos=sum(1 for r in rows if (r["검증기대R"] or 0) > 0),
    )
