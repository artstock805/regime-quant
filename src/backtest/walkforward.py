"""
Walk-forward 검증.

    [────── 학습 6개월 ──────][─ 검증 2개월 ─]
              [────── 학습 6개월 ──────][─ 검증 2개월 ─]
                        [────── 학습 6개월 ──────][─ 검증 2개월 ─]

학습 구간에서 전략별 레짐 성과를 뽑아 가중치를 정하고, **그 가중치를 한 번도 보지
않은 검증 구간**에 적용한다. 검증 구간들을 이어붙인 것이 out-of-sample 성적이고,
합격 판정은 이 숫자로만 한다.

전 구간을 한 번에 돌려 얻은 수익률은 참고도 되지 않는다. 가중치가 이미 그 구간의
정답을 알고 있기 때문이다.

**종목별로 독립 백테스트를 한다.** 각 종목에 전액을 굴린 결과이므로, 종목 간 자본
배분과 동시 포지션 제한이 들어간 포트폴리오 성적이 아니다. 그건 4단계 과제다.
여기서 답해야 하는 질문은 "이 방식에 우위가 있는가"이고, 그건 종목별로 봐야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace

import numpy as np
import pandas as pd

from ..config import Config, Costs
from ..regime import Regime
from ..strategies.base import Strategy
from .engine import BacktestResult, ExecParams, run
from .ensemble import WeightParams, combine, derive_weights, score_by_regime
from .metrics import Metrics, compute_metrics


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"학습 {self.train_start:%Y-%m-%d}~{self.train_end:%Y-%m-%d} "
                f"→ 검증 {self.test_start:%Y-%m-%d}~{self.test_end:%Y-%m-%d}")


def make_windows(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    train_months: int,
    test_months: int,
) -> list[Window]:
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    windows: list[Window] = []

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append(Window(train_start, train_end, train_end, test_end))
        train_start = train_start + pd.DateOffset(months=test_months)
    return windows


@dataclass
class WalkForwardResult:
    symbol: str
    oos_trades: pd.DataFrame
    oos_equity: pd.Series
    metrics: Metrics
    initial_equity: float = 0.0
    weights_per_window: list[pd.DataFrame] = field(default_factory=list)
    reward_per_window: list[dict] = field(default_factory=list)
    train_scores: list[pd.DataFrame] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    standalone: dict[str, Metrics] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"WalkForward({self.symbol}: {self.metrics.trades}거래 "
                f"{self.metrics.total_return*100:+.1f}% 샤프 {self.metrics.sharpe:.2f})")


def _slice(obj, lo, hi):
    return obj.loc[(obj.index >= lo) & (obj.index < hi)]


def run_standalone(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    strategies: list[Strategy],
    *,
    interval: str,
    costs: Costs,
    params: ExecParams,
    initial_equity: float,
    minute_df: pd.DataFrame | None,
    regimes: pd.Series,
    symbol: str = "",
    gate: bool = True,
) -> dict[str, BacktestResult]:
    """
    각 전략을 단독으로 돌린다. 레짐별 성과(=리더보드)의 원재료.

    gate=True 면 전략이 선언한 국면 밖에서 신호를 무효화한다. 우연히 다른 국면에서
    잘 나온 것을 실력으로 오인하면 가중치가 엉뚱하게 배분되기 때문이다.
    gate=False 는 게이팅이 실제로 값을 하는지 확인하는 대조군이다.
    """
    out: dict[str, BacktestResult] = {}
    atr = feats["atr"]
    for s in strategies:
        sig = s.generate(df, feats).copy()
        if gate:
            allowed = regimes.reindex(df.index).isin([str(r) for r in s.regimes])
            sig.loc[~allowed, ["direction", "strength"]] = 0.0
        out[s.name] = run(
            df, sig, atr, interval=interval, costs=costs, params=params,
            initial_equity=initial_equity, minute_df=minute_df, regimes=regimes,
            symbol=symbol, strategy=s.name,
        )
    return out


def derive_reward_by_regime(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    strategies: list[Strategy],
    *,
    interval: str,
    costs: Costs,
    base: ExecParams,
    minute_df: pd.DataFrame | None,
    regimes: pd.Series,
    candidates: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    min_trades: int = 50,
    gate: bool = True,
) -> dict[str, float]:
    """
    레짐별 최적 손익비를 학습 구간에서 뽑는다.

    실측에서 최적 손익비가 레짐마다 갈렸다 — 하락추세·고변동은 2.0, 횡보는 1.0.
    고정값을 쓰면 그 차이를 버리게 되고, 검증 구간을 보고 정하면 과최적화다.
    그래서 학습 구간에서만 뽑는다.

    표본이 min_trades 미만인 레짐은 전 구간 최적값으로 채운다 — 적은 표본에서
    고른 값은 잡음이다.
    """
    from dataclasses import replace as _replace

    per_ratio: dict[float, pd.DataFrame] = {}
    for ratio in candidates:
        res = run_standalone(
            df, feats, strategies, interval=interval, costs=costs,
            params=_replace(base, reward_ratio=ratio, reward_by_regime=None),
            initial_equity=10_000.0, minute_df=minute_df, regimes=regimes, gate=gate,
        )
        trades = pd.concat([r.trades for r in res.values()], ignore_index=True)
        per_ratio[ratio] = trades

    # 전 구간 기준 최적값 — 표본 부족 레짐의 대체값
    overall = {r: float(t["r_multiple"].mean()) if not t.empty else -np.inf
               for r, t in per_ratio.items()}
    fallback = max(overall, key=overall.get)

    out: dict[str, float] = {}
    for regime in [str(r) for r in Regime]:
        scores: dict[float, float] = {}
        for ratio, trades in per_ratio.items():
            g = trades[trades["regime"] == regime] if not trades.empty else trades
            if len(g) >= min_trades:
                scores[ratio] = float(g["r_multiple"].mean())
        out[regime] = max(scores, key=scores.get) if scores else fallback
    return out


def walk_forward(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    regimes: pd.Series,
    strategies: list[Strategy],
    cfg: Config,
    *,
    interval: str,
    minute_df: pd.DataFrame | None = None,
    symbol: str = "",
    weight_params: WeightParams | None = None,
    exec_params: ExecParams | None = None,
    gate: bool = True,
    adaptive_exit: bool = False,
    costs: Costs | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> WalkForwardResult:
    costs = costs or Costs.from_config(cfg)
    exec_params = exec_params or ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    wp = weight_params or WeightParams()
    initial = float(cfg.require("backtest.initial_equity"))

    # 창은 실제로 다루는 기간에서 만들어야 한다. config 전체 기간으로 만들면
    # 데이터가 없는 창이 생기고, 그 창을 건너뛰면서 창 목록과 가중치 목록의
    # 인덱스가 어긋난다 (상황판에서 가중치가 전부 0으로 보이는 사고가 있었다).
    windows = make_windows(
        start if start is not None else cfg.require("backtest.start"),
        end if end is not None else cfg.require("backtest.end"),
        int(cfg.require("backtest.walk_forward.train_months")),
        int(cfg.require("backtest.walk_forward.test_months")),
    )
    if not windows:
        raise ValueError("walk-forward 창을 만들 수 없습니다 — 기간이 너무 짧습니다")

    # 게이팅을 끄면 모든 전략이 모든 국면에서 발언권을 갖는다
    declared = ({s.name: s.regimes for s in strategies} if gate
                else {s.name: tuple(Regime) for s in strategies})
    equity_pieces: list[pd.Series] = []
    trade_pieces: list[pd.DataFrame] = []
    weights_log: list[pd.DataFrame | None] = []
    scores_log: list[pd.DataFrame | None] = []
    reward_log: list[dict[str, float]] = []

    running_equity = initial
    for w in windows:
        # ── 학습: 전략 단독 성과에서 레짐별 가중치를 뽑는다
        tr_df = _slice(df, w.train_start, w.train_end)
        tr_feats = _slice(feats, w.train_start, w.train_end)
        tr_min = _slice(minute_df, w.train_start, w.train_end) if minute_df is not None else None
        if len(tr_df) < 500:
            # 건너뛴 창에도 자리를 채운다. 리스트를 촘촘하게 만들면 창 번호와
            # 가중치 인덱스가 어긋나서, 소비하는 쪽이 남의 창 가중치를 읽는다.
            weights_log.append(None)
            scores_log.append(None)
            continue

        standalone = run_standalone(
            tr_df, tr_feats, strategies, interval=interval, costs=costs,
            params=exec_params, initial_equity=initial, minute_df=tr_min,
            regimes=regimes, symbol=symbol, gate=gate,
        )
        all_trades = pd.concat([r.trades for r in standalone.values()], ignore_index=True)
        scores = score_by_regime(all_trades, wp)
        weights = derive_weights(scores, declared, wp)
        weights_log.append(weights)
        scores_log.append(scores)

        # ── 적응형 청산: 레짐별 손익비도 학습 구간에서만 뽑는다
        window_params = exec_params
        if adaptive_exit:
            reward_map = derive_reward_by_regime(
                tr_df, tr_feats, strategies, interval=interval, costs=costs,
                base=exec_params, minute_df=tr_min, regimes=regimes, gate=gate,
            )
            window_params = dataclass_replace(exec_params, reward_by_regime=reward_map)
            reward_log.append(reward_map)

        # ── 검증: 학습에서 정한 가중치를 그대로 적용한다
        te_df = _slice(df, w.test_start, w.test_end)
        te_feats = _slice(feats, w.test_start, w.test_end)
        te_min = _slice(minute_df, w.test_start, w.test_end) if minute_df is not None else None
        if te_df.empty:
            continue

        sig_by_strat = {}
        for s in strategies:
            sig = s.generate(te_df, te_feats).copy()
            if gate:
                allowed = regimes.reindex(te_df.index).isin([str(r) for r in s.regimes])
                sig.loc[~allowed, ["direction", "strength"]] = 0.0
            sig_by_strat[s.name] = sig

        ens = combine(sig_by_strat, regimes, weights, wp)
        res = run(
            te_df, ens[["direction", "strength"]], te_feats["atr"],
            interval=interval, costs=costs, params=window_params,
            initial_equity=running_equity, minute_df=te_min, regimes=regimes,
            symbol=symbol, strategy="ensemble",
        )
        equity_pieces.append(res.equity)
        trade_pieces.append(res.trades)
        running_equity = float(res.equity.iloc[-1])

    oos_equity = (pd.concat(equity_pieces) if equity_pieces
                  else pd.Series(dtype="float64", name="equity"))
    oos_trades = (pd.concat(trade_pieces, ignore_index=True) if trade_pieces
                  else pd.DataFrame())
    bars_in_pos = int(oos_trades["hold_bars"].sum()) if not oos_trades.empty else 0
    metrics = compute_metrics(oos_trades, oos_equity, initial, interval, bars_in_pos)

    return WalkForwardResult(
        symbol=symbol, oos_trades=oos_trades, oos_equity=oos_equity, metrics=metrics,
        initial_equity=initial,
        weights_per_window=weights_log, reward_per_window=reward_log,
        train_scores=scores_log, windows=windows,
    )


def average_weights(weights_log: list[pd.DataFrame | None]) -> pd.DataFrame:
    """
    창마다 뽑힌 가중치의 평균. 가중치가 창마다 크게 흔들리면 그 자체가 경고다 —
    성과 랭킹이 잡음을 따라가고 있다는 뜻이다.

    건너뛴 창은 None 으로 자리만 채워져 있으므로 걸러낸다.
    """
    real = [w for w in weights_log if w is not None]
    if not real:
        return pd.DataFrame()
    return pd.concat(real).groupby(level=0).mean()


def weight_stability(weights_log: list[pd.DataFrame | None]) -> pd.DataFrame:
    """가중치의 창간 표준편차. 큰 값은 과최적화 위험 신호."""
    real = [w for w in weights_log if w is not None]
    if len(real) < 2:
        return pd.DataFrame()
    return pd.concat(real).groupby(level=0).std().round(3)
