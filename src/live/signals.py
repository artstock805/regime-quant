"""
실시간 신호 — 진입·청산 신호만 낸다. **주문은 내지 않는다.**

이 모듈에 주문 코드가 없는 것은 미완성이 아니라 설계다. 5.9년 검증에서
5항목 중 1개만 통과했고, Deflated Sharpe 는 0.865(기준 0.95)다. 신호를 보는 것과
그 신호로 돈을 거는 것은 다른 결정이고, 후자는 사람이 해야 한다.

**백테스트와 같은 인과 규칙을 쓴다.** 이게 이 파일의 존재 이유다.

    신호는 **마감된 봉**의 종가로 계산한다.
    체결은 **다음 봉 시가**에서 일어난다고 가정한다.

진행 중인 봉으로 판단하면 화면의 신호와 백테스트 숫자가 다른 것이 되고,
그러면 검증 결과가 신호를 보증하지 못한다. `closed_only=True` 가 그 경계다.

가중치는 어디서 오는가
    walk-forward 가 하는 것과 같다 — **최근 학습 구간(기본 6개월)에서 전략을
    단독으로 돌려** 레짐별 기대R 을 뽑고 거기서 가중치를 만든다. 오늘 성적으로
    오늘 가중치를 만들면 그게 미래 참조다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..backtest import ExecParams, WeightParams, combine, derive_weights, run_standalone
from ..backtest.ensemble import score_by_regime
from ..config import Config, Costs
from ..data.binance import fetch_klines, interval_timedelta
from ..data.derivatives import ALL_FIELDS, align_to_bars, get_derivatives
from ..indicators import compute_all
from ..regime import KOREAN, Regime, classify
from ..strategies import build_all


class MissingFeatures(RuntimeError):
    """전략이 요구하는 특징열이 없다. 대개 파생 데이터를 끈 경우다."""

    def __init__(self, symbol: str, missing: list[str], use_derivatives: bool):
        self.symbol, self.missing = symbol, missing
        hint = ("파생 데이터가 필요합니다 — --no-derivatives 를 빼고 다시 실행하세요"
                if not use_derivatives else
                "파생 데이터를 받아야 합니다: python scripts/fetch_derivatives.py")
        super().__init__(f"{symbol}: 특징열 없음 {missing} — {hint}")


# 지표 워밍업. Wilder 평활이 기준 구현과 수렴하려면 400봉이 필요하다(실측).
# 파생 z-score 가 168봉을 쓰므로 넉넉히 잡는다.
WARMUP_BARS = 900


@dataclass
class StrategyVote:
    strategy: str
    korean: str
    direction: int
    strength: float
    weight: float
    contribution: float
    has_voice: bool


@dataclass
class SymbolSignal:
    symbol: str
    interval: str
    bar_time: str                 # 판단 근거가 된 **마감된** 봉
    next_bar_time: str            # 체결이 일어난다고 가정하는 시각
    close: float
    regime: str
    regime_korean: str
    regime_confidence: float
    score: float
    threshold: float
    action: str                   # 롱진입 / 숏진입 / 청산 / 유지 / 관망
    direction: int
    prev_direction: int
    atr: float
    stop_price: float | None
    target_price: float | None
    risk_per_unit: float | None   # 1R 을 가격으로
    votes: list[StrategyVote] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _action(prev: int, cur: int) -> str:
    if prev == cur:
        return "유지" if cur != 0 else "관망"
    if cur == 1:
        return "롱진입"
    if cur == -1:
        return "숏진입"
    return "청산"


def _load(cfg: Config, symbol: str, interval: str, *, use_derivatives: bool,
          train_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    최근 봉을 받아 특징까지 계산한다.

    **마감된 봉만 받는다**(closed_only=True). 진행 중인 봉을 섞으면 값이 매분
    바뀌어 신호가 흔들리고, 백테스트 규칙과도 어긋난다.
    """
    step = interval_timedelta(interval)
    now = datetime.now(timezone.utc)
    need = timedelta(days=train_days) + step * WARMUP_BARS
    bars = fetch_klines(symbol, interval, now - need, now,
                        market=cfg.require("market"), closed_only=True)
    if bars.empty:
        return bars, bars

    if use_derivatives:
        deriv = get_derivatives(symbol, bars.index[0], now,
                                market=cfg.require("market"),
                                deriv_dir=cfg.path("paths.derivatives_dir"))
        bars = align_to_bars(bars, deriv, interval)
        fields = [c for c in ALL_FIELDS if c in bars.columns]
        if fields:
            have = bars[fields].notna().all(axis=1)
            if have.any():
                bars = bars[have]
    return bars, compute_all(bars)


def signal_for(
    cfg: Config,
    symbol: str,
    *,
    interval: str = "1h",
    use_derivatives: bool = True,
    strategies: list | None = None,
    weight_params: WeightParams | None = None,
    exec_params: ExecParams | None = None,
    costs: Costs | None = None,
    bars: pd.DataFrame | None = None,
    feats: pd.DataFrame | None = None,
) -> SymbolSignal | None:
    """
    종목 하나의 현재 신호. 데이터가 부족하면 None.

    bars/feats 를 넘기면 그것을 쓴다. FX 는 봉이 MT5 에서 오기 때문이다 —
    **주문을 낼 브로커의 호가로 신호를 계산해야** 체결과 어긋나지 않는다.
    넘기지 않으면 기존대로 캐시(바이낸스)에서 읽는다.
    """
    names = cfg.get("strategies")
    strats = strategies or [s for s in build_all()
                            if not names or s.name in set(names)]
    wp = weight_params or WeightParams(
        entry_threshold=float(cfg.get("live.entry_threshold",
                                      cfg.get("risk.entry_threshold", 0.85))),
        calibrate_strength=bool(cfg.get("live.calibrate_strength", True)),
    )
    ep = exec_params or ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    cost = costs or Costs.from_config(cfg)
    train_months = int(cfg.get("backtest.walk_forward.train_months", 6))
    train_days = train_months * 31

    if bars is None or feats is None:
        bars, feats = _load(cfg, symbol, interval,
                            use_derivatives=use_derivatives,
                            train_days=train_days)
    if bars.empty or len(bars) < WARMUP_BARS // 2:
        return None

    # 전략이 요구하는 특징열이 없으면 여기서 이유를 밝힌다.
    # 기본 전략(taker_flow_momentum)은 파생 데이터에서 오는 perp_taker_ls 를 쓴다.
    # --no-derivatives 로 끄면 KeyError 만 나고 원인이 보이지 않았다.
    missing = sorted({c for s in strats for c in s.needs if c not in feats.columns})
    if missing:
        raise MissingFeatures(symbol, missing, use_derivatives)

    regimes = classify(feats)
    reg_series = regimes["regime"]

    # ── 가중치: 최근 학습 구간에서만 뽑는다 (walk-forward 와 같은 규칙)
    cutoff = bars.index[-1] - timedelta(days=train_days)
    tr = bars.index >= cutoff
    train_bars, train_feats = bars[tr], feats[tr]
    weights = None
    if len(train_bars) >= 500:
        res = run_standalone(train_bars, train_feats, strats, interval=interval,
                             costs=cost, params=ep, initial_equity=10_000,
                             minute_df=None, regimes=reg_series[tr], symbol=symbol)
        trades = [r.trades for r in res.values() if not r.trades.empty]
        if trades:
            scores = score_by_regime(pd.concat(trades, ignore_index=True), wp)
            weights = derive_weights(scores, {s.name: s.regimes for s in strats}, wp)
    if weights is None:
        weights = derive_weights(pd.DataFrame(),
                                 {s.name: s.regimes for s in strats}, wp)

    # ── 신호
    sigs = {s.name: s.generate(bars, feats) for s in strats}
    ens = combine(sigs, reg_series, weights, wp)

    last, prev = ens.iloc[-1], (ens.iloc[-2] if len(ens) >= 2 else ens.iloc[-1])
    cur_dir, prev_dir = int(last["direction"]), int(prev["direction"])
    rid = reg_series.iloc[-1]
    rid = rid if isinstance(rid, str) else None
    close = float(bars["close"].iloc[-1])
    atr = float(feats["atr"].iloc[-1]) if "atr" in feats.columns else float("nan")

    risk = atr * ep.atr_stop_mult if np.isfinite(atr) else None
    stop = target = None
    if risk and cur_dir != 0:
        stop = close - cur_dir * risk
        target = close + cur_dir * risk * ep.reward_ratio

    votes = []
    for s in strats:
        sv = sigs[s.name].iloc[-1]
        declared = bool(rid and any(str(r) == rid for r in s.regimes))
        w = float(weights.at[rid, s.name]) if (rid and rid in weights.index) else 0.0
        votes.append(StrategyVote(
            strategy=s.name, korean=s.korean,
            direction=int(sv["direction"]) if declared else 0,
            strength=round(float(sv["strength"]) if declared else 0.0, 4),
            weight=round(w, 4),
            contribution=round(w * float(sv["strength"]) * float(sv["direction"])
                               if declared else 0.0, 4),
            has_voice=declared and w > 0,
        ))

    step = interval_timedelta(interval)
    return SymbolSignal(
        symbol=symbol, interval=interval,
        bar_time=bars.index[-1].isoformat(),
        next_bar_time=(bars.index[-1] + step).isoformat(),
        close=round(close, 8),
        regime=rid or "판정중",
        regime_korean=KOREAN.get(Regime(rid), "판정중") if rid else "판정중",
        regime_confidence=round(float(regimes["confidence"].iloc[-1]), 3)
        if pd.notna(regimes["confidence"].iloc[-1]) else 0.0,
        score=round(float(last["score"]), 4),
        threshold=wp.entry_threshold,
        action=_action(prev_dir, cur_dir),
        direction=cur_dir, prev_direction=prev_dir,
        atr=round(atr, 8) if np.isfinite(atr) else float("nan"),
        stop_price=None if stop is None else round(stop, 8),
        target_price=None if target is None else round(target, 8),
        risk_per_unit=None if risk is None else round(risk, 8),
        votes=votes,
        note=("신호는 마감된 봉 기준이고 체결은 다음 봉 시가를 가정한다. "
              "주문 기능은 없다 — 검증 1/5 이므로 사람이 판단해야 한다."),
    )


def scan(cfg: Config, symbols: list[str], *, interval: str = "1h",
         use_derivatives: bool = True, log=None) -> list[SymbolSignal]:
    """여러 종목을 훑는다. 한 종목이 실패해도 나머지는 계속한다."""
    out: list[SymbolSignal] = []
    for i, sym in enumerate(symbols, 1):
        try:
            s = signal_for(cfg, sym, interval=interval,
                           use_derivatives=use_derivatives)
        except Exception as exc:                       # 네트워크·데이터 문제
            if log:
                log(f"  {sym}: 실패 {type(exc).__name__}: {exc}")
            continue
        if s is None:
            if log:
                log(f"  {sym}: 데이터 부족")
            continue
        out.append(s)
        if log:
            log(f"  [{i}/{len(symbols)}] {sym:<12} {s.action:<6} "
                f"점수 {s.score:+.3f} · {s.regime_korean}")
    return out


def actionable(signals: list[SymbolSignal]) -> list[SymbolSignal]:
    """상태가 바뀐 것만 — 진입·청산. '유지/관망'은 알림 대상이 아니다."""
    return [s for s in signals if s.action in {"롱진입", "숏진입", "청산"}]
