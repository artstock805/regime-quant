"""
상황판 세션 — 검증된 백테스트 결과를 봉 단위 프레임으로 재구성한다.

엔진을 스트리밍용으로 고쳐 쓰지 않는다. 엔진에는 체결 순서·봉내부 선착 판정을
검증하는 테스트 16개가 붙어 있고, 그걸 흔들면 상황판에 뜨는 숫자를 믿을 수 없게 된다.

대신 이렇게 한다.
    1) walk-forward 를 완주해 거래·자산곡선·창별 가중치를 얻는다 (검증된 경로)
    2) 그 결과를 봉마다 스냅샷으로 되돌려 재생한다

따라서 화면의 모든 숫자는 백테스트가 실제로 산출한 값이다. 재생은 시각화일 뿐
별도의 계산 경로가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import ExecParams, WeightParams, walk_forward
from ..backtest.walkforward import WalkForwardResult, Window
from ..config import Config, Costs
from ..data.binance import interval_timedelta
from ..data.cache import get_klines
from ..indicators import compute_all
from ..regime import KOREAN, Regime, classify
from ..strategies import build_all

# 화면이 그리는 지표. 필요한 것만 프레임에 담는다 — 85개를 다 보내면 대역폭 낭비다.
PANEL_FEATURES = [
    "atr", "atr_pct", "adx", "di_plus", "di_minus", "rsi",
    "dc_upper", "dc_lower", "ema_spread", "volume_ratio",
    "efficiency_ratio", "choppiness", "bbw_percentile",
]


@dataclass
class Session:
    symbol: str
    interval: str
    bars: pd.DataFrame
    feats: pd.DataFrame
    regimes: pd.DataFrame
    signals: dict[str, pd.DataFrame]
    result: WalkForwardResult
    strategies: list = field(default_factory=list)
    params: ExecParams | None = None
    weight_params: WeightParams | None = None

    # 봉 인덱스 → 그 봉이 속한 walk-forward 창 번호
    window_of: np.ndarray = field(default_factory=lambda: np.array([]))
    # 봉 인덱스 → 자산
    equity: np.ndarray = field(default_factory=lambda: np.array([]))
    # 봉 인덱스 → 열린 포지션(dict) 또는 None
    position_at: list = field(default_factory=list)
    # 봉 인덱스 → 그 봉에서 발생한 체결 목록
    fills_at: dict[int, list] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.bars)

    # ─── 메타 ────────────────────────────────────────────────────

    def meta(self) -> dict[str, Any]:
        m = self.result.metrics
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "bars": self.n,
            "start": self.bars.index[0].isoformat(),
            "end": self.bars.index[-1].isoformat(),
            "windows": len(self.result.windows),
            "initial_equity": self.result.initial_equity,
            "strategies": [
                {"name": s.name, "korean": s.korean,
                 "regimes": [str(r) for r in s.regimes]}
                for s in self.strategies
            ],
            "entry_threshold": self.weight_params.entry_threshold if self.weight_params else None,
            "atr_stop_mult": self.params.atr_stop_mult if self.params else None,
            "reward_ratio": self.params.reward_ratio if self.params else None,
            "metrics": {
                "trades": m.trades, "total_return": m.total_return,
                "sharpe": m.sharpe, "max_drawdown": m.max_drawdown,
                "profit_factor": (m.profit_factor if np.isfinite(m.profit_factor) else None),
                "win_rate": m.win_rate, "expectancy": m.expectancy,
                "total_fees": m.total_fees,
            },
            # 실전 모드는 잠겨 있다. 합격 기준을 통과하지 못했다.
            "live_unlocked": False,
            "live_locked_reason": "합격 기준 미달 (5.9년 OOS 0/4) — 실전 모드는 잠겨 있습니다",
        }

    def candles(self, lo: int, hi: int) -> list[dict]:
        block = self.bars.iloc[lo:hi]
        return [
            {"t": ts.isoformat(), "o": float(r.open), "h": float(r.high),
             "l": float(r.low), "c": float(r.close), "v": float(r.volume)}
            for ts, r in block.iterrows()
        ]

    # ─── 프레임 ──────────────────────────────────────────────────

    def frame(self, i: int) -> dict[str, Any]:
        """봉 i 시점의 화면 상태 전체."""
        i = max(0, min(self.n - 1, i))
        ts = self.bars.index[i]
        row = self.bars.iloc[i]
        reg = self.regimes.iloc[i]
        regime = reg["regime"] if pd.notna(reg["regime"]) else None

        weights = self._weights_for(i)
        votes, score = self._votes(i, regime, weights)
        pos = self.position_at[i]

        feats = {
            k: (None if pd.isna(self.feats[k].iloc[i]) else float(self.feats[k].iloc[i]))
            for k in PANEL_FEATURES if k in self.feats.columns
        }

        # 체결은 직전 봉의 판단으로 일어났다. 이번 봉의 점수와 나란히 보여주면
        # "점수가 문턱 미달인데 진입했다"로 오해하게 되므로, 그 체결을 만든
        # 점수를 함께 실어 보낸다.
        fills = [dict(f) for f in self.fills_at.get(i, [])]
        if fills and i > 0:
            prev_votes, prev_score = self._votes(
                i - 1,
                self.regimes["regime"].iloc[i - 1]
                if pd.notna(self.regimes["regime"].iloc[i - 1]) else None,
                self._weights_for(i - 1),
            )
            for f in fills:
                if f["kind"] == "entry":
                    f["decision_score"] = prev_score
                    f["decision_t"] = self.bars.index[i - 1].isoformat()

        return {
            "i": i,
            "t": ts.isoformat(),
            "bar": {"o": float(row.open), "h": float(row.high),
                    "l": float(row.low), "c": float(row.close), "v": float(row.volume)},
            "regime": {
                "id": regime,
                "korean": KOREAN.get(Regime(regime), "판정중") if regime else "판정중",
                "confidence": None if pd.isna(reg["confidence"]) else float(reg["confidence"]),
                "bars_in": None if pd.isna(reg["bars_in"]) else int(reg["bars_in"]),
            },
            "window": int(self.window_of[i]) if self.window_of[i] >= 0 else None,
            "votes": votes,
            "score": score,
            "threshold": self.weight_params.entry_threshold if self.weight_params else 0.35,
            "position": pos,
            "equity": float(self.equity[i]),
            "fills": fills,
            "features": feats,
            # 화면이 의미를 정확히 쓰도록 명시한다.
            #   votes/score = 이번 봉 종가 기준 판단 → 다음 봉 시가에 체결
            #   fills       = 직전 봉 판단으로 이번 봉 시가에 체결된 것
            "semantics": {"votes": "next_bar_open", "fills": "from_prev_bar"},
        }

    def _weights_for(self, i: int) -> pd.DataFrame | None:
        """봉 i 에 실제로 적용됐던 창의 가중치. 학습 구간에서 뽑힌 값이다."""
        w = int(self.window_of[i])
        if w < 0 or w >= len(self.result.weights_per_window):
            return None
        return self.result.weights_per_window[w]

    def _votes(self, i: int, regime, weights) -> tuple[list[dict], float]:
        """전략별 신호와 가중치, 그리고 합산 점수."""
        out, score = [], 0.0
        for s in self.strategies:
            sig = self.signals[s.name]
            d = float(sig["direction"].iloc[i])
            st = float(sig["strength"].iloc[i])
            declared = any(str(r) == str(regime) for r in s.regimes) if regime else False
            w = 0.0
            if weights is not None and regime and s.name in weights.columns:
                w = float(weights.at[str(regime), s.name])
            if not declared:
                d, st = 0.0, 0.0     # 국면 밖에서는 발언권이 없다
            contrib = w * st * d
            score += contrib
            out.append({
                "name": s.name, "korean": s.korean,
                "direction": d, "strength": st, "weight": w,
                "declared": declared, "contrib": contrib,
            })
        out.sort(key=lambda v: (-v["weight"], v["korean"]))
        return out, score


# ─── 구성 ────────────────────────────────────────────────────────

def _window_index(bars: pd.DataFrame, windows: list[Window]) -> np.ndarray:
    """각 봉이 어느 검증 창에 속하는지. 학습 구간과 창 밖은 -1."""
    out = np.full(len(bars), -1, dtype=int)
    idx = bars.index
    for k, w in enumerate(windows):
        mask = (idx >= w.test_start) & (idx < w.test_end)
        out[mask] = k
    return out


def _rebuild_position_track(
    bars: pd.DataFrame, trades: pd.DataFrame, interval: str
) -> tuple[list, dict[int, list]]:
    """
    거래 목록에서 봉별 포지션 상태와 체결 이벤트를 되돌린다.

    거래는 이미 검증된 백테스트가 낸 것이므로, 여기서 손익을 다시 계산하지 않는다.
    화면에 필요한 것(방향·수량·평단·손절·목표·미실현)만 채운다.
    """
    n = len(bars)
    position_at: list = [None] * n
    fills_at: dict[int, list] = {}
    if trades.empty:
        return position_at, fills_at

    pos_by_time = bars.index.get_indexer
    close = bars["close"].to_numpy()

    for _, t in trades.iterrows():
        lo = int(pos_by_time([pd.Timestamp(t["entry_time"])])[0])
        hi = int(pos_by_time([pd.Timestamp(t["exit_time"])])[0])
        if lo < 0:
            continue
        if hi < 0:
            hi = n - 1

        side = "롱" if t["direction"] > 0 else "숏"
        for j in range(lo, min(hi, n - 1) + 1):
            upnl = (close[j] - t["entry_price"]) * t["qty"] * t["direction"]
            position_at[j] = {
                "side": side, "direction": int(t["direction"]),
                "qty": float(t["qty"]), "entry": float(t["entry_price"]),
                "notional": float(t["notional"]),
                "risk_amount": float(t["risk_amount"]),
                "upnl": float(upnl),
                "upnl_r": float(upnl / t["risk_amount"]) if t["risk_amount"] else 0.0,
                "held_bars": j - lo,
                "strategy": t["strategy"], "regime": t["regime"],
                "strength": float(t["strength"]),
            }

        fills_at.setdefault(lo, []).append({
            "kind": "entry", "side": side, "qty": float(t["qty"]),
            "price": float(t["entry_price"]), "fee": None,
            "strategy": t["strategy"], "reason": "앙상블 진입",
            "t": pd.Timestamp(t["entry_time"]).isoformat(),
        })
        fills_at.setdefault(hi, []).append({
            "kind": "exit", "side": side, "qty": float(t["qty"]),
            "price": float(t["exit_price"]),
            "net_pnl": float(t["net_pnl"]), "r_multiple": float(t["r_multiple"]),
            "fee": float(t["fees"]), "reason": t["exit_reason"],
            "strategy": t["strategy"],
            "t": pd.Timestamp(t["exit_time"]).isoformat(),
        })
    return position_at, fills_at


def _equity_track(bars: pd.DataFrame, result: WalkForwardResult) -> np.ndarray:
    """
    자산곡선을 전체 봉 인덱스에 맞춘다.

    walk-forward 는 검증 구간만 거래하므로 학습 구간에는 자산 데이터가 없다.
    앞은 초기자본으로, 사이 구멍은 직전 값으로 채운다 — 거래가 없었으니 자산도 그대로다.
    """
    eq = pd.Series(np.nan, index=bars.index)
    if not result.oos_equity.empty:
        eq.loc[result.oos_equity.index] = result.oos_equity.to_numpy()
    eq = eq.ffill().fillna(result.initial_equity)
    return eq.to_numpy(dtype="float64")


def build_session(
    cfg: Config,
    symbol: str,
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
    entry_threshold: float | None = None,
) -> Session:
    """walk-forward 를 돌리고 그 결과로 재생 세션을 만든다."""
    kd = cfg.path("paths.klines_dir")
    start = start or cfg.require("backtest.start")
    end = end or cfg.require("backtest.end")

    bars = get_klines(symbol, interval, start, end,
                      market=cfg.require("market"), klines_dir=kd)
    if bars.empty:
        raise ValueError(f"{symbol} {interval} 데이터가 없습니다 ({start} → {end})")
    minute = get_klines(symbol, "1m", start, end,
                        market=cfg.require("market"), klines_dir=kd)

    feats = compute_all(bars)
    regimes = classify(feats)

    names = cfg.get("strategies")
    strategies = [s for s in build_all() if not names or s.name in set(names)]

    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    wp = WeightParams() if entry_threshold is None else WeightParams(entry_threshold=entry_threshold)

    result = walk_forward(
        bars, feats, regimes["regime"], strategies, cfg, interval=interval,
        minute_df=minute, symbol=symbol, weight_params=wp, exec_params=params,
        costs=Costs.from_config(cfg),
        # 창을 세션 기간에서 만들게 한다. config 전체 기간으로 만들면 데이터 없는
        # 창이 생기고 창 번호와 가중치 인덱스가 어긋난다.
        start=start, end=end,
    )

    signals = {s.name: s.generate(bars, feats) for s in strategies}
    window_of = _window_index(bars, result.windows)
    position_at, fills_at = _rebuild_position_track(bars, result.oos_trades, interval)

    return Session(
        symbol=symbol, interval=interval, bars=bars, feats=feats, regimes=regimes,
        signals=signals, result=result, strategies=strategies, params=params,
        weight_params=wp, window_of=window_of,
        equity=_equity_track(bars, result),
        position_at=position_at, fills_at=fills_at,
    )


def first_traded_bar(session: Session) -> int:
    """거래가 시작되는 첫 봉. 재생 시작점 기본값 — 학습 구간을 지루하게 볼 필요 없다."""
    if not session.fills_at:
        return 0
    return max(0, min(session.fills_at) - 60)
