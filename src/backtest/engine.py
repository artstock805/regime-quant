"""
백테스트 엔진.

체결 순서를 엄격히 지킨다. 이 순서를 흐트리면 백테스트가 거짓말을 시작한다.

    봉 i-1 종가  →  신호 계산 (여기까지의 데이터만 사용)
    봉 i   시가  →  체결 (슬리피지·수수료 적용)
    봉 i   내부  →  1분봉으로 손절/목표 선착 판정
    봉 i   종가  →  자산 평가(미실현 포함)

신호가 나온 봉의 종가로 체결하는 것은 미래 참조다 — 종가를 알기 전에는 주문을
낼 수 없다. 엔진이 이 지연을 강제하므로 전략은 신경 쓰지 않는다.

**봉 내부 선착 판정**이 이 엔진의 존재 이유다. 5분봉만 보면 그 봉 안에서 손절과
목표 중 무엇이 먼저 닿았는지 알 수 없다. 이걸 얼버무리는 것이 백테스트에서 가장
흔한 거짓말이고, 손익비 2:1 전략에서 결과를 두 배로 부풀린다.
1분봉을 주면 순서대로 확인하고, 한 분봉 안에서 둘 다 닿았으면 **손절이 먼저**로
본다(보수적). 1분봉이 없으면 그 봉은 손절만 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import Costs
from ..data.binance import interval_timedelta
from .metrics import Metrics, compute_metrics

TRADE_COLUMNS = [
    "symbol", "strategy", "direction", "entry_time", "entry_price", "exit_time",
    "exit_price", "qty", "notional", "risk_amount", "gross_pnl", "fees", "slip_cost", "cost_r",
    "net_pnl", "r_multiple", "gross_r", "price_return_pct",
    "hold_bars", "exit_reason", "regime", "strength", "mae_pct", "mfe_pct",
]


@dataclass(frozen=True)
class ExecParams:
    """실행 규칙. 전부 config.yaml 의 risk 섹션에서 온다."""
    risk_per_trade: float = 0.01
    atr_stop_mult: float = 1.5
    reward_ratio: float = 2.0
    daily_loss_limit: float = 0.03
    trailing: bool = True
    # 레버리지 상한. 1% 리스크 + ATR 손절이면 명목가가 자산의 2~3배가 되는 게
    # 정상이라, 상한이 없으면 저변동 구간에서 비현실적인 사이즈가 나온다.
    max_leverage: float = 3.0
    # 확신도가 이 값 미만이면 진입하지 않는다
    min_strength: float = 0.0
    # 레짐별 손익비. 진입 시점 레짐으로 목표 거리를 정한다.
    # None 이면 reward_ratio 고정값을 쓴다. 실측에서 최적 손익비가 레짐마다
    # 달랐기 때문에 둔 장치다 (하락추세·고변동 2.0 / 횡보 1.0).
    # 값은 학습 구간에서 뽑아야 한다 — 검증 구간을 보고 정하면 과최적화다.
    reward_by_regime: dict[str, float] | None = None

    def reward_for(self, regime) -> float:
        if not self.reward_by_regime or regime is None:
            return self.reward_ratio
        return self.reward_by_regime.get(str(regime), self.reward_ratio)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: Metrics
    interval: str
    initial_equity: float
    bars_in_position: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"BacktestResult({self.metrics.trades}거래, "
                f"{self.metrics.total_return*100:+.1f}%, 샤프 {self.metrics.sharpe:.2f})")


def _minute_bounds(
    bar_index: pd.DatetimeIndex, minute_index: pd.DatetimeIndex, span: timedelta
) -> tuple[np.ndarray, np.ndarray]:
    """각 신호 봉에 대응하는 1분봉 구간 [시작, 끝) 의 정수 인덱스."""
    starts = np.searchsorted(minute_index.values, bar_index.values, side="left")
    ends = np.searchsorted(minute_index.values, (bar_index + span).values, side="left")
    return starts, ends


def run(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    atr: pd.Series,
    *,
    interval: str,
    costs: Costs,
    params: ExecParams,
    initial_equity: float,
    minute_df: pd.DataFrame | None = None,
    regimes: pd.Series | None = None,
    symbol: str = "",
    strategy: str = "",
) -> BacktestResult:
    """
    단일 종목·단일 신호열 백테스트.

    df       신호 타임프레임 OHLCV
    signals  direction(-1/0/+1), strength(0~1) — df 와 같은 인덱스
    atr      손절 폭 산정용 ATR — df 와 같은 인덱스
    minute_df  1분봉. 없으면 봉 내부 판정이 손절 전용으로 축소된다.
    """
    if not df.index.equals(signals.index):
        raise ValueError("signals 인덱스가 df 와 다릅니다")
    if not df.index.equals(atr.index):
        raise ValueError("atr 인덱스가 df 와 다릅니다")

    n = len(df)
    span = interval_timedelta(interval)
    slip = costs.slippage_rate
    fee = costs.fee_rate

    op = df["open"].to_numpy("float64")
    hi = df["high"].to_numpy("float64")
    lo = df["low"].to_numpy("float64")
    cl = df["close"].to_numpy("float64")
    atr_a = atr.to_numpy("float64")
    direction = signals["direction"].to_numpy("float64")
    strength = signals["strength"].to_numpy("float64")
    times = df.index
    dates = df.index.normalize().values
    reg_a = (regimes.reindex(df.index).to_numpy(dtype=object)
             if regimes is not None else np.full(n, None, dtype=object))

    have_minutes = minute_df is not None and not minute_df.empty
    if have_minutes:
        m_hi = minute_df["high"].to_numpy("float64")
        m_lo = minute_df["low"].to_numpy("float64")
        m_start, m_end = _minute_bounds(df.index, minute_df.index, span)

    cash = initial_equity
    equity_curve = np.full(n, initial_equity, dtype="float64")
    bars_in_pos = 0
    trades: list[dict] = []
    skipped = {"확신도미달": 0, "ATR없음": 0, "일손실한도": 0, "사이즈0": 0}

    pos: dict | None = None
    day_start_cash = cash
    current_day = dates[0] if n else None

    def _close_position(exit_px: float, exit_i: int, reason: str) -> None:
        nonlocal cash, pos
        assert pos is not None
        d = pos["dir"]
        # 청산 슬리피지는 항상 불리한 방향으로
        fill = exit_px * (1.0 - slip * d)
        gross = (fill - pos["entry_px"]) * pos["qty"] * d
        # 슬리피지는 체결가에 이미 녹아 있어 따로 차감하지 않는다.
        # 다만 '비용이 우위를 먹었는지'를 판단하려면 금액을 알아야 하므로 기록한다.
        slip_cost = slip * (abs(pos["raw_entry"] * pos["qty"]) + abs(exit_px * pos["qty"]))
        exit_fee = abs(fill * pos["qty"]) * fee
        total_fees = pos["entry_fee"] + exit_fee
        net = gross - exit_fee   # 진입 수수료는 진입 시 이미 차감했다

        cash += net
        net_pnl = net - pos["entry_fee"]
        risk_amount = pos["risk_equity"] * params.risk_per_trade
        trades.append({
            "symbol": symbol, "strategy": strategy, "direction": int(d),
            "entry_time": pos["entry_time"], "entry_price": pos["entry_px"],
            "exit_time": times[exit_i], "exit_price": fill,
            "qty": pos["qty"], "notional": pos["notional"],
            "risk_amount": risk_amount,
            "gross_pnl": gross, "fees": total_fees, "slip_cost": slip_cost,
            # 이 트레이드가 비용으로 낸 금액을 1R 단위로. 추정이 아니라 실제 금액이다.
            "cost_r": (total_fees + slip_cost) / risk_amount if risk_amount > 0 else 0.0,
            "net_pnl": net_pnl,
            # R배수: 이 트레이드가 감수한 리스크의 몇 배를 벌었는지. 계좌 크기와
            # 무관하게 비교되므로 트레이드 품질 지표로는 수익률보다 낫다.
            "r_multiple": net_pnl / risk_amount if risk_amount > 0 else 0.0,
            # 비용을 되돌린 기대값. 음수면 전략에 우위가 없다는 뜻이고,
            # 양수인데 순기대값이 음수면 우위가 비용보다 작다는 뜻이다 — 처방이 다르다.
            "gross_r": ((net_pnl + total_fees + slip_cost) / risk_amount
                        if risk_amount > 0 else 0.0),
            "price_return_pct": (fill / pos["entry_px"] - 1.0) * 100.0 * d,
            "hold_bars": exit_i - pos["entry_i"],
            "exit_reason": reason, "regime": pos["regime"], "strength": pos["strength"],
            "mae_pct": pos["mae"] * 100.0, "mfe_pct": pos["mfe"] * 100.0,
        })
        pos = None

    def _open_position(i: int, d: float, stren: float) -> None:
        nonlocal cash, pos
        a = atr_a[i]
        if not np.isfinite(a) or a <= 0:
            skipped["ATR없음"] += 1
            return
        entry_px = op[i] * (1.0 + slip * d)
        stop_dist = a * params.atr_stop_mult
        if stop_dist <= 0:
            skipped["ATR없음"] += 1
            return

        equity_now = cash
        qty = equity_now * params.risk_per_trade / stop_dist
        # 레버리지 상한
        max_qty = equity_now * params.max_leverage / entry_px
        qty = min(qty, max_qty)
        if qty <= 0:
            skipped["사이즈0"] += 1
            return

        entry_fee = abs(entry_px * qty) * fee
        cash -= entry_fee
        # 목표 거리는 진입 시점 레짐에 따라 달라질 수 있다 (적응형 청산)
        reward = params.reward_for(reg_a[i])
        pos = {
            "dir": d, "entry_px": entry_px, "raw_entry": op[i], "qty": qty,
            "notional": entry_px * qty, "reward": reward,
            "stop": entry_px - stop_dist * d,
            "target": entry_px + stop_dist * reward * d,
            "entry_i": i, "entry_time": times[i], "entry_fee": entry_fee,
            "strength": stren, "regime": reg_a[i], "risk_equity": equity_now,
            "best": entry_px, "mae": 0.0, "mfe": 0.0,
        }

    for i in range(1, n):
        # ─ 날짜가 바뀌면 일일 손실 카운터를 리셋
        if dates[i] != current_day:
            current_day = dates[i]
            day_start_cash = cash

        # ─ Step A: 봉 i 시가에서 봉 i-1 신호에 따라 행동
        want = direction[i - 1]
        if pos is not None and want != pos["dir"]:
            _close_position(op[i], i, "signal")

        if pos is None and want != 0.0:
            # 현금 변화로 판정한다 — 진입·청산 수수료가 자동으로 포함된다
            blocked = (cash - day_start_cash) <= -params.daily_loss_limit * day_start_cash
            if blocked:
                skipped["일손실한도"] += 1
            elif strength[i - 1] < params.min_strength:
                skipped["확신도미달"] += 1
            else:
                _open_position(i, want, strength[i - 1])

        # ─ Step B: 봉 i 내부에서 손절/목표 선착 판정
        if pos is not None:
            d = pos["dir"]
            hit, level, reason = False, 0.0, ""

            if have_minutes and m_end[i] > m_start[i]:
                for k in range(m_start[i], m_end[i]):
                    stop_hit = (m_lo[k] <= pos["stop"]) if d > 0 else (m_hi[k] >= pos["stop"])
                    tgt_hit = (m_hi[k] >= pos["target"]) if d > 0 else (m_lo[k] <= pos["target"])
                    if stop_hit:
                        # 같은 분봉에서 둘 다 닿으면 손절이 먼저 — 보수적으로
                        hit, level, reason = True, pos["stop"], "stop"
                        break
                    if tgt_hit:
                        hit, level, reason = True, pos["target"], "target"
                        break
            else:
                # 1분봉이 없으면 손절만 본다. 목표를 함께 보면 어느 쪽이 먼저였는지
                # 모르는 상태에서 유리한 쪽을 고르게 되어 결과가 부풀려진다.
                stop_hit = (lo[i] <= pos["stop"]) if d > 0 else (hi[i] >= pos["stop"])
                if stop_hit:
                    hit, level, reason = True, pos["stop"], "stop"

            # 최대 역행/순행 폭 기록 (손절 폭 적정성 진단용)
            adverse = (pos["entry_px"] - lo[i]) if d > 0 else (hi[i] - pos["entry_px"])
            favour = (hi[i] - pos["entry_px"]) if d > 0 else (pos["entry_px"] - lo[i])
            pos["mae"] = max(pos["mae"], adverse / pos["entry_px"])
            pos["mfe"] = max(pos["mfe"], favour / pos["entry_px"])

            if hit:
                _close_position(level, i, reason)
            elif params.trailing:
                # 유리한 방향으로 간 만큼 손절을 끌어올린다 (되돌리지는 않는다)
                a = atr_a[i]
                if np.isfinite(a) and a > 0:
                    pos["best"] = max(pos["best"], cl[i]) if d > 0 else min(pos["best"], cl[i])
                    trail = pos["best"] - a * params.atr_stop_mult * d
                    pos["stop"] = max(pos["stop"], trail) if d > 0 else min(pos["stop"], trail)

        # ─ Step C: 봉 i 종가 기준 자산 평가
        if pos is not None:
            unrealised = (cl[i] - pos["entry_px"]) * pos["qty"] * pos["dir"]
            equity_curve[i] = cash + unrealised
            bars_in_pos += 1
        else:
            equity_curve[i] = cash

    # 마지막 봉에서 열려 있으면 종가로 정리한다 — 미청산 포지션을 성과에서 빼면
    # 손실 중인 포지션을 숨기는 셈이 된다
    if pos is not None:
        _close_position(cl[n - 1], n - 1, "end")
        equity_curve[n - 1] = cash

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity = pd.Series(equity_curve, index=df.index, name="equity")
    metrics = compute_metrics(trades_df, equity, initial_equity, interval, bars_in_pos)

    return BacktestResult(
        trades=trades_df, equity=equity, metrics=metrics, interval=interval,
        initial_equity=initial_equity, bars_in_position=bars_in_pos,
        skipped={k: v for k, v in skipped.items() if v},
    )


def exec_params_from_config(cfg) -> ExecParams:
    return ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
