"""
교차검증 — 같은 규칙을 backtrader 로 다시 돌려 엔진의 거래 목록과 대조한다.

**왜 두 번 돌리나.** 이 프로젝트에서 숫자를 망친 것은 언제나 규칙이 아니라
배선이었다. stop_loss=0 을 진짜 손절로 받아 1R 이 진입가가 된 적이 있고,
spread==0 을 "비용 없음"으로 읽어 에너지 계열이 유일한 양수라는 거짓 결과가
나온 적이 있다. 둘 다 규칙은 맞았고 배선이 틀렸다. 같은 사람이 같은 코드를
다시 읽어서는 잘 안 보인다 — **다른 구현이 같은 답을 내는지** 보는 편이 낫다.

**무엇을 증명하고 무엇을 증명하지 못하나.** 이걸 분명히 해 둔다.

    증명한다   봉 인덱싱·신호 지연(backtrader 피드가 독립적으로 정렬한다)
               ATR (bt.indicators.ATR 은 완전히 다른 구현이다)
               손절·추적손절·슬리피지·수수료·손익 산술
               상태기계 (포지션 추적, 일일 손실 카운터 리셋)

    증명 못 한다  **규칙 자체가 옳은지.** 둘 다 같은 명세를 구현한다.
                  명세가 틀렸으면 둘이 사이좋게 같이 틀린다.

**ATR 은 워밍업 뒤 완전히 일치한다.** EURUSD 1시간봉 2,986봉 실측에서
256봉 이후 상대오차가 정확히 0 이었다(중위도 최대도 0.000e+00). 그래서
`warmup` 이전 봉에서 시작한 거래는 비교에서 뺀다 — 거기서 어긋나는 것은
엔진 버그가 아니라 Wilder 평활이 아직 수렴하지 않은 것이다.

**봉 단위만 다룬다(minute_df 없음).** 엔진은 1분봉이 있으면 봉 내부 선착을
판정하고, 없으면 **손절만** 본다. 이 하네스는 후자만 재현한다. 그게 한계인
동시에 지금 FX 가 실제로 도는 경로다 — run_standalone 이 minute_df=None 으로
불린다. 1분봉 경로는 아직 이 하네스로 검증되지 않았다.

**첫 어긋남이 전부다.** 거래 하나가 어긋나면 현금이 달라지고, 현금이 달라지면
그 뒤 모든 거래의 수량이 달라져 어긋남이 폭포처럼 번진다. 그래서 보고서는
**처음 어긋난 지점**을 맨 위에 놓는다. 그 아래는 대개 같은 원인의 그림자다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import backtrader as bt
import numpy as np
import pandas as pd

from ..config import Costs
from .engine import ExecParams, run

# ATR(Wilder) 이 기준 구현과 수렴하는 데 필요한 봉 수. 실측 256 에 여유를 뒀다.
WARMUP_BARS = 300

# 비교하는 열과 허용 오차. 가격은 상대오차, 시각은 완전 일치를 요구한다.
PRICE_TOLERANCE = 1e-9


class SignalData(bt.feeds.PandasData):
    """
    OHLCV 에 direction/strength 를 실어 나른다.

    **정렬을 backtrader 에게 맡기는 것이 요점이다.** 신호를 위치 인덱스로
    직접 끼워 넣으면 엔진과 같은 off-by-one 을 그대로 복제하게 된다. 피드가
    스스로 맞추게 두어야 인덱싱이 독립적으로 검증된다.
    """

    lines = ("direction", "strength")
    params = (("direction", "direction"), ("strength", "strength"))


@dataclass
class Divergence:
    """한 거래에서 어긋난 항목."""
    index: int
    field: str
    engine: object
    backtrader: object

    def __str__(self) -> str:
        return (f"거래 #{self.index} · {self.field}: "
                f"엔진 {self.engine!r} ≠ backtrader {self.backtrader!r}")


@dataclass
class CrossCheck:
    engine_trades: pd.DataFrame
    bt_trades: pd.DataFrame
    divergences: list[Divergence] = field(default_factory=list)
    compared: int = 0
    warmup_cutoff: pd.Timestamp | None = None

    @property
    def agrees(self) -> bool:
        return not self.divergences and len(self.engine_trades) == len(self.bt_trades)

    @property
    def first(self) -> Divergence | None:
        return min(self.divergences, key=lambda d: d.index) if self.divergences else None

    def report(self) -> str:
        lines = [
            f"비교 구간 {self.warmup_cutoff} 이후 · "
            f"엔진 {len(self.engine_trades)}거래 · backtrader {len(self.bt_trades)}거래",
        ]
        if self.agrees:
            lines.append(f"일치 — {self.compared}건 전부 같다")
            return "\n".join(lines)

        if len(self.engine_trades) != len(self.bt_trades):
            lines.append(f"거래 수가 다르다 "
                         f"({len(self.engine_trades)} vs {len(self.bt_trades)})")
        f = self.first
        if f is not None:
            lines.append(f"첫 어긋남 → {f}")
            lines.append("  (그 뒤의 어긋남은 대개 현금이 달라져 번진 그림자다)")
        by_field: dict[str, int] = {}
        for d in self.divergences:
            by_field[d.field] = by_field.get(d.field, 0) + 1
        if by_field:
            lines.append("항목별 어긋남: " + " · ".join(
                f"{k} {v}건" for k, v in sorted(by_field.items(), key=lambda x: -x[1])))
        return "\n".join(lines)


class _Shadow(bt.Strategy):
    """
    엔진과 같은 규칙을 backtrader 의 데이터 접근으로 다시 구현한다.

    **backtrader 의 broker 주문을 쓰지 않는다.** 실측으로 확정한 이유가 있다 —
    backtrader 는 손절 주문을 **제출된 다음 봉부터** 평가한다(cheat_on_open 도
    이걸 바꾸지 못한다). 엔진은 진입한 그 봉에서 바로 손절을 본다. 그대로
    붙이면 두 엔진이 한 봉씩 체계적으로 어긋나고, 그 소음에 진짜 버그가 묻힌다.
    그래서 broker 는 빼고, backtrader 는 **독립적인 봉 정렬과 ATR** 로만 쓴다.
    """

    params = (
        ("costs", None), ("exec_params", None),
        ("initial_equity", 10_000.0), ("atr_period", 14),
    )

    def __init__(self) -> None:
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.cash = float(self.p.initial_equity)
        self.holding: dict | None = None
        self.trades: list[dict] = []
        self.day = None
        self.day_start_cash = self.cash

    # ─── 청산·진입 (engine._close_position / _open_position 과 같은 산술) ──

    def _close(self, exit_px: float, reason: str) -> None:
        p, ep, c = self.holding, self.p.exec_params, self.p.costs
        assert p is not None
        slip, fee = c.slippage_rate, c.fee_rate
        d = p["dir"]
        fill = exit_px * (1.0 - slip * d)
        gross = (fill - p["entry_px"]) * p["qty"] * d
        slip_cost = slip * (abs(p["raw_entry"] * p["qty"]) + abs(exit_px * p["qty"]))
        exit_fee = abs(fill * p["qty"]) * fee
        net = gross - exit_fee
        self.cash += net
        net_pnl = net - p["entry_fee"]
        risk_amount = p["risk_equity"] * ep.risk_per_trade
        self.trades.append({
            "direction": int(d),
            "entry_time": p["entry_time"], "entry_price": p["entry_px"],
            "exit_time": self.data.datetime.datetime(0), "exit_price": fill,
            "qty": p["qty"], "net_pnl": net_pnl,
            "r_multiple": net_pnl / risk_amount if risk_amount > 0 else 0.0,
            "exit_reason": reason,
        })
        self.holding = None

    def _open(self, d: float, stren: float) -> None:
        ep, c = self.p.exec_params, self.p.costs
        a = float(self.atr[0])
        if not np.isfinite(a) or a <= 0:
            return
        slip, fee = c.slippage_rate, c.fee_rate
        entry_px = float(self.data.open[0]) * (1.0 + slip * d)
        stop_dist = a * ep.atr_stop_mult
        if stop_dist <= 0:
            return
        equity_now = self.cash
        qty = min(equity_now * ep.risk_per_trade / stop_dist,
                  equity_now * ep.max_leverage / entry_px)
        if qty <= 0:
            return
        entry_fee = abs(entry_px * qty) * fee
        self.cash -= entry_fee
        self.holding = {
            "dir": d, "entry_px": entry_px, "raw_entry": float(self.data.open[0]),
            "qty": qty, "stop": entry_px - stop_dist * d,
            "entry_time": self.data.datetime.datetime(0), "entry_fee": entry_fee,
            "risk_equity": equity_now, "best": entry_px, "strength": stren,
        }

    # ─── 봉 하나 (engine 의 Step A → B → C 와 같은 순서) ─────────────

    def next(self) -> None:
        ep = self.p.exec_params
        today = self.data.datetime.date(0)
        if today != self.day:
            self.day = today
            self.day_start_cash = self.cash

        # Step A — 봉 i 시가에서 봉 i-1 신호에 따라 행동한다.
        # **[-1] 로 읽는 것이 요점이다.** 위치 인덱스로 직접 끼워 넣으면 엔진과
        # 같은 off-by-one 을 복제하게 된다. 피드가 맞추게 두어야 독립 검증이다.
        want = float(self.data.direction[-1])
        stren = float(self.data.strength[-1])

        if self.holding is not None and want != self.holding["dir"]:
            self._close(float(self.data.open[0]), "signal")

        if self.holding is None and want != 0.0:
            blocked = ((self.cash - self.day_start_cash)
                       <= -ep.daily_loss_limit * self.day_start_cash)
            if not blocked and stren >= ep.min_strength:
                self._open(want, stren)

        # Step B — 봉 i 내부. 1분봉이 없으므로 **손절만** 본다.
        if self.holding is not None:
            d = self.holding["dir"]
            stop = self.holding["stop"]
            hit = (float(self.data.low[0]) <= stop if d > 0
                   else float(self.data.high[0]) >= stop)
            if hit:
                self._close(stop, "stop")
            elif ep.trailing:
                a = float(self.atr[0])
                if np.isfinite(a) and a > 0:
                    cl = float(self.data.close[0])
                    best = (max(self.holding["best"], cl) if d > 0
                            else min(self.holding["best"], cl))
                    self.holding["best"] = best
                    trail = best - a * ep.atr_stop_mult * d
                    self.holding["stop"] = (max(stop, trail) if d > 0
                                            else min(stop, trail))

    def stop(self) -> None:
        # 마지막 봉에서 열려 있으면 종가로 정리한다 — 엔진과 같다.
        if self.holding is not None:
            self._close(float(self.data.close[0]), "end")


# 비교 항목. 시각은 완전 일치, 숫자는 상대오차 PRICE_TOLERANCE 이내를 요구한다.
COMPARED = ("direction", "entry_time", "entry_price", "exit_time", "exit_price",
            "qty", "net_pnl", "r_multiple", "exit_reason")


def _same(a, b) -> bool:
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return np.isnan(fa) and np.isnan(fb)
        scale = max(abs(fa), abs(fb), 1e-12)
        return abs(fa - fb) / scale <= PRICE_TOLERANCE
    return a == b


def crosscheck(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    atr: pd.Series,
    *,
    interval: str,
    costs: Costs,
    params: ExecParams,
    initial_equity: float = 10_000.0,
    warmup: int = WARMUP_BARS,
    symbol: str = "",
    strategy: str = "",
    atr_period: int = 14,
) -> CrossCheck:
    """
    같은 입력을 엔진과 backtrader 에 각각 먹이고 거래 목록을 대조한다.

    **워밍업 구간의 신호를 양쪽 다 지운다.** backtrader 는 ATR 이 데워질
    때까지 next() 를 부르지 않으므로, 그냥 돌리면 엔진만 앞에서 거래를 하고
    현금이 달라진 채로 비교가 시작된다. 그러면 모든 거래가 어긋나고 보고서가
    쓸모없어진다. 양쪽에서 같은 구간을 지우면 **같은 현금으로 같은 봉에서**
    출발한다.
    """
    if warmup >= len(df):
        raise ValueError(f"워밍업({warmup})이 봉 수({len(df)})보다 많습니다")

    masked = signals.copy()
    masked.iloc[:warmup, masked.columns.get_indexer(["direction", "strength"])] = 0.0

    eng = run(df, masked, atr, interval=interval, costs=costs, params=params,
              initial_equity=initial_equity, minute_df=None,
              symbol=symbol, strategy=strategy)

    feed = df[["open", "high", "low", "close", "volume"]].copy()
    feed["direction"] = masked["direction"].to_numpy()
    feed["strength"] = masked["strength"].to_numpy()
    if feed.index.tz is not None:
        feed.index = feed.index.tz_localize(None)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(SignalData(dataname=feed))
    cerebro.addstrategy(_Shadow, costs=costs, exec_params=params,
                        initial_equity=initial_equity, atr_period=atr_period)
    shadow = cerebro.run()[0]

    et = eng.trades.copy()
    if not et.empty:
        for col in ("entry_time", "exit_time"):
            s = pd.to_datetime(et[col])
            et[col] = s.dt.tz_localize(None) if s.dt.tz is not None else s
    bt_df = pd.DataFrame(shadow.trades)

    divergences: list[Divergence] = []
    compared = min(len(et), len(bt_df))
    for i in range(compared):
        a, b = et.iloc[i], bt_df.iloc[i]
        for col in COMPARED:
            if not _same(a[col], b[col]):
                divergences.append(Divergence(i, col, a[col], b[col]))

    # 거래 시각에서 tz 를 떼었으므로 기준선도 같은 모양이어야 한다 —
    # 한쪽만 tz 를 달고 있으면 비교 자체가 TypeError 로 터진다.
    return CrossCheck(engine_trades=et, bt_trades=bt_df,
                      divergences=divergences, compared=compared,
                      warmup_cutoff=feed.index[warmup])
