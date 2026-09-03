"""
성과 지표.

모든 지표는 **순손익**(수수료·슬리피지 차감 후) 기준이다. 총손익 기준 지표는
전략을 좋게 보이게 만들 뿐이라 아예 계산하지 않는다.

샤프 연율화에는 봉 수를 넘겨야 한다. 5분봉은 연 105,120봉(288×365)이고,
타임프레임을 바꾸면 값이 달라진다 — 기본값을 두지 않는 이유다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BARS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}


@dataclass(frozen=True)
class Metrics:
    trades: int
    net_pnl: float
    total_return: float        # 초기자본 대비 배수 - 1
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float        # 양수 비율 (0.20 = 20% 낙폭)
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float          # 트레이드당 기대 순손익
    avg_hold_bars: float
    exposure: float            # 포지션을 들고 있던 봉 비율
    total_fees: float

    @classmethod
    def zero(cls) -> Metrics:
        """
        거래가 없을 때의 빈 성적.

        위치 인자로 0을 늘어놓으면 필드를 추가할 때마다 조용히 어긋난다
        (실제로 total_fees 를 추가했을 때 인자 개수가 안 맞아 터졌다).
        필드 이름으로 채워서 그 사고를 원천 차단한다.
        """
        return cls(**{f: 0 if f == "trades" else 0.0 for f in cls.__dataclass_fields__})

    def as_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        return [
            f"거래         {self.trades:,}회",
            f"순손익       ${self.net_pnl:,.2f} (수수료 ${self.total_fees:,.2f})",
            f"총수익률     {self.total_return * 100:+.2f}%",
            f"CAGR         {self.cagr * 100:+.2f}%",
            f"샤프         {self.sharpe:.2f}",
            f"소르티노     {self.sortino:.2f}",
            f"최대낙폭     {self.max_drawdown * 100:.2f}%",
            f"손익비(PF)   {self.profit_factor:.2f}",
            f"승률         {self.win_rate * 100:.1f}%",
            f"기대값/거래  ${self.expectancy:+,.2f}",
            f"평균보유     {self.avg_hold_bars:.1f}봉",
            f"시장노출     {self.exposure * 100:.1f}%",
        ]


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((1.0 - equity / peak).max())


def sharpe_ratio(returns: pd.Series, bars_per_year: int) -> float:
    """봉 단위 수익률의 연율화 샤프. 무위험수익률 0 가정."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(bars_per_year))


def sortino_ratio(returns: pd.Series, bars_per_year: int) -> float:
    """하방 변동성만으로 나눈다 — 상방 변동을 위험으로 세지 않는다."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if downside.empty:
        return float("inf") if r.mean() > 0 else 0.0
    dd = float(np.sqrt((downside**2).mean()))
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(bars_per_year))


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    initial_equity: float,
    interval: str,
    bars_in_position: int | None = None,
) -> Metrics:
    bpy = BARS_PER_YEAR.get(interval)
    if bpy is None:
        raise KeyError(f"연간 봉 수를 모르는 인터벌: {interval}")

    n = len(trades)
    if n == 0 or equity.empty:
        return Metrics.zero()

    net = trades["net_pnl"]
    wins, losses = net[net > 0], net[net <= 0]

    bar_returns = equity.pct_change()
    years = len(equity) / bpy
    final = float(equity.iloc[-1])

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return Metrics(
        trades=n,
        net_pnl=float(net.sum()),
        total_return=final / initial_equity - 1.0,
        cagr=(final / initial_equity) ** (1.0 / years) - 1.0 if years > 0 and final > 0 else 0.0,
        sharpe=sharpe_ratio(bar_returns, bpy),
        sortino=sortino_ratio(bar_returns, bpy),
        max_drawdown=max_drawdown(equity),
        profit_factor=pf,
        win_rate=float((net > 0).mean()),
        avg_win=float(wins.mean()) if not wins.empty else 0.0,
        avg_loss=float(losses.mean()) if not losses.empty else 0.0,
        expectancy=float(net.mean()),
        avg_hold_bars=float(trades["hold_bars"].mean()),
        exposure=(bars_in_position / len(equity)) if bars_in_position is not None else 0.0,
        total_fees=float(trades["fees"].sum()),
    )


def metrics_table(named: dict[str, Metrics]) -> pd.DataFrame:
    """여러 결과를 나란히 비교하는 표. 리더보드의 원재료."""
    rows = []
    for name, m in named.items():
        rows.append({
            "전략": name,
            "거래": m.trades,
            "수익률_%": round(m.total_return * 100, 2),
            "샤프": round(m.sharpe, 2),
            "MDD_%": round(m.max_drawdown * 100, 2),
            "PF": round(m.profit_factor, 2) if np.isfinite(m.profit_factor) else np.inf,
            "승률_%": round(m.win_rate * 100, 1),
            "기대값": round(m.expectancy, 2),
            "보유봉": round(m.avg_hold_bars, 1),
            "노출_%": round(m.exposure * 100, 1),
        })
    return pd.DataFrame(rows)


def passes_acceptance(m: Metrics, cfg) -> tuple[bool, list[str]]:
    """
    설정의 합격 기준을 통과했는지와, 떨어진 이유.
    기준을 못 넘으면 기준을 낮추는 게 아니라 전략이나 레짐 정의를 고쳐야 한다.
    """
    reasons = []
    if m.sharpe < cfg.require("acceptance.min_sharpe"):
        reasons.append(f"샤프 {m.sharpe:.2f} < {cfg.require('acceptance.min_sharpe')}")
    if m.max_drawdown > cfg.require("acceptance.max_drawdown"):
        reasons.append(f"MDD {m.max_drawdown*100:.1f}% > {cfg.require('acceptance.max_drawdown')*100:.0f}%")
    if m.trades < cfg.require("acceptance.min_trades"):
        reasons.append(f"거래 {m.trades}회 < {cfg.require('acceptance.min_trades')}회")
    if m.profit_factor < cfg.require("acceptance.min_profit_factor"):
        reasons.append(f"PF {m.profit_factor:.2f} < {cfg.require('acceptance.min_profit_factor')}")
    return (not reasons), reasons
