"""
교차검증 하네스 자체를 검증한다.

**한 번도 실패하지 않는 하네스는 아무것도 증명하지 못한다.** 두 엔진이
일치한다는 보고는, 그 하네스가 어긋남을 실제로 잡아낼 때만 뜻이 있다.
그래서 여기서는 버그를 일부러 심고 **잡히는지**를 본다. 심는 버그는 이
프로젝트가 실제로 겪었던 종류로 고른다 — 지연 한 봉, 부호 뒤집힘, 비용 누락.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# backtrader 가 없으면 이 파일만 건너뛴다 — requirements.txt 에 그렇게 적혀
# 있고, 매매·상황판은 backtrader 없이도 돌아야 한다.
pytest.importorskip("backtrader")

from src.backtest import ExecParams                                 # noqa: E402
from src.backtest import crosscheck as xc_mod                       # noqa: E402
from src.backtest.crosscheck import crosscheck                      # noqa: E402
from src.config import Costs                                        # noqa: E402

WARMUP = 300
EP = ExecParams(risk_per_trade=0.01, atr_stop_mult=2.0, reward_ratio=1.5,
                daily_loss_limit=0.03)
COSTS = Costs(fee_rate=0.0002, slippage_bps=1.0)


def _bars(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """거래가 실제로 나도록 변동이 있는 무작위 보행."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 0.35, n))
    span = np.abs(rng.normal(0, 0.25, n)) + 0.1
    return pd.DataFrame({
        "open": close - rng.normal(0, 0.1, n),
        "high": close + span,
        "low": close - span,
        "close": close,
        "volume": rng.uniform(800, 1200, n),
    }, index=idx)


def _signals(df: pd.DataFrame) -> pd.DataFrame:
    """20봉 이동평균 위/아래로 방향을 뒤집는다. 규칙이 무엇인지는 중요하지 않다."""
    ma = df["close"].rolling(20, min_periods=20).mean()
    d = np.sign(df["close"] - ma).fillna(0.0)
    return pd.DataFrame({"direction": d,
                         "strength": pd.Series(0.6, index=df.index)})


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    from src.indicators.core import atr
    return atr(df, period)


def _run(df=None, signals=None, **kw):
    df = _bars() if df is None else df
    signals = _signals(df) if signals is None else signals
    return crosscheck(df, signals, _atr(df), interval="1h", costs=COSTS,
                      params=EP, warmup=WARMUP, **kw)


# ─── 기준선: 손대지 않으면 일치해야 한다 ──────────────────────────

def test_the_two_engines_agree_on_untouched_input():
    xc = _run()
    assert xc.compared > 10, "거래가 너무 적어 이 검증은 아무것도 말하지 않는다"
    assert xc.agrees, xc.report()


def test_both_engines_start_from_the_same_bar():
    """워밍업 마스킹이 실제로 걸렸는가 — 안 걸리면 현금이 달라져 전부 어긋난다."""
    xc = _run()
    assert (xc.engine_trades["entry_time"] >= xc.warmup_cutoff).all()


# ─── 심은 버그가 잡히는가 ────────────────────────────────────────

def _engine_with(mutate):
    """엔진 호출만 가로채 입력이나 출력을 비튼다. backtrader 쪽은 그대로다."""
    real = xc_mod.run

    def patched(df, signals, atr, **kw):
        return mutate(real, df, signals, atr, kw)
    return patched


def test_a_one_bar_lag_in_the_engine_is_caught(monkeypatch):
    """신호를 한 봉 당겨 먹이면 — 이 프로젝트가 가장 두려워하는 미래 참조다."""
    def mutate(real, df, signals, atr, kw):
        return real(df, signals.shift(-1).fillna(0.0), atr, **kw)
    monkeypatch.setattr(xc_mod, "run", _engine_with(mutate))
    xc = _run()
    assert not xc.agrees, "한 봉 미래 참조를 못 잡았다 — 하네스가 쓸모없다"


def test_a_flipped_slippage_sign_is_caught(monkeypatch):
    """슬리피지를 유리한 쪽으로 붙이면 성적이 조용히 좋아진다."""
    def mutate(real, df, signals, atr, kw):
        kw["costs"] = Costs(fee_rate=COSTS.fee_rate,
                            slippage_bps=-COSTS.slippage_bps)
        return real(df, signals, atr, **kw)
    monkeypatch.setattr(xc_mod, "run", _engine_with(mutate))
    xc = _run()
    assert not xc.agrees, "슬리피지 부호 뒤집힘을 못 잡았다"
    assert any(d.field in ("entry_price", "exit_price", "net_pnl")
               for d in xc.divergences)


def test_a_missing_fee_is_caught(monkeypatch):
    """수수료를 빼먹으면 손익이 부풀려진다 — 비용은 이 프로젝트의 사인(死因)이다."""
    def mutate(real, df, signals, atr, kw):
        kw["costs"] = Costs(fee_rate=0.0, slippage_bps=COSTS.slippage_bps)
        return real(df, signals, atr, **kw)
    monkeypatch.setattr(xc_mod, "run", _engine_with(mutate))
    xc = _run()
    assert not xc.agrees, "수수료 누락을 못 잡았다"


def test_a_wrong_stop_distance_is_caught(monkeypatch):
    """손절 배수가 다르면 청산 시점이 달라진다."""
    def mutate(real, df, signals, atr, kw):
        kw["params"] = ExecParams(risk_per_trade=EP.risk_per_trade,
                                  atr_stop_mult=EP.atr_stop_mult * 1.5,
                                  reward_ratio=EP.reward_ratio,
                                  daily_loss_limit=EP.daily_loss_limit)
        return real(df, signals, atr, **kw)
    monkeypatch.setattr(xc_mod, "run", _engine_with(mutate))
    xc = _run()
    assert not xc.agrees, "손절 거리 오류를 못 잡았다"


def test_a_single_perturbed_exit_price_is_caught(monkeypatch):
    """거래 하나만 아주 조금 틀려도 잡아야 한다 — 큰 어긋남만 잡으면 소용없다."""
    def mutate(real, df, signals, atr, kw):
        res = real(df, signals, atr, **kw)
        if not res.trades.empty:
            res.trades.loc[res.trades.index[0], "exit_price"] *= 1.000001
        return res
    monkeypatch.setattr(xc_mod, "run", _engine_with(mutate))
    xc = _run()
    assert not xc.agrees
    assert xc.first is not None and xc.first.field == "exit_price"


# ─── 보고서가 읽을 만한가 ────────────────────────────────────────

def test_the_report_leads_with_the_first_divergence(monkeypatch):
    """
    거래 하나가 어긋나면 현금이 달라져 그 뒤가 전부 번진다. 보고서가 마지막
    어긋남을 맨 위에 놓으면 사람이 엉뚱한 곳을 파게 된다.
    """
    def mutate(real, df, signals, atr, kw):
        res = real(df, signals, atr, **kw)
        for i in (3, 7, 11):
            if len(res.trades) > i:
                res.trades.loc[res.trades.index[i], "qty"] *= 1.01
        return res
    monkeypatch.setattr(xc_mod, "run", _engine_with(mutate))
    xc = _run()
    assert xc.first is not None and xc.first.index == 3
    assert "첫 어긋남" in xc.report()


def test_the_report_says_so_plainly_when_they_agree():
    assert "일치" in _run().report()


def test_warmup_longer_than_the_data_is_refused():
    df = _bars(100)
    with pytest.raises(ValueError, match="워밍업"):
        crosscheck(df, _signals(df), _atr(df), interval="1h", costs=COSTS,
                   params=EP, warmup=WARMUP)
