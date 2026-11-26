"""테스트 공용 픽스처."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── 테스트는 실제 계좌에 닿지 않는다 ────────────────────────────
#
# 실제로 있었던 일이다. 엔진을 start() 로 켜는 테스트가 루프 스레드를 남겼고,
# 테스트가 끝나 가짜 브로커가 걷히자 그 스레드가 **진짜 MT5 를 상대로** 28종목
# 한 바퀴를 돌았다(running=True 인 채로). MT5 가 없는 PC 에서는 MT5Error 로
# 끝나 아무도 몰랐고, 터미널이 깔린 PC 에서 전체 테스트가 정시를 넘길 때마다
# 멈춘 것처럼 보이면서 드러났다. 그때 진입 신호가 났다면 테스트가 주문을 냈다.
#
# 두 겹으로 막는다.
#   1) 테스트 세션 전체에서 실제 MT5 모듈을 못 쓰게 한다. 새는 스레드가 생겨도
#      계좌에 닿지 못한다. 가짜 MT5 가 필요한 테스트는 자기가 갈아 끼운다.
#   2) 테스트가 켠 엔진 루프는 그 테스트가 끝날 때 멈춘다.

@pytest.fixture(scope="session", autouse=True)
def _no_real_mt5():
    from src.fx import broker, mt5

    def blocked():
        raise mt5.MT5Error("테스트에서는 실제 MT5 를 쓰지 않는다")

    patch = pytest.MonkeyPatch()
    patch.setattr(mt5, "_mt5", blocked)
    patch.setattr(broker, "_mt5", blocked)     # broker 는 이름을 따로 들고 있다
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def _stop_engine_loops(monkeypatch):
    from src.fx import engine, keepawake

    # 엔진 루프는 도는 동안 PC 의 절전을 막는다. 테스트가 이 PC 의 전원 상태에
    # 손대지 않게 한다 — 요청이 됐다고만 답한다.
    monkeypatch.setattr(keepawake, "_set", lambda flags: True)
    # 엔진 루프는 계좌 손실 한도를 보려고 입출금 내역을 읽는다. 테스트는 실제 계좌의
    # 내역에 닿지 않는다 — 입금액을 모르는 것으로 두면 한도가 듣지 않는다. 한도를
    # 시험하는 테스트는 자기가 갈아 끼운다.
    monkeypatch.setattr(engine, "cash_flows", lambda *a, **k: {"net_funding": 0.0})
    monkeypatch.setattr(engine.FxTrader, "_seed_cache", None)

    started: list = []
    spawn = engine.FxTrader._spawn

    def tracked(self):
        started.append(self)
        return spawn(self)

    monkeypatch.setattr(engine.FxTrader, "_spawn", tracked)
    yield
    for trader in started:
        trader.state.running = False
        trader._stop.set()
        thread = trader._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)


def make_candles(
    n: int, interval: str = "5min", start: str = "2026-01-01", seed: int = 0,
    derivatives: bool = True,
) -> pd.DataFrame:
    """
    검증용 합성 캔들. 값이 인덱스에서 결정론적으로 나오도록 만들어서
    집계 결과를 손으로 계산해 검증할 수 있게 한다.

    derivatives=True 면 파생 시장 열(미결제약정·롱숏비율·펀딩)도 붙인다.
    인과성 검사가 레지스트리 전체를 훑기 때문에, 이 열이 없으면 파생 지표 9종이
    검사에서 빠지는 게 아니라 입력 부족으로 **에러**가 난다. 기본을 True 로 둔 이유다.
    derivatives=False 는 '파생 데이터가 없을 때도 파이프라인이 도는가'를 볼 때 쓴다.

    spread 도 같은 이유로 기본 포함이다 — MT5(FX) 봉에는 있고 암호화폐 봉에는
    없는 열이라, 없을 때 건너뛰는지는 test_orderflow.py 에서 따로 본다.
    """
    idx = pd.date_range(start, periods=n, freq=interval, tz="UTC", name="open_time")
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.3, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.3, n)
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.uniform(1, 10, n),
            "quote_volume": rng.uniform(100, 1000, n),
            "trades": rng.integers(10, 100, n),
            "taker_buy_base": rng.uniform(0.5, 5, n),
            # FX(MT5) 봉에만 있는 열. 인과성 검사가 레지스트리 전체를 훑기
            # 때문에 이게 없으면 spread_z 가 검사에서 빠지는 게 아니라 입력
            # 부족으로 **에러**가 난다 (파생 열과 같은 이유).
            "spread": rng.integers(15, 40, n).astype("float64"),
        },
        index=idx,
    )
    if derivatives:
        df = df.join(_make_derivatives(idx, close, rng, interval))
    return df


def _make_derivatives(idx, close, rng, interval: str) -> pd.DataFrame:
    """
    파생 열의 합성판. 실제 데이터의 **구조**를 흉내 내는 것이 중요하다.

    특히 펀딩은 8시간마다 갱신되어 그 사이 봉에서는 같은 값이 반복된다. 이 계단
    구조가 없으면 롤링 표준편차가 실제보다 커져서 funding_zscore 가 실제와 전혀
    다르게 동작하고, 0으로 나누는 경로를 테스트가 밟지 못한다.
    """
    n = len(idx)
    oi = 10_000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    step_hours = pd.Timedelta(pd.tseries.frequencies.to_offset(interval)).total_seconds() / 3600
    bars_per_funding = max(1, int(round(8 / step_hours))) if step_hours else 1
    n_blocks = n // bars_per_funding + 1
    funding = np.repeat(rng.normal(0.0001, 0.0002, n_blocks), bars_per_funding)[:n]
    return pd.DataFrame(
        {
            "open_interest": oi,
            "open_interest_value": oi * close,
            "top_trader_account_ls": np.exp(rng.normal(0.4, 0.15, n)),
            "top_trader_position_ls": np.exp(rng.normal(0.1, 0.2, n)),
            "global_account_ls": np.exp(rng.normal(0.6, 0.2, n)),
            "taker_ls_ratio": np.exp(rng.normal(0.0, 0.35, n)),
            "funding_rate": funding,
        },
        index=idx,
    )


@pytest.fixture
def candles_5m() -> pd.DataFrame:
    return make_candles(288, "5min")  # 하루치
