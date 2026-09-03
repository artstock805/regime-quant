"""
파생 시장 지표 — 캔들이 말해 주지 않는 것.

캔들은 '가격이 어디로 갔는가'만 담는다. 같은 상승이라도 신규 롱이 쌓이며 오른
것과 숏이 청산당하며 오른 것은 이후 전개가 다르다. 그 구분은 미결제약정(OI)이
있어야 가능하고, OI 는 캔들 안에 없다.

이 모듈을 만든 이유는 분명하다. 캔들 기반 특징 67개가 전부 |IC|<0.05 였고,
총우위 +0.0165R 을 비용 0.0252R 이 152% 먹었다. **우위 자체를 캔들 밖에서 가져오지
못하면 이 시스템은 끝난다.** 그러니 여기서 나오는 답이 "여기도 없다"여도 그것은
성공한 실험이다 — 남은 길이 하나 줄어든 것이기 때문이다.

입력 열은 `src/data/derivatives.py` 가 봉에 붙여 준다. 없으면 compute_all 이
조용히 건너뛰므로, 파생 데이터를 안 받은 캔들로도 파이프라인은 그대로 돈다.

모두 인과적이다 — t 시점 값은 t 까지의 데이터만 쓴다. `tests/test_causality.py`
가 새 지표를 자동으로 포함하므로 미래 참조를 들고 들어오면 즉시 걸린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import sma, zscore
from .registry import register

OI = ("open_interest",)
OI_PRICE = ("close", "open_interest")


# ─────────────────────────────────────────────────────────────────
#  미결제약정 — 포지션이 쌓이는가 풀리는가
# ─────────────────────────────────────────────────────────────────

@register("oi_change", "derivatives", {"lookback": 24}, needs=OI)
def oi_change(df: pd.DataFrame, lookback: int = 24) -> pd.Series:
    """OI 변화율. 절대 계약수는 종목마다 규모가 달라 비교가 안 된다."""
    oi = df["open_interest"]
    return oi.pct_change(lookback)


@register("oi_zscore", "derivatives", {"period": 168}, needs=OI)
def oi_zscore(df: pd.DataFrame, period: int = 168) -> pd.Series:
    """OI 가 자기 최근 분포에서 어디쯤인가. 1주(168시간) 기준."""
    return zscore(df["open_interest"], period)


@register(
    "oi_price_regime", "derivatives", {"lookback": 24},
    outputs=("oi_price_regime", "long_buildup", "short_buildup"),
    needs=OI_PRICE,
)
def oi_price_regime(df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """
    OI 와 가격의 조합. 선물 시장을 읽는 고전적인 네 칸이다.

        가격↑ OI↑   신규 롱 유입      (+1)  추세에 돈이 들어온다
        가격↓ OI↑   신규 숏 유입      (-1)  하락에 돈이 들어온다
        가격↑ OI↓   숏 청산(숏스퀴즈)  (+0.5) 연료가 소진되는 상승
        가격↓ OI↓   롱 청산            (-0.5) 연료가 소진되는 하락

    청산 구간에 절반 점수를 주는 이유는 방향은 같지만 지속성이 다르기 때문이다.
    포지션이 풀리며 나온 움직임은 새 돈이 없어서 이어지기 어렵다.
    """
    dp = df["close"].pct_change(lookback)
    doi = df["open_interest"].pct_change(lookback)
    up, oi_up = dp > 0, doi > 0

    score = pd.Series(np.nan, index=df.index, name="oi_price_regime")
    valid = dp.notna() & doi.notna()
    score[valid & up & oi_up] = 1.0
    score[valid & ~up & oi_up] = -1.0
    score[valid & up & ~oi_up] = 0.5
    score[valid & ~up & ~oi_up] = -0.5

    return pd.DataFrame({
        "oi_price_regime": score,
        "long_buildup": (valid & up & oi_up).astype(float).where(valid),
        "short_buildup": (valid & ~up & oi_up).astype(float).where(valid),
    })


# ─────────────────────────────────────────────────────────────────
#  펀딩 — 쏠림의 가격
# ─────────────────────────────────────────────────────────────────

@register("funding_zscore", "derivatives", {"period": 90}, needs=("funding_rate",))
def funding_zscore(df: pd.DataFrame, period: int = 90) -> pd.Series:
    """
    펀딩이 자기 분포에서 얼마나 극단인가.

    펀딩은 8시간마다 갱신되므로 1시간봉에서는 같은 값이 8번 반복된다. 그래서
    period 를 봉 수가 아니라 **갱신 횟수**로 생각해야 한다. 90봉 ≈ 11회 갱신 ≈ 3.7일.
    """
    return zscore(df["funding_rate"], period)


@register("funding_cum", "derivatives", {"lookback": 72}, needs=("funding_rate",))
def funding_cum(df: pd.DataFrame, lookback: int = 72) -> pd.Series:
    """
    누적 펀딩. 한 번 튀는 것보다 **오래 한쪽이 비용을 내고 있는 상태**가 의미 있다.

    1시간봉에서 펀딩 값이 8번씩 반복되므로 단순 합은 8배 부풀려진다. 평균을 쓰고
    구간 길이를 곱해 실제 누적 비용에 비례하게 맞춘다.
    """
    return df["funding_rate"].rolling(lookback, min_periods=lookback).mean() * (lookback / 8.0)


# ─────────────────────────────────────────────────────────────────
#  포지셔닝 — 누가 어느 쪽에 서 있는가
# ─────────────────────────────────────────────────────────────────

@register("perp_taker_ls", "derivatives", {"period": 24}, needs=("taker_ls_ratio",))
def perp_taker_ls(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """
    테이커 매수/매도 비율을 대칭 지표로. 비율 1.0 이 중립이라 로그를 씌운다 —
    2배 매수(+0.69)와 2배 매도(-0.69)가 같은 크기가 되어야 평균이 왜곡되지 않는다.

    캔들 쪽 `taker_imbalance` 와 이름이 비슷하지만 소스가 다르다. 저쪽은 봉마다
    오는 taker_buy_base 의 비중이고, 이쪽은 metrics 의 5분 집계 비율이다.
    둘 다 남겨 둔 이유는 표본 방식이 달라 서로 다른 잡음을 갖기 때문이다.
    """
    ratio = df["taker_ls_ratio"].replace(0, np.nan)
    return sma(np.log(ratio), period)


@register("crowd_vs_smart", "derivatives", {"period": 24},
          needs=("global_account_ls", "top_trader_position_ls"))
def crowd_vs_smart(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """
    군중(전체 계정)과 큰손(상위 트레이더 포지션)이 갈리는 정도.

    양수면 개인이 롱 쪽으로 큰손보다 더 쏠려 있다는 뜻이다. 둘 다 비율이라
    로그 차이로 재야 방향이 대칭이 된다.
    """
    crowd = np.log(df["global_account_ls"].replace(0, np.nan))
    smart = np.log(df["top_trader_position_ls"].replace(0, np.nan))
    return sma(crowd - smart, period)


@register("top_trader_tilt", "derivatives", {"period": 168},
          needs=("top_trader_position_ls",))
def top_trader_tilt(df: pd.DataFrame, period: int = 168) -> pd.Series:
    """상위 트레이더 포지션 쏠림이 자기 최근 분포에서 어디쯤인가."""
    return zscore(np.log(df["top_trader_position_ls"].replace(0, np.nan)), period)


@register("account_position_gap", "derivatives", {"period": 24},
          needs=("top_trader_account_ls", "top_trader_position_ls"))
def account_position_gap(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """
    같은 상위 트레이더를 계정 수로 셀 때와 포지션 크기로 잴 때의 차이.

    계정 기준은 롱이 많은데 포지션 기준은 숏이 크다면, **수는 적지만 큰 자금이
    반대편에 서 있다**는 뜻이다. 이 괴리는 두 값을 따로 봐서는 보이지 않는다.
    """
    acct = np.log(df["top_trader_account_ls"].replace(0, np.nan))
    posn = np.log(df["top_trader_position_ls"].replace(0, np.nan))
    return sma(acct - posn, period)
