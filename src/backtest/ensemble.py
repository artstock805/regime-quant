"""
레짐별 가중 앙상블.

두 단계로 나눈다. 한 번에 하면 순환이 된다 — 가중치를 얻으려면 성과가 필요하고
성과를 얻으려면 가중치가 필요하기 때문이다.

    1단계  각 전략을 **단독으로** 돌려 레짐별 성과를 얻는다  (= 전략 리더보드)
    2단계  그 성과에서 가중치를 뽑아 앙상블을 돌린다

walk-forward에서는 1단계를 **학습 구간**에서만 하고 2단계를 **검증 구간**에서 한다.
이렇게 하지 않으면 가중치가 자기가 평가할 구간의 성과를 이미 알고 있는 상태가 되어
수익률이 통째로 거짓이 된다.

가중치 산정에는 **R배수 기대값**을 쓴다. 총수익률로 매기면 노출이 큰 전략이 무조건
이기고, 승률로 매기면 작게 여러 번 이기고 크게 한 번 지는 전략이 이긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..regime import ALL_REGIMES, KOREAN, Regime


@dataclass(frozen=True)
class WeightParams:
    # 해당 레짐에서 이만큼 거래가 없으면 표본이 부족하다고 보고 발언권을 주지 않는다
    min_trades: int = 20
    # 기대 R배수가 이 값 이하면 가중치 0 (발언권 박탈)
    min_expectancy_r: float = 0.0
    # 한 전략이 독식하지 못하게 하는 상한
    max_weight: float = 0.45
    # 앙상블 합산 점수가 이 값을 넘어야 진입한다
    entry_threshold: float = 0.35
    # 확신도를 **자기 최근 분포의 백분위**로 바꿔서 문턱과 맞출지.
    #
    # 왜 필요한가. 전략은 확신도를 자기 마음대로 0~1 로 만든다. taker_flow_momentum
    # 은 scale(|z|, 0.15, 2.0) 을 쓰는데 실측 중위가 0.104 다. 그러면 문턱 0.35 는
    # 실질적으로 |z|>=0.80 을 요구하고, 전략이 선언한 진입 조건(|z|>=0.15)과 **5배
    # 어긋난다.** 신호의 85~89% 가 설계 의도와 무관하게 버려진다.
    #
    # 백분위로 바꾸면 문턱이 "내 최근 신호 중 상위 몇 %인가"라는 같은 뜻을 갖는다.
    # 전략이 달라도, 타임프레임이 달라도 같은 잣대가 된다.
    #
    # 기본을 False 로 둔 이유: 켜면 결과가 바뀌므로 **새로운 시험 한 번**이다.
    # Deflated Sharpe 가 그만큼 깎인다. 둘을 나란히 재고 고르기 위해 옵션으로 둔다.
    calibrate_strength: bool = False
    # 백분위를 낼 때 참고할 과거 신호 개수. 과거만 본다.
    calibration_window: int = 500


def calibrated_strength(sig: pd.DataFrame, window: int = 500) -> pd.Series:
    """
    확신도를 **자기 최근 신호 분포에서의 백분위**로 바꾼다.

    전략마다 확신도 척도가 다르면 하나의 문턱이 전략마다 다른 뜻이 된다.
    실측: taker_flow_momentum 확신도 중위 0.104 → 문턱 0.35 는 상위 15% 만 통과.
    같은 문턱이 다른 전략에서는 상위 60% 를 통과시킬 수도 있다. 비교가 성립하지 않는다.

    백분위로 바꾸면 문턱 0.35 는 어디서나 "내 최근 신호 중 상위 65% 이내"를 뜻한다.

    **인과성**: `rolling(window).rank(pct=True)` 는 창이 현재 봉에서 끝나므로 과거와
    현재만 본다. 미래 값은 들어오지 않는다. 무신호 봉(direction==0)은 분포에서 빼고
    계산한다 — 0 이 대부분인 분포에 백분위를 매기면 신호 봉이 전부 상위로 몰린다.
    """
    active = sig["direction"] != 0
    conf = sig["strength"].where(active)
    ranked = conf.dropna().rolling(window, min_periods=20).rank(pct=True)
    # 표본이 모일 때까지는 원래 확신도를 쓴다. NaN 으로 두면 초기 구간이 통째로 죽는다.
    out = ranked.reindex(sig.index)
    return out.fillna(sig["strength"]).where(active, 0.0)


def score_by_regime(trades: pd.DataFrame, params: WeightParams | None = None) -> pd.DataFrame:
    """
    전략 × 레짐별 성과 표. 상황판의 전략 리더보드가 그대로 이 표다.

    trades 에는 여러 전략의 거래가 섞여 있어도 된다 (strategy / regime 열로 구분).
    """
    p = params or WeightParams()
    if trades.empty:
        return pd.DataFrame(columns=["strategy", "regime", "trades", "expectancy_r",
                                     "win_rate", "total_r", "eligible"])

    rows = []
    for (strat, reg), g in trades.groupby(["strategy", "regime"], dropna=True):
        exp_r = float(g["r_multiple"].mean())
        rows.append({
            "strategy": strat,
            "regime": reg,
            "trades": len(g),
            "expectancy_r": exp_r,
            "win_rate": float((g["net_pnl"] > 0).mean()),
            "total_r": float(g["r_multiple"].sum()),
            "eligible": bool(len(g) >= p.min_trades and exp_r > p.min_expectancy_r),
        })
    return pd.DataFrame(rows).sort_values(["regime", "expectancy_r"], ascending=[True, False])


def derive_weights(
    scores: pd.DataFrame,
    strategy_regimes: dict[str, tuple[Regime, ...]],
    params: WeightParams | None = None,
) -> pd.DataFrame:
    """
    레짐 × 전략 가중치 행렬. 각 레짐의 가중치 합은 1(발언권 있는 전략이 없으면 0).

    strategy_regimes 는 전략이 스스로 선언한 유효 국면이다. 선언하지 않은 국면에서는
    성과가 좋아도 가중치를 주지 않는다 — 우연히 맞은 것을 실력으로 오인하지 않기 위해.
    """
    p = params or WeightParams()
    strategies = sorted(strategy_regimes)
    matrix = pd.DataFrame(
        0.0, index=[str(r) for r in ALL_REGIMES], columns=strategies, dtype="float64"
    )
    if scores.empty:
        return matrix

    for reg in matrix.index:
        rows = scores[(scores["regime"] == reg) & scores["eligible"]]
        raw: dict[str, float] = {}
        for _, row in rows.iterrows():
            strat = row["strategy"]
            declared = strategy_regimes.get(strat, ())
            if not any(str(x) == reg for x in declared):
                continue          # 선언하지 않은 국면은 건너뛴다
            raw[strat] = max(0.0, float(row["expectancy_r"]))

        total = sum(raw.values())
        if total <= 0:
            continue

        # 상한이 실현 불가능하면 완화한다. 적격 전략이 2개인데 상한이 0.45면
        # 합의 최대가 0.90 이라 정규화가 성립하지 않는다. 그대로 두면 그 레짐의
        # 확신도가 조용히 깎여서 진입 문턱을 못 넘게 된다.
        cap = max(p.max_weight, 1.0 / len(raw))

        w = {k: v / total for k, v in raw.items()}
        # 물채우기(water-filling): 초과분을 '남은 여유'에 비례해 나눈다.
        # 현재 가중치에 비례해 나누면 상한을 다시 넘겨 진동한다.
        for _ in range(50):
            over = [k for k, v in w.items() if v > cap + 1e-12]
            if not over:
                break
            excess = sum(w[k] - cap for k in over)
            for k in over:
                w[k] = cap
            room = {k: cap - w[k] for k in w if k not in over}
            room_total = sum(room.values())
            if room_total <= 1e-12:
                break
            for k, r in room.items():
                w[k] += excess * r / room_total

        for k, v in w.items():
            matrix.loc[reg, k] = v

    return matrix


def combine(
    signals: dict[str, pd.DataFrame],
    regimes: pd.Series,
    weights: pd.DataFrame,
    params: WeightParams | None = None,
) -> pd.DataFrame:
    """
    전략별 신호를 레짐 가중치로 합산해 하나의 direction/strength 로 만든다.

    각 전략의 기여 = 가중치 × 확신도 × 방향.
    합산 점수의 절대값이 문턱을 넘으면 그 방향으로 진입한다.
    """
    p = params or WeightParams()
    if not signals:
        raise ValueError("signals 가 비어 있습니다")

    index = next(iter(signals.values())).index
    reg = regimes.reindex(index).astype(object)

    score = pd.Series(0.0, index=index)
    for strat, sig in signals.items():
        if strat not in weights.columns:
            continue
        if not sig.index.equals(index):
            raise ValueError(f"{strat}: 신호 인덱스가 다릅니다")
        # 각 봉의 레짐에 해당하는 가중치를 끌어온다
        w = reg.map(lambda r: weights.at[str(r), strat] if pd.notna(r) else 0.0).astype("float64")
        conf = (calibrated_strength(sig, p.calibration_window)
                if p.calibrate_strength else sig["strength"])
        score = score + w * conf * sig["direction"]

    direction = pd.Series(0.0, index=index)
    direction[score >= p.entry_threshold] = 1.0
    direction[score <= -p.entry_threshold] = -1.0

    return pd.DataFrame({
        "direction": direction,
        "strength": score.abs().clip(0.0, 1.0).where(direction != 0, 0.0),
        "score": score,
    })


def leaderboard(
    scores: pd.DataFrame,
    weights: pd.DataFrame,
    regime: Regime | str,
    korean_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    특정 레짐의 전략 리더보드. 상황판 중앙 패널이 이 표를 그대로 그린다.
    가중치 0 인 전략은 '제외'로 남겨 둔다 — 왜 빠졌는지 보여야 하기 때문에.
    """
    reg = str(regime)
    rows = []
    sub = scores[scores["regime"] == reg]
    for strat in weights.columns:
        row = sub[sub["strategy"] == strat]
        w = float(weights.at[reg, strat]) if reg in weights.index else 0.0
        if row.empty:
            rows.append({"전략": korean_names.get(strat, strat) if korean_names else strat,
                         "거래": 0, "기대R": np.nan, "승률_%": np.nan,
                         "가중치": w, "상태": "표본없음"})
            continue
        r = row.iloc[0]
        rows.append({
            "전략": korean_names.get(strat, strat) if korean_names else strat,
            "거래": int(r["trades"]),
            "기대R": round(float(r["expectancy_r"]), 3),
            "승률_%": round(float(r["win_rate"]) * 100, 1),
            "가중치": round(w, 3),
            "상태": "참여" if w > 0 else ("표본부족" if r["trades"] < 20 else "제외"),
        })
    out = pd.DataFrame(rows).sort_values("가중치", ascending=False).reset_index(drop=True)
    out.attrs["regime"] = KOREAN.get(Regime(reg), reg) if reg in {str(x) for x in ALL_REGIMES} else reg
    return out
