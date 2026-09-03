"""
지표 성적표 — 어떤 보조지표가 값을 하는지 정기적으로 재고, 그 순위를 믿을 수
있는지까지 함께 낸다.

**이 파일은 순위표를 만들면서 동시에 그 순위표를 의심하도록 설계됐다.**
이유가 있다. 96개 특징을 수익성으로 줄 세워 1등을 고르면 그건 시험 96회다.
이 프로젝트의 Deflated Sharpe 는 시험 50회 기준 이미 0.33 이다(기준 0.95).
그리고 같은 종류의 순위가 실제로 쓸모없다는 것을 이미 측정했다 — 종목 성적의
분기별 순위 지속성이 +0.024(t=0.36) 였고, "직전 분기 최악을 뺀다"는 규칙은
기대값을 -0.0087 에서 -0.0113 으로 **악화**시켰다.

그래서 이 모듈이 내놓는 것은 세 가지다.
    1) 순위      — 지표별 기간외(out-of-sample) 기대R
    2) 신뢰도    — 다중검정 보정 후에도 유의한 지표가 몇 개인가
    3) **지속성** — 이번 기간의 순위가 다음 기간에도 유지되는가

3번이 0 이면 1번은 장식이다. 그걸 숨기지 않는다.

방향을 어디서 얻는가 (미래 참조가 생기는 지점)
    지표값이 높을 때 오르는지 내리는지를 **전 구간으로 정하면 그게 미래 참조다.**
    기간을 나누고, 각 기간의 방향은 **직전 기간**에서만 배운다. 그러면 채점이
    자동으로 기간외가 되고, 동시에 지속성이 측정된다 — 방향이 유지되지 않는
    지표는 점수가 저절로 나빠진다.

기대R 은 **근사값**이다
    엔진처럼 봉 내부에서 손절·목표 선착을 판정하지 않는다. 지평 h 봉 뒤 종가로
    청산했다고 보고 1R(=atr_stop_mult x ATR)로 나눈다. 그래서 **절대 수준은
    실제보다 낙관적**이다(손절에 걸리는 경로를 무시한다). 다만 모든 지표에 같은
    잣대를 쓰므로 **순위 비교에는 쓸 수 있다.** 최종 판정은 언제나 walk-forward 다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, holm, t_test

# 가격 수준에 가까워 '예측력'이 아니라 '가격과 같은 것'을 재게 되는 열.
# screen_features.py 와 같은 기준을 쓴다 — 두 곳에서 달라지면 비교가 깨진다.
LEVEL_LIKE = frozenset({
    "dc_upper", "dc_lower", "dc_mid", "kc_upper", "kc_lower",
    "bb_upper", "bb_lower", "st_line", "psar", "tenkan", "kijun",
    "obv", "atr", "momentum", "macd", "macd_signal", "awesome_osc", "kst",
})


@dataclass
class ScoreboardConfig:
    horizon: int = 4                 # 몇 봉 뒤 종가로 채점하는가
    atr_stop_mult: float = 4.0       # 1R 정의 — 엔진과 같아야 비교가 성립한다
    cost_r: float = 0.0242           # 실측 왕복 비용(R 단위). 129종목 1시간봉 기준
    entry_quantile: float = 0.85     # 신호로 볼 |z| 분위 (상위 15%)
    z_period: int = 168              # 표준화 창 — 과거만 본다
    period_freq: str = "Q"           # 기간 분할 단위 (Q=분기, M=월, 2Q=반년)
    min_period_obs: int = 500        # 이보다 적은 기간은 채점하지 않는다
    min_direction_obs: int = 500     # 방향 학습에 필요한 최소 관측


@dataclass
class IndicatorScore:
    name: str
    periods: int
    obs: int
    expectancy_r: float
    t: float
    p: float
    p_holm: float
    p_bh: float
    positive_periods: int
    direction_flips: int             # 방향이 기간마다 몇 번 뒤집혔는가
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _causal_z(s: pd.Series, period: int) -> pd.Series:
    """자기 최근 분포로 표준화. 창이 현재 봉에서 끝나므로 과거만 본다."""
    mu = s.rolling(period, min_periods=period).mean()
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def build_panel(
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    cfg: ScoreboardConfig | None = None,
) -> pd.DataFrame:
    """
    종목별 (봉, 특징) 을 받아 채점용 긴 표로 만든다.

    frames: {종목: (bars, feats)}
    반환 열: symbol · time · period · r_unit · fwd · <특징 z 값들>

    r_unit 은 이 봉의 1R 을 가격 단위로 담는다. 종목·시기마다 변동성이 다르므로
    수익률을 그대로 비교하면 변동성 큰 종목이 무조건 이긴다. R 로 나눠야 비교된다.
    """
    c = cfg or ScoreboardConfig()
    parts = []
    for sym, (bars, feats) in frames.items():
        if bars.empty or feats.empty:
            continue
        cols = [x for x in feats.columns if x not in LEVEL_LIKE]
        if not cols:
            continue
        atr = feats.get("atr")
        if atr is None:
            raise KeyError("atr 열이 필요합니다 — 1R 을 정의할 수 없습니다")

        block = pd.DataFrame(index=bars.index)
        block["symbol"] = sym
        block["r_unit"] = (atr * c.atr_stop_mult).replace(0, np.nan)
        # 미래수익률. **채점 라벨로만 쓴다.** 신호로 쓰면 그게 미래 참조다.
        block["fwd"] = bars["close"].shift(-c.horizon) - bars["close"]
        for col in cols:
            block[col] = _causal_z(feats[col], c.z_period)
        parts.append(block)

    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts)
    panel = panel.dropna(subset=["r_unit", "fwd"])
    panel["time"] = panel.index
    # to_period 는 'QE' 를 받지 않고(리샘플 전용) 타임존도 버리며 경고한다.
    # 전 구간이 UTC 로 통일돼 있으므로 먼저 떼어 내고, 리샘플 표기는 기간 표기로 옮긴다.
    freq = c.period_freq.upper().replace("QE", "Q").replace("ME", "M").replace("YE", "Y")
    panel["period"] = panel.index.tz_localize(None).to_period(freq).astype(str)
    return panel.reset_index(drop=True)


def score_periods(panel: pd.DataFrame, cfg: ScoreboardConfig | None = None) -> pd.DataFrame:
    """
    기간별 지표 점수. **방향은 직전 기간에서만 배운다.**

    반환 열: period · indicator · obs · expectancy_r · direction
    direction 은 그 기간에 쓴 방향(+1 순추종 / -1 역추종)이다. 기간마다 뒤집히면
    그 지표는 방향조차 불안정하다는 뜻이고, 순위 상단에 있어도 믿을 수 없다.
    """
    c = cfg or ScoreboardConfig()
    if panel.empty:
        return pd.DataFrame(columns=["period", "indicator", "obs",
                                     "expectancy_r", "direction"])

    indicators = [x for x in panel.columns
                  if x not in {"symbol", "time", "period", "r_unit", "fwd"}]
    periods = sorted(panel["period"].unique())
    r_all = panel["fwd"] / panel["r_unit"]

    rows = []
    for prev, cur in zip(periods, periods[1:]):
        prev_mask = panel["period"] == prev
        cur_mask = panel["period"] == cur
        if cur_mask.sum() < c.min_period_obs:
            continue
        for ind in indicators:
            z_prev, z_cur = panel.loc[prev_mask, ind], panel.loc[cur_mask, ind]
            # ── 방향 학습: 직전 기간에서만
            ok_prev = z_prev.notna() & r_all[prev_mask].notna()
            if ok_prev.sum() < c.min_direction_obs:
                continue
            lead = float(np.sign(np.corrcoef(z_prev[ok_prev], r_all[prev_mask][ok_prev])[0, 1])
                         ) if z_prev[ok_prev].std() > 0 else 0.0
            if lead == 0.0 or not np.isfinite(lead):
                continue

            # ── 채점: 현재 기간. 문턱은 직전 기간 분포에서 정한다(미래 참조 방지)
            thresh = float(z_prev[ok_prev].abs().quantile(c.entry_quantile))
            ok_cur = z_cur.notna() & r_all[cur_mask].notna()
            fire = ok_cur & (z_cur.abs() >= thresh)
            if fire.sum() < 30:
                continue
            direction = lead * np.sign(z_cur[fire])
            r = direction * r_all[cur_mask][fire] - c.cost_r
            rows.append({"period": cur, "indicator": ind, "obs": int(fire.sum()),
                         "expectancy_r": float(r.mean()), "direction": lead})
    return pd.DataFrame(rows)


def aggregate(per_period: pd.DataFrame, panel: pd.DataFrame | None = None,
              cfg: ScoreboardConfig | None = None) -> list[IndicatorScore]:
    """
    기간 점수를 지표별로 합쳐 순위를 만든다. 다중검정 보정을 반드시 적용한다.

    지표 96개를 각각 검정하면 5% 유의수준에서 **우연히 5개쯤은 통과한다.**
    보정 없는 순위표는 그 우연을 발견으로 착각하게 만든다.
    """
    if per_period.empty:
        return []
    stats: dict[str, dict] = {}
    for ind, g in per_period.groupby("indicator"):
        vals = g["expectancy_r"].to_numpy()
        # 기간 단위로 검정한다. 거래 단위로 하면 같은 기간의 관측이 서로
        # 독립이 아니라서 t 가 부풀고, 그건 이미 군집 검정에서 확인한 문제다.
        tt = t_test(vals, min_n=3)
        stats[ind] = {
            "periods": len(vals), "obs": int(g["obs"].sum()),
            "expectancy_r": float(vals.mean()),
            "t": tt.t if np.isfinite(tt.t) else 0.0,
            "p": tt.p if np.isfinite(tt.p) else 1.0,
            "positive_periods": int((vals > 0).sum()),
            "direction_flips": int((g.sort_values("period")["direction"]
                                    .diff().fillna(0) != 0).sum()),
        }
    pvals = {k: v["p"] for k, v in stats.items()}
    hp, bh = holm(pvals), benjamini_hochberg(pvals)
    out = [
        IndicatorScore(name=k, p_holm=hp[k], p_bh=bh[k],
                       significant=bool(hp[k] < 0.05 and v["t"] > 0), **v)
        for k, v in stats.items()
    ]
    return sorted(out, key=lambda s: s.expectancy_r, reverse=True)


def rank_persistence(per_period: pd.DataFrame, n_perm: int = 2000,
                     seed: int = 0) -> dict:
    """
    **이 성적표에서 가장 중요한 숫자.** 이번 기간 순위가 다음 기간에도 유지되는가.

    유지되지 않으면 순위표는 장식이다. 종목 순위에서 이미 겪었다 —
    분기 상관 +0.024(t=0.36) 였고, 그 순위를 믿고 최악을 뺐더니 성적이 나빠졌다.

    귀무분포는 순열로 낸다(기간 쌍이 적어 근사식보다 정확하다).
    순위를 표준화해 두면 스피어만 상관이 내적 한 번이라, n_perm 회를 다 돌려도 싸다.
    **드로우 수를 아끼면 안 된다** — 10회만 뽑으면 p 해상도가 0.1 이고, 관측이
    그 10개를 넘기만 하면 p=0 이 찍혀 없는 패턴을 찾았다고 말하게 된다(실제로 겪었다).
    """
    if per_period.empty:
        return {"pairs": 0, "mean_rho": float("nan"), "p": float("nan"),
                "verdict": "표본부족"}
    piv = per_period.pivot_table(index="period", columns="indicator",
                                 values="expectancy_r")
    periods = sorted(piv.index)

    def _std_ranks(s: pd.Series) -> np.ndarray:
        # copy=True 가 필요하다. pandas 가 읽기전용 뷰를 줄 수 있어 제자리 연산이 터진다.
        r = np.array(s.rank().to_numpy(dtype="float64"), copy=True)
        r -= r.mean()
        sd = r.std()
        return r / sd if sd > 0 else r

    pairs = []
    for a, b in zip(periods, periods[1:]):
        x, y = piv.loc[a], piv.loc[b]
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue
        xs, ys = _std_ranks(x[ok]), _std_ranks(y[ok])
        if xs.std() == 0 or ys.std() == 0:
            continue
        pairs.append((xs, ys))
    if not pairs:
        return {"pairs": 0, "mean_rho": float("nan"), "p": float("nan"),
                "verdict": "표본부족"}

    rhos = [float(xs @ ys / len(xs)) for xs, ys in pairs]
    mean_rho = float(np.mean(rhos))

    # 귀무분포: 쌍마다 y 를 독립적으로 섞고 쌍 평균을 낸다. n_perm 회 전부 돈다.
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype="float64")
    for k in range(n_perm):
        null[k] = np.mean([float(xs @ rng.permutation(ys) / len(xs))
                           for xs, ys in pairs])
    p = float((np.abs(null) >= abs(mean_rho)).mean())

    verdict = ("순위가 유지된다 — 참고할 값이 있다" if p < 0.05 and mean_rho > 0
               else "순위가 역전된다 — 상위 지표를 따라가면 손해다"
               if p < 0.05 and mean_rho < 0
               else "순위가 무작위와 구분되지 않는다 — 이 표로 지표를 고르면 안 된다")
    return {"pairs": len(rhos), "mean_rho": mean_rho,
            "rhos": [round(r, 4) for r in rhos], "p": p,
            "null_sd": float(null.std()), "n_perm": n_perm, "verdict": verdict}


def top_stability(per_period: pd.DataFrame, k: int = 5) -> dict:
    """
    상위 k 개가 다음 기간에도 상위 k 에 남는 비율.

    순위상관은 전체 순서를 보는데, 실제로 쓸 때 관심은 **1등이 다음에도 1등인가**다.
    무작위면 k/N 이 나온다. 그 기준선과 함께 보고한다.
    """
    if per_period.empty:
        return {"pairs": 0, "hit_rate": float("nan"), "baseline": float("nan")}
    piv = per_period.pivot_table(index="period", columns="indicator",
                                 values="expectancy_r")
    periods = sorted(piv.index)
    hits, total, n_ind = 0, 0, 0
    for a, b in zip(periods, periods[1:]):
        x, y = piv.loc[a].dropna(), piv.loc[b].dropna()
        common = x.index.intersection(y.index)
        if len(common) < k * 2:
            continue
        n_ind = max(n_ind, len(common))
        top_a = set(x[common].nlargest(k).index)
        top_b = set(y[common].nlargest(k).index)
        hits += len(top_a & top_b)
        total += k
    if not total:
        return {"pairs": 0, "hit_rate": float("nan"), "baseline": float("nan")}
    return {"pairs": total // k, "k": k, "hit_rate": hits / total,
            "baseline": k / n_ind if n_ind else float("nan"),
            "n_indicators": n_ind}


# ─── 스냅샷 (정기 실행용) ────────────────────────────────────────

@dataclass
class Snapshot:
    created: str
    config: dict
    symbols: list[str]
    indicators: list[dict]
    persistence: dict
    top_stability: dict
    trials: int = 0                     # 이 성적표가 소비한 시험 횟수
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def make_snapshot(scores: list[IndicatorScore], per_period: pd.DataFrame,
                  symbols: list[str], cfg: ScoreboardConfig) -> Snapshot:
    pers = rank_persistence(per_period)
    stab = top_stability(per_period)
    notes = [
        "기대R 은 근사값이다 — 봉 내부 손절·목표 선착을 판정하지 않으므로 절대 수준은 "
        "실제보다 낙관적이다. 순위 비교에만 쓴다. 최종 판정은 walk-forward.",
        f"이 성적표는 지표 {len(scores)}개를 동시에 검정한다 = 시험 {len(scores)}회. "
        "Deflated Sharpe 를 볼 때 이 횟수를 더해야 정직하다.",
    ]
    if pers.get("p") is not None and np.isfinite(pers.get("p", np.nan)) and pers["p"] >= 0.05:
        notes.append("**지속성이 확인되지 않았다. 이 순위로 지표를 고르면 안 된다.**")
    return Snapshot(
        created=datetime.now(timezone.utc).isoformat(),
        config=asdict(cfg), symbols=sorted(symbols),
        indicators=[s.as_dict() for s in scores],
        persistence=pers, top_stability=stab,
        trials=len(scores), notes=notes,
    )


def diff_snapshots(prev: dict | None, cur: Snapshot) -> dict:
    """
    지난 스냅샷과 비교. **새로 추가된 지표**와 순위 변동을 뽑는다.

    새 지표는 눈에 띄게 표시해야 한다 — 지표를 추가하고 성적표를 다시 돌리는 것은
    시험 횟수를 늘리는 행위이고, 그 사실이 보이지 않으면 조용히 과적합한다.
    """
    cur_names = [i["name"] for i in cur.indicators]
    if not prev:
        return {"is_first": True, "new": cur_names, "gone": [], "moves": [],
                "trials_before": 0, "trials_after": cur.trials}

    prev_names = [i["name"] for i in prev.get("indicators", [])]
    prev_rank = {n: i for i, n in enumerate(prev_names, 1)}
    cur_rank = {n: i for i, n in enumerate(cur_names, 1)}
    moves = [
        {"name": n, "before": prev_rank[n], "after": cur_rank[n],
         "delta": prev_rank[n] - cur_rank[n]}
        for n in cur_names if n in prev_rank
    ]
    moves.sort(key=lambda m: -abs(m["delta"]))
    return {
        "is_first": False,
        "new": [n for n in cur_names if n not in prev_rank],
        "gone": [n for n in prev_names if n not in cur_rank],
        "moves": moves,
        "trials_before": int(prev.get("trials", 0)),
        "trials_after": int(prev.get("trials", 0)) + cur.trials,
    }
