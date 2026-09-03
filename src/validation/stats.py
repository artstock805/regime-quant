"""
통계 검정 프리미티브 — 검증 도구 전체가 여기 하나를 쓴다.

이 파일이 생긴 이유: 같은 t검정과 부트스트랩이 robustness.py 와 significance.py 에
각각 있었다. 판정 기준이 두 곳에 있으면 언젠가 갈라지고, 그러면 어느 쪽 숫자를
믿어야 할지 알 수 없게 된다.

scipy 를 쓰지 않는다. 필요한 것은 정규분포 CDF/분위수와 이항분포뿐이고,
이분법과 math.comb 로 충분히 정확하다. 의존성이 적을수록 재현이 쉽다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

EULER = 0.5772156649015329


# ─── 분포 ────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """표준정규 분위수. 이분법 — 200회면 배정도 한계까지 수렴한다."""
    if not 0.0 < p < 1.0:
        return float("nan")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binomial_p_at_least(k: int, n: int, prob: float = 0.5) -> float:
    """k개 이상이 성공할 확률 (단측). 부호 검정에 쓴다."""
    if n <= 0:
        return float("nan")
    k = max(0, min(k, n))
    return sum(math.comb(n, i) * prob**i * (1 - prob) ** (n - i) for i in range(k, n + 1))


# ─── 기대값 검정 ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TTest:
    n: int
    mean: float
    sd: float
    se: float
    t: float
    p: float
    ci_lo: float
    ci_hi: float

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.t):
            return "표본부족"
        if self.ci_lo > 0:
            return "양의우위"
        if self.ci_hi < 0:
            return "음의우위"
        return "판정불가"

    @property
    def significant(self) -> bool:
        """0보다 유의하게 큰가. 부트스트랩 신뢰구간 하한으로 판단한다."""
        return bool(np.isfinite(self.ci_lo) and self.ci_lo > 0)

    def trades_needed(self, target_t: float = 1.96) -> int | None:
        """
        관측된 효과크기를 target_t 로 만들려면 몇 거래가 필요한가.

        이 숫자가 현실적으로 도달 불가능하면 그 전략은 '안 통한다'가 아니라
        '통하는지 증명할 수 없다'가 정확한 표현이다.
        """
        if self.mean == 0 or not np.isfinite(self.sd) or self.sd == 0:
            return None
        n = (target_t * self.sd / self.mean) ** 2
        return int(n) if np.isfinite(n) and 0 < n < 1e12 else None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        d["trades_needed"] = self.trades_needed()
        return d


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """
    평균의 부트스트랩 신뢰구간.

    R배수 분포는 정규가 아니다 — 손절에서 잘리고 목표에서 잘려 두 봉우리가 되고
    꼬리가 두껍다. t분포 가정 대신 재표본으로 구간을 잡는다.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)

    # (n_boot, n) 을 한 번에 잡으면 안 된다. 거래 5만 건에 4000회 재표본이면
    # 1.48 GiB 를 한 덩어리로 요구하고 그대로 MemoryError 가 난다(실제로 겪었다).
    # 결과는 같으므로 재표본을 나눠서 돈다. 청크 크기는 한 번에 ~64MB 를 넘지 않게 잡는다.
    per_draw = max(1, int(64 * 1024 * 1024 / (8 * n)))
    means = np.empty(n_boot, dtype="float64")
    done = 0
    while done < n_boot:
        take = min(per_draw, n_boot - done)
        means[done:done + take] = x[rng.integers(0, n, size=(take, n))].mean(axis=1)
        done += take
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def t_test(x: np.ndarray, n_boot: int = 4000, seed: int = 0, min_n: int = 20) -> TTest:
    """평균이 0인지에 대한 검정 + 부트스트랩 구간."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    nan = float("nan")
    if n < min_n:
        return TTest(n, nan, nan, nan, nan, nan, nan, nan)

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    t = mean / se if se > 0 else 0.0
    p = 2.0 * (1.0 - norm_cdf(abs(t)))
    lo, hi = bootstrap_ci(x, n_boot=n_boot, seed=seed)
    return TTest(n, mean, sd, se, t, p, lo, hi)


# ─── 군집 강건 표준오차 ──────────────────────────────────────────

@dataclass
class ClusteredTest:
    """시간 군집을 반영한 평균 검정."""
    n: int
    n_clusters: int
    mean: float
    se_naive: float
    se_clustered: float
    t_naive: float
    t: float
    p: float
    inflation: float          # 순진한 SE 를 몇 배로 키워야 했는가
    effective_n: float        # 독립 관측으로 환산하면 몇 개인가

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.t):
            return "표본부족"
        if self.p < 0.05:
            return "양의우위" if self.mean > 0 else "음의우위"
        return "판정불가"

    def as_dict(self) -> dict:
        return asdict(self)


def clustered_mean_test(
    x: np.ndarray, cluster: np.ndarray, min_clusters: int = 10
) -> ClusteredTest:
    """
    평균이 0인지 검정하되 **같은 군집 안의 관측은 독립이 아니라고 본다**.

    왜 필요한가. 암호화폐는 종목 간 상관이 매우 높다. 같은 시각에 134종목에서
    열린 거래는 사실상 하나의 시장 충격을 134번 센 것에 가깝다. 순진한 t검정은
    이걸 독립 관측 134개로 세므로 t 가 √134 배까지 부풀 수 있다.
    **종목을 늘려서 유의성을 만들어내는 것이 바로 이 착시다.**

    군집을 시간(예: 주 단위)으로 잡으면, 같은 주에 열린 거래는 함께 움직였다고
    보고 그만큼 신뢰구간을 넓힌다.

        Var(mean) = (1/N²) · Σ_g ( Σ_{i∈g} (x_i - mean) )²

    군집 수가 적으면 이 추정 자체가 불안정하므로 유한표본 보정을 넣고,
    min_clusters 미만이면 판정을 포기한다(조용히 틀린 답을 주는 것보다 낫다).
    """
    x = np.asarray(x, dtype="float64")
    cluster = np.asarray(cluster)
    ok = np.isfinite(x)
    x, cluster = x[ok], cluster[ok]
    n = len(x)
    nan = float("nan")
    if n < 20:
        return ClusteredTest(n, 0, nan, nan, nan, nan, nan, nan, nan, nan)

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se_naive = sd / math.sqrt(n) if sd > 0 else 0.0

    codes, inv = np.unique(cluster, return_inverse=True)
    g = len(codes)
    dev = x - mean
    # 군집별 잔차 합
    sums = np.bincount(inv, weights=dev, minlength=g)
    var = float((sums ** 2).sum()) / (n ** 2)
    # 유한 군집 보정 — 군집이 적을수록 분산을 키운다
    if g > 1:
        var *= (g / (g - 1)) * ((n - 1) / max(n - 1, 1))
    se_cl = math.sqrt(var) if var > 0 else 0.0

    if g < min_clusters or se_cl <= 0:
        return ClusteredTest(n, g, mean, se_naive, se_cl,
                             mean / se_naive if se_naive else nan,
                             nan, nan, nan, nan)

    t = mean / se_cl
    p = 2.0 * (1.0 - norm_cdf(abs(t)))
    inflation = se_cl / se_naive if se_naive > 0 else nan
    # 군집 SE 가 순진한 SE 보다 **작을** 수도 있다(군집 내 상관이 음수인 경우).
    # 그때 n/팽창² 은 n 보다 커지는데, '독립 관측 수'가 실제 관측 수를 넘는다는
    # 말은 성립하지 않는다. 보정이 표본을 늘려 준 것이 아니라 늘리지 않은 것이므로
    # n 에서 자른다. 자르기 전 값을 감추지 않으려고 팽창률은 그대로 보고한다.
    eff = (min(n, n / (inflation ** 2))
           if np.isfinite(inflation) and inflation > 0 else nan)
    return ClusteredTest(n, g, mean, se_naive, se_cl,
                         mean / se_naive if se_naive else nan,
                         t, p, inflation, eff)


def time_clusters(times, freq: str = "W") -> np.ndarray:
    """진입 시각을 주(또는 다른 주기) 단위 군집 라벨로."""
    import pandas as pd
    ts = pd.to_datetime(pd.Series(list(times)), utc=True)
    # to_period 는 타임존을 버리며 경고한다. UTC 로 통일돼 있으므로 먼저 떼어 낸다.
    return ts.dt.tz_localize(None).dt.to_period(freq).astype(str).to_numpy()


# ─── 다중검정 보정 ───────────────────────────────────────────────

def holm(pvals: dict[str, float]) -> dict[str, float]:
    """
    Holm-Bonferroni 보정 p값.

    종목 8개를 각각 검정하면 하나가 우연히 통과할 확률이 커진다.
    Holm 은 Bonferroni 보다 검정력이 높으면서 family-wise 오류율을 지킨다.
    """
    items = sorted(pvals.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else 1.0))
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        p = p if np.isfinite(p) else 1.0
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = adj
    return out


def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, float]:
    """BH FDR 보정 p값. Holm 보다 덜 보수적 — 탐색 단계에 맞다."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else 1.0))
    m = len(items)
    out, running = {}, 1.0
    for i in range(m - 1, -1, -1):
        name, p = items[i]
        p = p if np.isfinite(p) else 1.0
        adj = min(running, p * m / (i + 1))
        running = adj
        out[name] = adj
    return out


# ─── Deflated Sharpe ────────────────────────────────────────────

@dataclass(frozen=True)
class DeflatedSharpe:
    sharpe: float
    skew: float
    kurtosis: float
    expected_max: float
    dsr: float
    n_trials: int

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.dsr) and self.dsr > 0.95)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def deflated_sharpe(
    x: np.ndarray, n_trials: int, trial_sharpes: np.ndarray | None = None
) -> DeflatedSharpe:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    N번 시험하면 귀무가설에서도 최대 샤프가 0보다 커진다. 그 기대치를 구하고,
    관측값이 그것을 넘을 확률을 준다. 왜도·첨도도 반영한다.

    x 는 **거래 단위 R배수**를 넣는다. 자산곡선 기반 샤프는 노출이 낮으면
    분모가 작아져 부풀려진다 — 실제로 노출 5.6% 종목의 샤프가 1.32로 나왔지만
    거래 단위 t는 1.40(p=0.16)으로 유의하지 않았다.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    nan = float("nan")
    if n < 30:
        return DeflatedSharpe(nan, nan, nan, nan, nan, n_trials)

    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd == 0:
        return DeflatedSharpe(nan, nan, nan, nan, nan, n_trials)
    sr = mean / sd
    z = (x - mean) / sd
    skew, kurt = float((z**3).mean()), float((z**4).mean())

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var_sr = float(np.var(np.asarray(trial_sharpes, dtype="float64"), ddof=1))
    else:
        var_sr = sr**2
    if var_sr <= 0 or n_trials < 2:
        return DeflatedSharpe(sr, skew, kurt, nan, nan, n_trials)

    e_max = math.sqrt(var_sr) * (
        (1 - EULER) * norm_ppf(1 - 1.0 / n_trials)
        + EULER * norm_ppf(1 - 1.0 / (n_trials * math.e))
    )
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr**2))
    dsr = norm_cdf((sr - e_max) * math.sqrt(n - 1) / denom)
    return DeflatedSharpe(sr, skew, kurt, e_max, dsr, n_trials)


# ─── 집중도 ─────────────────────────────────────────────────────

def gain_concentration(values: np.ndarray) -> float:
    """
    이익이 한 구간에 몰려 있는 정도 (0~1). 최고 양수값 / 양수값 합계.

    '전체 합계 대비 비중'을 쓰면 안 된다 — 합계가 음수면 음수가 나와 무의미하다.
    실제로 그 지표가 -62%, -47% 같은 값을 뱉은 적이 있다.
    """
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    gains = v[v > 0]
    if gains.size == 0:
        return float("nan")
    return float(gains.max() / gains.sum())


def sign_test(values: dict[str, float]) -> dict:
    """
    몇 개가 양수인지로 일관성을 본다.

    우위가 실재하면 한 종목이 아니라 대부분에서 나온다. 귀무가설(우위 없음)에서
    각각이 양수일 확률을 1/2로 보고 이항검정한다.
    """
    vals = {k: v for k, v in values.items() if np.isfinite(v)}
    n = len(vals)
    pos = sum(1 for v in vals.values() if v > 0)
    return {
        "n": n, "positive": pos,
        "p": binomial_p_at_least(pos, n) if n else float("nan"),
        "significant": bool(n and binomial_p_at_least(pos, n) < 0.05),
        "values": vals,
    }
