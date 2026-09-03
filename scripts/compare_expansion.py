"""
확장 전후 대조 — 사전 예측과 실측을 나란히 놓는다.

    python scripts/compare_expansion.py --base runs/deriv-taker-only --wide runs/taker-129

**예측을 먼저 박고 나중에 대조하는 이유**: 결과를 본 뒤에 해석을 고르면 어떤 결과든
설명이 된다. 이 프로젝트는 이미 그 함정에 여러 번 빠졌다 — 2년 표본에서 BNB 샤프
1.32 를 보고 "통한다"고 했다가 5.9년에서 -0.33 이 나왔다.

확장 전(8종목)에서 잰 배증계수로 세운 예측:
    명목 거래 16.8배 → 유효표본 3.57배 → 군집 t ≈ 1.10 (유의 미달)
이 파일은 그 예측이 맞았는지 틀렸는지를 기록으로 남긴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validation.stats import clustered_mean_test, t_test, time_clusters  # noqa: E402

pd.set_option("display.width", 200)

# 확장 전에 못박은 예측. 결과를 보고 고치지 않는다.
PREDICTION = {
    "명목 거래 배수": 16.8,
    "유효표본 배수": 3.57,
    "군집 t": 1.10,
    "판정": "유의 미달",
}

# 예측의 알려진 결함 — 결과를 보기 **전에** 기록해 둔다.
#
# 위 배증계수(1.79→1.58→1.42)는 8종목 **앙상블**(3,171 거래, SE 팽창 1.38배)에서
# 쟀다. 그런데 정작 확장하는 대상인 테이커 전략은 8종목에서 팽창이 **1.01배**다
# (383 거래 / 139주 = 주당 2.8건이라 같은 주에 겹치는 거래가 거의 없다).
# 거래가 성긴 전략은 군집 손실이 작다.
#
# 즉 예측은 더 촘촘한 전략의 감쇠 곡선을 빌려 쓴 것이라 **비관 쪽으로 편향**돼
# 있을 수 있다. 129종목이면 주당 거래가 ~45건으로 늘어 팽창도 커지겠지만,
# 출발점이 1.01 이므로 실측이 1.10 을 넘어도 놀랄 일이 아니다.
# 이 한계를 미리 적어 두는 이유는, 결과가 좋게 나왔을 때 "예측이 맞았다"와
# "예측 방법이 틀렸는데 운이 좋았다"를 구분하기 위해서다.
KNOWN_FLAW = (
    "배증계수를 앙상블(팽창 1.38)에서 쟀는데 대상 전략은 팽창 1.01 이다 — "
    "예측이 비관 쪽으로 편향됐을 수 있다"
)


def load(run: Path) -> tuple[dict, pd.DataFrame | None]:
    rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
    tp = run / "trades.parquet"
    tr = None
    if tp.exists():
        tr = pd.read_parquet(tp)
        tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    return rep, tr


def summarise(name: str, rep: dict, tr: pd.DataFrame | None) -> dict:
    pooled = rep["sections"].get("통합 검정", {})
    row = {
        "실행": name,
        "종목": len(rep["meta"]["symbols"]),
        "거래": pooled.get("trades"),
        "순기대R": pooled.get("net_mean"),
        "순진t": pooled.get("net_t"),
        "총기대R": pooled.get("gross_mean"),
        "통과": f"{rep['passed']}/{rep['total_checks']}",
    }
    cl = rep["sections"].get("군집 강건 검정", {}).get("net")
    if cl:
        row.update({"군집t": cl.get("t"), "팽창": cl.get("inflation"),
                    "유효표본": cl.get("effective_n")})
    elif tr is not None:
        r = clustered_mean_test(tr["r_multiple"].to_numpy(),
                                time_clusters(tr["entry_time"], "W"))
        row.update({"군집t": r.t, "팽창": r.inflation, "유효표본": r.effective_n})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="확장 전후 대조")
    ap.add_argument("--base", default="runs/deriv-taker-only")
    ap.add_argument("--wide", default="runs/taker-129")
    args = ap.parse_args()

    rows, data = [], {}
    for label, path in (("확장 전", args.base), ("확장 후", args.wide)):
        p = Path(path) if Path(path).is_absolute() else ROOT / path
        if not (p / "report.json").exists():
            print(f"리포트가 없습니다: {p}")
            return 1
        rep, tr = load(p)
        rows.append(summarise(label, rep, tr))
        data[label] = (rep, tr)

    tbl = pd.DataFrame(rows)
    print("=" * 96)
    print("확장 전후")
    print("=" * 96)
    print(tbl.round(4).to_string(index=False))

    before, after = rows[0], rows[1]
    print("\n" + "=" * 96)
    print("사전 예측 vs 실측")
    print("=" * 96)
    got_trades = (after["거래"] / before["거래"]) if before["거래"] else float("nan")
    got_eff = ((after.get("유효표본") or np.nan) / (before.get("유효표본") or np.nan))
    checks = [
        ("명목 거래 배수", PREDICTION["명목 거래 배수"], got_trades),
        ("유효표본 배수", PREDICTION["유효표본 배수"], got_eff),
        ("군집 t", PREDICTION["군집 t"], after.get("군집t")),
    ]
    print(f"{'항목':<16}{'예측':>10}{'실측':>10}{'오차':>10}")
    for name, pred, got in checks:
        err = (got - pred) if (got is not None and np.isfinite(got)) else float("nan")
        print(f"{name:<16}{pred:>10.2f}{got if got is not None else float('nan'):>10.2f}"
              f"{err:>10.2f}")

    t_got = after.get("군집t")
    if t_got is not None and np.isfinite(t_got):
        verdict = "유의" if abs(t_got) > 1.96 and t_got > 0 else "유의 미달"
        hit = "예측이 맞았다" if verdict == PREDICTION["판정"] else "예측이 틀렸다"
        print(f"\n  판정 예측 '{PREDICTION['판정']}' vs 실측 '{verdict}' → {hit}")
        if verdict == "유의 미달":
            need = (1.96 / t_got) ** 2 if t_got > 0 else float("inf")
            print(f"  t=1.96 까지 유효표본이 {need:.1f}배 더 필요하다")

    # 종목별 분포 — 확장이 무엇을 바꿨는가
    rep_a, tr_a = data["확장 후"]
    if tr_a is not None:
        per = tr_a.groupby("symbol")["r_multiple"].agg(["count", "mean"])
        per = per[per["count"] >= 30]
        print("\n" + "=" * 96)
        print(f"확장 후 종목별 (거래 30건 이상 {len(per)}종목)")
        print("=" * 96)
        print(f"  양수 {int((per['mean'] > 0).sum())}/{len(per)}종목 "
              f"({(per['mean'] > 0).mean() * 100:.0f}%)")
        print(f"  기대R 중위 {per['mean'].median():+.4f} · "
              f"사분위 {per['mean'].quantile(.25):+.4f} ~ {per['mean'].quantile(.75):+.4f}")
        print("\n  상위 5:")
        print(per.sort_values("mean", ascending=False).head(5).round(4).to_string())
        print("  하위 5:")
        print(per.sort_values("mean").head(5).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
