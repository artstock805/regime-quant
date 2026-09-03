"""
검증 인프라.

전략이 아니라 **이 프로젝트의 실질적 산출물**이다. "잘 나온 결과가 우연인지"를
판별하는 도구 모음이고, 실제로 이 세션에서 자기 결론을 다섯 번 기각했다.

    stats        t검정·부트스트랩·Holm/BH 보정·Deflated Sharpe·부호검정
    audit        데이터 품질 감사 (봉 누락·중복·OHLC 정합성·이상치)
    permutation  블록 순열 검정 — 신호 순서를 흩어도 성과가 나오는가
    sensitivity  파라미터 고원 검사 + 비용 손익분기 배수
    efficiency   워크포워드 효율 (학습 대비 검증 성과 비율)
    report       JSON + HTML 리포트, 실행 간 비교, HTML 재생성

한 번에 다 돌리려면: python scripts/validate.py
"""

from .audit import AuditResult, audit_bars, audit_cache, audit_summary
from .efficiency import EfficiencyResult, walk_forward_efficiency
from .permutation import PermutationResult, block_permute, permutation_test
from .report import ValidationReport, compare_reports, load_report, rerender
from .sensitivity import (
    CostSensitivity, PlateauResult, cost_sensitivity, parameter_plateau,
)
from .stats import (
    DeflatedSharpe, TTest, benjamini_hochberg, binomial_p_at_least, bootstrap_ci,
    deflated_sharpe, gain_concentration, holm, norm_cdf, norm_ppf, sign_test, t_test,
)

__all__ = [
    "AuditResult", "audit_bars", "audit_cache", "audit_summary",
    "EfficiencyResult", "walk_forward_efficiency",
    "PermutationResult", "block_permute", "permutation_test",
    "ValidationReport", "compare_reports", "load_report", "rerender",
    "CostSensitivity", "PlateauResult", "cost_sensitivity", "parameter_plateau",
    "DeflatedSharpe", "TTest", "benjamini_hochberg", "binomial_p_at_least",
    "bootstrap_ci", "deflated_sharpe", "gain_concentration", "holm",
    "norm_cdf", "norm_ppf", "sign_test", "t_test",
]
