"""
유튜브 영상의 트레이딩 전략을 FX 자동매매 엔진의 투표에 넣는다.

    URL → 자막(없으면 오디오 전사) → LLM 이 정량 규칙 JSON 으로 추출 → 검증 → 저장
    → 화면에서 켜면 엔진 투표에 참여 (발언권은 최근 성적이 정한다)

    pipeline.run(url)        추출해서 저장 (꺼진 채로)
    store.set_enabled(...)   켜기·끄기
    enabled_strategies()     기본 군 엔진이 매 바퀴 읽는 목록
    short_strategies()       단기 군 엔진이 읽는 목록 (엔진 봉보다 짧은 전략)
"""

from .strategy import (
    PREFIX, YoutubeStrategy, enabled_strategies, short_interval, short_strategies,
)

__all__ = ["PREFIX", "YoutubeStrategy", "enabled_strategies",
           "short_interval", "short_strategies"]
