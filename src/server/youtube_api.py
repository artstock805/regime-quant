"""
유튜브 전략 API — 상황판의 "유튜브 전략" 카드가 쓴다.

    GET    /api/youtube/strategies               저장된 전략 + 켜짐 여부
    POST   /api/youtube/extract                  URL 을 분석해 저장 (꺼진 채로) + 자동 검증
    POST   /api/youtube/strategies/{name}/enable 켜기·끄기 (본문 {"on": true|false})
    DELETE /api/youtube/strategies/{name}        삭제

켠 전략은 **FX 엔진의 다음 바퀴부터** 투표에 들어간다. 엔진이 매 바퀴 저장소를
다시 읽기 때문에 재시작이 필요 없다. 발언권(가중치)은 엔진이 최근 학습 구간
성적으로 정하고, 그 값은 /api/fx/engine 의 signals[].votes 에 실려 온다.

분석은 자막을 받고 LLM 을 부르느라 수십 초가 걸린다. 동기 핸들러라 FastAPI 가
스레드풀에서 돌리므로 그동안 시세 갱신이 막히지 않는다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import load_config
from ..youtube import PREFIX, pipeline, store, validation
from ..youtube.strategy import YoutubeStrategy, strategy_group, timeframe_mode

router = APIRouter()


class ExtractRequest(BaseModel):
    url: str
    name: str | None = None
    overwrite: bool = False


class EnableRequest(BaseModel):
    on: bool


class ValidateRequest(BaseModel):
    confirm: bool


def _view(record: dict, engine_interval: str) -> dict[str, Any]:
    spec = record["spec"]
    reason = store.block_reason(record, load_config())
    return {
        **spec,
        "engine_name": PREFIX + spec["strategy_name"],
        "enabled": record["enabled"],
        "source_url": record["source_url"],
        "created_at": record["created_at"],
        # 영상의 봉을 엔진의 봉 위에서 어떻게 다루는지:
        #   same         그대로 (같은 봉이거나 영상이 봉을 말하지 않았다)
        #   resampled    영상 봉이 더 길다 — 엔진 봉을 묶어서 영상 봉으로 계산한다
        #   unsupported  영상 봉이 더 짧다 — 계산할 수 없어 켜도 투표하지 않는다
        "timeframe_mode": timeframe_mode(spec["timeframe"], engine_interval),
        # 어느 투표 군에 들어가는지: main(기존 전략과 함께) / short(단기 전략끼리 따로
        # 투표·진입·청산) / null(적용 불가 — 1분봉 등)
        "group": strategy_group(spec["timeframe"], engine_interval),
        # 과거 검증 결과. null 이면 아직 검증하지 않았다.
        "validation": record.get("validation"),
        # 과거 검증에서 손실로 나와 켤 수 없고 투표에도 들어가지 않는다.
        "blocked": reason is not None,
        # 왜 켤 수 없는지 (손실 / 검증 안 함 / 판정 불가). 켤 수 있으면 null.
        "block_reason": reason,
    }


def _llm_ready() -> bool:
    try:
        pipeline.provider()
        return True
    except pipeline.PipelineError:
        return False


@router.get("/api/youtube/strategies")
def youtube_strategies() -> dict[str, Any]:
    cfg = load_config()
    interval = str(cfg.get("fx.interval", "1h"))
    return {
        "llm_ready": _llm_ready(),
        "engine_interval": interval,
        "hold_bars": YoutubeStrategy.defaults()["hold_bars"],
        "banned": sorted(n for n in (cfg.get("fx.exclude_strategies") or [])
                         if str(n).startswith(PREFIX)),
        "strategies": [_view(r, interval) for r in store.load_all(cfg)],
        # 지금 돌고 있는(또는 마지막으로 돈) 검증. 화면이 진행 상황을 여기서 읽는다.
        "validation_job": validation.status(cfg),
        # 앞의 검증이 끝나기를 기다리는 전략 이름 (대입 직후 자동 검증의 대기 줄).
        "validation_queue": validation.queued(),
        # 손실로 검증돼 보관함으로 옮긴 전략. 목록에는 넣지 않고 따로 준다 —
        # 화면의 보관함이 규칙·검증 성적·영상을 보여 준다.
        "archived": [_view(r, interval) for r in store.load_archived(cfg)],
    }


@router.post("/api/youtube/strategies/{name}/validate")
def youtube_validate(name: str, req: ValidateRequest) -> dict[str, Any]:
    """
    이 전략을 과거 봉으로 검증한다. 몇 분 걸리므로 **별도 프로세스에서 낮은
    우선순위로** 돌리고 바로 돌아온다. 진행 상황과 결과는 전략 목록에 실려 온다.

    한 번에 하나만 돈다. JSON 본문만 받는다 — 다른 출처의 웹페이지가 CPU 를
    몇 분씩 쓰는 일을 시킬 수 없게.
    """
    cfg = load_config()
    try:
        return validation.start(name, cfg)
    except store.StoreError as exc:
        raise HTTPException(404, str(exc)) from exc
    except validation.ValidationError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/api/youtube/extract")
def youtube_extract(req: ExtractRequest) -> dict[str, Any]:
    cfg = load_config()
    try:
        record = pipeline.run(req.url, name=req.name or None,
                              overwrite=req.overwrite, cfg=cfg)
    except pipeline.PipelineError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:                 # 네트워크·LLM API 오류
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    # 대입하면 바로 과거 검증을 건다. 이미 도는 검증이 있으면 줄을 선다.
    # 검증할 수 없는 전략(1분봉 등)은 그냥 저장만 한다 — 목록이 이유를 말해 준다.
    try:
        validation.enqueue(record["spec"]["strategy_name"], cfg)
        queued = True
    except (validation.ValidationError, store.StoreError):
        queued = False
    return {**_view(record, str(cfg.get("fx.interval", "1h"))), "validation_queued": queued}


@router.post("/api/youtube/strategies/{name}/enable")
def youtube_enable(name: str, req: EnableRequest) -> dict[str, Any]:
    """
    **JSON 본문으로만 받는다.** 쿼리스트링만으로 켜지면 브라우저에 열린 아무
    웹페이지가 `fetch(..., {method: "POST", mode: "no-cors"})` 한 줄로 전략을 켤 수
    있다(이름은 추측 가능하다). application/json 은 사전 요청(preflight)을 거쳐야
    하고 이 서버는 그것을 허용하지 않으므로, 다른 출처의 페이지는 여기 닿지 못한다.
    """
    cfg = load_config()
    try:
        record = store.set_enabled(name, req.on, cfg)
    except store.BlockedError as exc:
        raise HTTPException(409, str(exc)) from exc
    except store.StoreError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _view(record, str(cfg.get("fx.interval", "1h")))


@router.delete("/api/youtube/strategies/{name}")
def youtube_delete(name: str) -> dict[str, Any]:
    try:
        store.delete(name, load_config())
    except store.StoreError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": name}
