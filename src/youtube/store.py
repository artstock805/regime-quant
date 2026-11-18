"""
유튜브 전략 저장소 — 전략 하나가 JSON 파일 하나다.

    runs/youtube/<strategy_name>.json
    {
      "spec": { ...src/youtube/spec.py 의 명세... },
      "source_url": "https://www.youtube.com/watch?v=...",
      "created_at": "2026-11-16T05:00:00+00:00",
      "enabled": false
    }

**파이썬 파일을 만들지 않는다.** 자막에서 나온 내용이 코드가 되면 그 코드가
매매 프로세스 안에서 임포트된다. 여기 저장되는 것은 데이터뿐이고, 수식은
src/youtube/expr.py 의 평가기만 읽는다.

**새로 저장하면 항상 꺼진 채다.** 같은 영상을 다시 분석해도 LLM 이 다른 수식을
내놓을 수 있다(실측: UT Bot 근사가 실행마다 달라졌다). 켜져 있던 전략을 덮어쓴
순간 내용이 바뀐 채로 투표에 남으면, 사람이 본 적 없는 규칙이 주문에 관여한다.

위치는 `youtube.dir` (기본 runs/youtube). runs/ 는 git 에서 제외되므로 각자의
전략이 저장소에 섞여 들어가지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, Config, load_config
from .spec import SpecError, check_name, normalize, validate

logger = logging.getLogger(__name__)
DEFAULT_DIR = "runs/youtube"
# 저장·토글·삭제는 읽고-고쳐-쓰기다. 겹치면 토글이 방금 덮어쓴 새 명세를 옛 명세로
# 되돌리면서 **켠 채로** 남긴다 — "새로 저장하면 꺼져 있다"가 깨진다.
_write_lock = threading.Lock()


class StoreError(ValueError):
    """저장 규칙에 걸렸다 (이미 있음, 다른 영상의 전략 등)."""


def directory(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    path = Path(str(cfg.get("youtube.dir", DEFAULT_DIR)))
    return path if path.is_absolute() else ROOT / path


def _path(name: str, cfg: Config | None) -> Path:
    try:
        return directory(cfg) / f"{check_name(name)}.json"
    except SpecError as exc:
        raise StoreError(str(exc)) from exc


def _read(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise SpecError(f"전략 파일이 객체가 아닙니다: {path.name}")
    spec = normalize(record.get("spec"))
    validate(spec)
    if spec["strategy_name"] != path.stem:
        raise SpecError(f"파일 이름과 전략 이름이 다릅니다: {path.name}")
    return {"spec": spec,
            "source_url": str(record.get("source_url", "")),
            "created_at": str(record.get("created_at", "")),
            "enabled": record.get("enabled") is True,
            # 과거 검증 결과(src/youtube/validation.py). 없으면 아직 검증하지 않은 것이다.
            "validation": (record["validation"]
                           if isinstance(record.get("validation"), dict) else None)}


def _write(path: Path, record: dict) -> None:
    """임시 파일에 쓰고 한 번에 바꾼다. 엔진이 반쯤 쓰인 파일을 읽으면 안 된다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


def load_all(cfg: Config | None = None) -> list[dict]:
    """
    저장된 전략 전부. 읽을 수 없는 파일은 건너뛰고 경고를 남긴다 —
    손으로 고치다 깨진 파일 하나가 엔진 전체를 세우면 안 된다.
    """
    folder = directory(cfg)
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            out.append(_read(path))
        except Exception as exc:     # 무엇이든 — 이 목록은 엔진이 매 바퀴 읽는다
            logger.warning("유튜브 전략 파일을 읽지 못해 건너뜁니다: %s (%s)", path.name, exc)
    return out


def get(name: str, cfg: Config | None = None) -> dict:
    path = _path(name, cfg)
    if not path.exists():
        raise StoreError(f"없는 전략입니다: {name}")
    try:
        return _read(path)
    except Exception as exc:
        raise StoreError(f"전략 파일이 깨졌습니다: {path.name} ({exc})") from exc


def save(spec: dict, source_url: str, *, overwrite: bool = False,
         allow_other_source: bool = False, cfg: Config | None = None) -> dict:
    """
    검증을 통과한 명세만 저장한다. 항상 enabled=false 로 저장한다.

    이름은 LLM 이 짓는다. 다른 영상에서 같은 이름이 나오면 남의 전략을 덮어쓰게
    되므로, 출처가 다르면 overwrite 만으로는 덮어쓰지 않는다. 사람이 이름을
    직접 지정한 경우(allow_other_source)에만 허용한다.
    """
    spec = normalize(spec)
    validate(spec)
    path = _path(spec["strategy_name"], cfg)
    with _write_lock:
        if path.exists():
            if not overwrite:
                raise StoreError(f"이미 있는 전략입니다: {spec['strategy_name']} "
                                 "(다시 분석해 덮어쓰려면 덮어쓰기를 켜세요)")
            try:
                previous = json.loads(path.read_text(encoding="utf-8")).get("source_url")
            except Exception:
                previous = None
            if previous != source_url and not allow_other_source:
                raise StoreError(f"{spec['strategy_name']} 은 다른 영상에서 만든 전략입니다. "
                                 "덮어쓰려면 이름을 직접 지정하세요")
        record = {"spec": spec, "source_url": source_url,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "enabled": False}
        _write(path, record)
    return record


class BlockedError(StoreError):
    """과거 검증에서 손실로 나온 전략이라 켤 수 없다."""


def blocked_by_validation(record: dict, cfg: Config | None = None) -> bool:
    """
    과거 검증에서 **비용 후 기대값이 음수**로 나온 전략인가. 그런 전략은 켤 수 없고
    투표에도 들어가지 않는다.

    엔진의 발언권은 최근 6개월 성적으로 정해진다. 6개월은 짧아서, 9년 내내 잃은
    전략도 최근 운이 좋았던 통화쌍·국면에서는 발언권을 얻는다(실제로 그랬다).
    긴 검증이 손실이라고 말했으면 그쪽을 믿는다.

    검증하지 않은 전략은 막지 않는다 — 모르는 것과 나쁜 것은 다르다. 거래가 너무
    적어 판정하지 못한 전략(net_r 가 없다)도 막지 않는다.
    같은 이름으로 다시 분석하면 검증 결과가 지워지므로 다시 켤 수 있다.
    """
    return block_reason(record, cfg) is not None


def block_reason(record: dict, cfg: Config | None = None) -> str | None:
    """
    이 전략을 켤 수 없는 이유. 켤 수 있으면 None.

    손실로 검증된 전략은 언제나 막는다. `youtube.require_validation` 이 켜져 있으면
    (저장소 기본 설정) **검증을 통과한 전략만** 켤 수 있다 — 검증하지 않은 전략과,
    거래가 너무 적어 판정하지 못한 전략도 막는다. 검증을 돌렸다는 것과 통과했다는
    것은 다르다. 판정이 없으면 괜찮다는 근거도 없다.
    """
    result = record.get("validation") or None
    net = (result or {}).get("net_r")
    if isinstance(net, (int, float)) and not isinstance(net, bool):
        return f"과거 검증에서 손실({net:+.3f}R)로 나왔습니다" if net < 0 else None
    if not bool((cfg or load_config()).get("youtube.require_validation", False)):
        return None
    if result is None:
        return "아직 과거 검증을 하지 않았습니다. 검증을 통과해야 켤 수 있습니다"
    return f"거래가 너무 적어({result.get('trades', 0)}건) 판정할 수 없습니다"


def set_enabled(name: str, enabled: bool, cfg: Config | None = None) -> dict:
    with _write_lock:
        current = get(name, cfg)
        reason = block_reason(current, cfg) if enabled else None
        if reason:
            raise BlockedError(f"{name} 은 켤 수 없습니다 — {reason}")
        record = {**current, "enabled": bool(enabled)}
        _write(_path(name, cfg), record)
    return record


def set_validation(name: str, result: dict, cfg: Config | None = None) -> bool:
    """
    검증 결과를 전략에 붙인다. 붙였으면 True.

    검증은 몇 분이 걸린다. 그 사이 같은 이름으로 전략을 다시 분석해 덮어썼다면
    이 결과는 **옛 전략의 성적**이다 — 새 전략에 붙이면 본 적 없는 규칙에 남의
    성적표가 달린다. 검증을 시작할 때의 created_at 과 지금 것이 다르면 버린다.
    """
    with _write_lock:
        record = get(name, cfg)
        if record["created_at"] != result.get("spec_created_at"):
            return False
        updated = {**record, "validation": result}
        if blocked_by_validation(updated, cfg):
            # 손실로 나왔으면 그 자리에서 끈다. 켜진 채 "차단됨"이라고만 표시하면
            # 화면의 토글과 실제 상태가 어긋난다.
            updated["enabled"] = False
        _write(_path(name, cfg), updated)
    return True


# ─── 보관함 ──────────────────────────────────────────────────────
#
# 손실로 검증된 전략은 여기로 옮긴다. 켤 수도 없는 전략이 목록에 쌓이면 쓸 만한
# 것이 그 사이에 묻힌다. 지우지는 않는다 — 어떤 영상이 어떤 성적이었는지가
# 남아 있어야 같은 영상을 다시 분석하는 헛수고를 안 한다.
# load_all 은 이 폴더를 읽지 않으므로 화면 목록과 엔진 투표에서 함께 빠진다.
ARCHIVE = "archive"


def archive_directory(cfg: Config | None = None) -> Path:
    return directory(cfg) / ARCHIVE


def archive(name: str, cfg: Config | None = None) -> dict:
    """전략을 보관함으로 옮긴다. 꺼서 옮기고, 같은 이름의 옛 보관본은 덮어쓴다."""
    with _write_lock:
        record = {**get(name, cfg), "enabled": False}
        _write(archive_directory(cfg) / f"{check_name(name)}.json", record)
        _path(name, cfg).unlink()
    return record


def load_archived(cfg: Config | None = None) -> list[dict]:
    """보관된 전략. 읽을 수 없는 파일은 건너뛴다."""
    folder = archive_directory(cfg)
    out = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            out.append(_read(path))
        except Exception as exc:
            logger.warning("보관된 전략 파일을 읽지 못해 건너뜁니다: %s (%s)", path.name, exc)
    return out


def delete(name: str, cfg: Config | None = None) -> None:
    path = _path(name, cfg)
    with _write_lock:
        if not path.exists():
            raise StoreError(f"없는 전략입니다: {name}")
        path.unlink()
