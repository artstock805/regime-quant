"""
감시 통화쌍 — 엔진이 훑고 화면이 호가를 띄우는 목록.

**기본은 전체다.** 이 브로커에서 거래할 수 있는 FX 쌍을 전부 본다. 줄이고 싶으면
화면(FX 상황판 → 감시 통화쌍)에서 고른다.

어디서 정해지는가 — 위에서부터 먼저 있는 것을 쓴다.

    1) 화면에서 고른 것      runs/fx_watchlist.json  {"symbols": ["EURUSD", ...]}
    2) 설정에 적은 것        config 의 fx.symbols
    3) 둘 다 비었으면        전체

빈 목록은 "아무것도 안 본다"가 아니라 **"전체"** 다. 전부 체크해서 저장한 것도
빈 목록으로 저장한다 — 이름을 박아 두면 브로커가 심볼을 추가했을 때 조용히 빠진다.

**엔진과 화면이 같은 함수를 쓴다.** 예전에는 엔진과 화면 API 가 목록을 따로
정했고(화면은 주요 14개로 잘랐다), 둘이 어긋나면 화면에 없는 종목에서 주문이
나간다.

**넓힐수록 비용이 는다.** 교차 통화쌍은 스프레드가 넓다. 이 프로젝트의 결론이
"우위는 실재하는데 비용이 그것을 먹는다"였으므로, 화면의 '비용/1R' 을 보고 고를 것.
동시 보유 한도(live.max_positions)는 목록 크기와 무관하게 그대로 적용된다.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from ..config import ROOT, Config
from .symbols import DEFAULT_PROVIDER, Pair, catalogue, resolve

logger = logging.getLogger(__name__)
DEFAULT_FILE = "runs/fx_watchlist.json"
_lock = threading.Lock()


def _path(cfg: Config) -> Path:
    path = Path(str(cfg.get("fx.watchlist_file", DEFAULT_FILE)))
    return path if path.is_absolute() else ROOT / path


def saved(cfg: Config) -> list[str] | None:
    """화면에서 저장한 선택. 저장한 적이 없거나 파일을 읽을 수 없으면 None."""
    path = _path(cfg)
    if not path.exists():
        return None
    try:
        symbols = json.loads(path.read_text(encoding="utf-8"))["symbols"]
        if not isinstance(symbols, list):
            raise TypeError("symbols 가 목록이 아닙니다")
        return [str(s).upper() for s in symbols]
    except Exception as exc:
        # 깨진 파일 때문에 엔진이 서면 안 된다. 설정값(없으면 전체)으로 돌아간다.
        logger.warning("감시 목록 파일을 읽지 못해 설정값을 씁니다: %s (%s)", path.name, exc)
        return None


def chosen(cfg: Config) -> list[str]:
    """고른 통화쌍의 표준 이름. **빈 목록은 전체**를 뜻한다."""
    picked = saved(cfg)
    if picked is not None:
        return picked
    return [str(s).upper() for s in (cfg.get("fx.symbols") or [])]


def pairs(cfg: Config, available: list[str],
          provider: str = DEFAULT_PROVIDER) -> list[Pair]:
    """감시할 통화쌍. 브로커에 없는 이름은 건너뛴다 — 없는 심볼로는 주문이 안 나간다."""
    want = chosen(cfg)
    if not want:
        return catalogue(available, provider=provider)
    found = [resolve(w, available, provider) for w in want]
    return [p for p in found if p is not None]


def save(symbols: list[str], cfg: Config, available: list[str]) -> list[str]:
    """
    선택을 저장하고 저장된 목록을 돌려준다(빈 목록 = 전체).

    브로커에 없는 이름은 거부한다. 조용히 버리면 "골랐는데 왜 안 보지"가 된다.
    """
    everything = [p.canonical for p in catalogue(available)]
    wanted = {str(s).upper() for s in symbols}
    unknown = sorted(wanted - set(everything))
    if unknown:
        raise ValueError(f"이 브로커에 없는 통화쌍입니다: {', '.join(unknown)}")
    if not wanted:
        raise ValueError("통화쌍을 하나 이상 고르세요")
    # 카탈로그 순서(주요 통화쌍 먼저)를 유지한다 — 동시 보유 한도에 걸릴 때
    # 앞에 있는 종목이 먼저 자리를 얻는다.
    picked = [] if wanted == set(everything) else [c for c in everything if c in wanted]

    path = _path(cfg)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps({"symbols": picked}, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    return picked
