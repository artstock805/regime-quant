"""
설정 로더.

config/config.yaml 을 읽어 점 표기(dot notation)로 꺼낼 수 있는 객체로 감싼다.
경로는 프로젝트 루트 기준으로 절대경로화해서 어느 디렉터리에서 실행해도 동일하게 동작한다.

**설정은 세 겹으로 쌓인다.** 배포판이 남의 컴퓨터에서 그대로 도는 것이 목적이다.

    1) config/config.yaml    저장소에 들어가는 기본값. **경로는 전부 상대경로**다.
    2) config/local.yaml     각자의 컴퓨터 설정. git 에서 제외된다(없어도 된다).
    3) REGIME_QUANT_DATA     데이터 위치만 환경변수로 덮어쓴다.

이 구조가 없으면 데이터를 다른 드라이브에 둔 사람의 절대경로가 저장소에 박혀
받는 쪽에서 바로 깨진다. 실제로 이 프로젝트가 그 상태였다(D:/regime-quant-data).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/config.py → 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class Config:
    """중첩 딕셔너리를 'a.b.c' 로 조회하는 얇은 래퍼."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._raw
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, path: str) -> Any:
        """없으면 조용히 넘어가지 않고 바로 터진다 — 설정 오타를 런타임 뒤로 미루지 않는다."""
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"설정 키가 없습니다: {path} ({DEFAULT_CONFIG})")
        return value

    def path(self, path: str) -> Path:
        """설정에 적힌 상대경로를 프로젝트 루트 기준 절대경로로 바꿔 준다."""
        p = Path(self.require(path))
        return p if p.is_absolute() else ROOT / p

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    def __repr__(self) -> str:
        return f"Config(market={self.get('market')}, universe={len(self.get('universe', []))}종목)"


LOCAL_CONFIG = ROOT / "config" / "local.yaml"
DATA_ENV = "REGIME_QUANT_DATA"


def _deep_merge(base: dict, over: dict) -> dict:
    """중첩 딕셔너리를 덮어쓴다. 통째로 교체하면 local.yaml 에 전부 적어야 한다."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_data_env(raw: dict) -> dict:
    """
    REGIME_QUANT_DATA 로 데이터 위치만 바꾼다.

    받는 사람이 대용량 데이터를 다른 드라이브에 두고 싶을 때 설정 파일을 고치지
    않아도 되게 한다 — 1분봉이 종목당 90MB 라 129종목이면 16GB 다.
    """
    base = os.environ.get(DATA_ENV)
    if not base:
        return raw
    root = Path(base)
    paths = dict(raw.get("paths", {}))
    paths["data_dir"] = str(root)
    paths["klines_dir"] = str(root / "klines")
    paths["derivatives_dir"] = str(root / "derivatives")
    return {**raw, "paths": paths}


def load_config(path: str | Path | None = None, *, use_local: bool = True) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if use_local and path is None and LOCAL_CONFIG.exists():
        with LOCAL_CONFIG.open("r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})

    return Config(_apply_data_env(raw))


@dataclass(frozen=True)
class Costs:
    """수수료·슬리피지를 한 덩어리로 들고 다닌다. 백테스터가 이걸 빼면 결과가 전부 거짓말이 된다."""

    fee_rate: float
    slippage_bps: float

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def entry_cost_rate(self) -> float:
        """진입 1회 비용률(수수료 + 슬리피지)."""
        return self.fee_rate + self.slippage_rate

    def round_trip_rate(self) -> float:
        """진입+청산 왕복 비용률. 전략의 기대수익이 이 값을 넘지 못하면 의미가 없다."""
        return 2 * self.entry_cost_rate()

    @classmethod
    def from_config(cls, cfg: Config) -> Costs:
        return cls(
            fee_rate=float(cfg.require("backtest.fee_rate")),
            slippage_bps=float(cfg.require("backtest.slippage_bps")),
        )
