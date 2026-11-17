"""
감시 통화쌍.

    · **기본은 전체다.** 설정도 저장된 선택도 없으면 브로커의 FX 쌍을 전부 본다.
    · 화면에서 고른 것이 설정보다 먼저다.
    · 엔진과 화면 API 가 같은 목록을 쓴다 — 어긋나면 화면에 없는 종목에서 주문이 나간다.
    · 깨진 저장 파일이 엔진을 세우지 않는다.

MT5 는 가짜로 갈아 끼운다.
"""

from __future__ import annotations

import json

import pytest

from src.config import Config, load_config
from src.fx import engine as eng
from src.fx import watchlist
from src.fx.symbols import catalogue

# 접미사 변형(EURUSD#)과 FX 가 아닌 것이 섞인 브로커 심볼 목록
AVAILABLE = ["EURUSD", "EURUSD#", "GBPUSD", "USDJPY", "AUDNZD", "NZDCHF", "GOLD"]
EVERYTHING = [p.canonical for p in catalogue(AVAILABLE)]


def config(tmp_path, **fx) -> Config:
    return Config({"fx": {"watchlist_file": str(tmp_path / "wl.json"), **fx}})


def names(cfg) -> list[str]:
    return [p.canonical for p in watchlist.pairs(cfg, AVAILABLE)]


# ─── 기본값 ──────────────────────────────────────────────────────

def test_the_default_is_every_fx_pair_the_broker_has(tmp_path):
    cfg = config(tmp_path)
    assert watchlist.chosen(cfg) == []
    assert names(cfg) == EVERYTHING
    assert len(EVERYTHING) == 5 and "GOLD" not in EVERYTHING


def test_the_committed_config_leaves_the_list_empty():
    """저장소 기본 설정이 목록을 박아 두면 '기본은 전체'가 말뿐이 된다."""
    assert load_config(use_local=False).get("fx.symbols") == []


def test_cross_pairs_are_not_cut_from_the_default(tmp_path):
    """예전에는 목록이 비면 주요 통화쌍만 봤다(majors_only). AUDNZD·NZDCHF 가 빠졌다."""
    assert {"AUDNZD", "NZDCHF"} <= set(names(config(tmp_path)))


# ─── 고르기 ──────────────────────────────────────────────────────

def test_a_saved_choice_narrows_the_list(tmp_path):
    cfg = config(tmp_path)
    assert watchlist.save(["usdjpy", "EURUSD"], cfg, AVAILABLE) == ["EURUSD", "USDJPY"]
    assert names(cfg) == ["EURUSD", "USDJPY"]          # 카탈로그 순서를 따른다


def test_the_screen_choice_beats_the_config(tmp_path):
    cfg = config(tmp_path, symbols=["GBPUSD"])
    assert names(cfg) == ["GBPUSD"]                    # 저장 전에는 설정값
    watchlist.save(["EURUSD"], cfg, AVAILABLE)
    assert names(cfg) == ["EURUSD"]


def test_choosing_everything_is_saved_as_all(tmp_path):
    """이름을 박아 두면 브로커가 심볼을 추가했을 때 조용히 빠진다."""
    cfg = config(tmp_path, symbols=["GBPUSD"])
    assert watchlist.save(EVERYTHING, cfg, AVAILABLE) == []
    assert json.loads((tmp_path / "wl.json").read_text())["symbols"] == []
    assert names(cfg) == EVERYTHING                    # 설정의 GBPUSD 로 돌아가지 않는다
    grown = AVAILABLE + ["EURGBP"]
    assert "EURGBP" in [p.canonical for p in watchlist.pairs(cfg, grown)]


@pytest.mark.parametrize("symbols", [[], ["EURTRY"], ["EURUSD", "GOLD"], ["../x"]])
def test_bad_choices_are_refused_and_nothing_is_written(tmp_path, symbols):
    cfg = config(tmp_path)
    with pytest.raises(ValueError):
        watchlist.save(symbols, cfg, AVAILABLE)
    assert not (tmp_path / "wl.json").exists()


def test_a_symbol_the_broker_dropped_is_skipped(tmp_path):
    cfg = config(tmp_path)
    watchlist.save(["EURUSD", "AUDNZD"], cfg, AVAILABLE)
    assert [p.canonical for p in watchlist.pairs(cfg, ["EURUSD", "GBPUSD"])] == ["EURUSD"]


@pytest.mark.parametrize("content", ["{not json", "[]", '{"symbols": "EURUSD"}', '{"x": 1}'])
def test_a_broken_file_falls_back_instead_of_stopping_the_engine(tmp_path, content):
    cfg = config(tmp_path, symbols=["GBPUSD"])
    (tmp_path / "wl.json").write_text(content, encoding="utf-8")
    assert names(cfg) == ["GBPUSD"]


def test_no_temp_file_is_left_behind(tmp_path):
    watchlist.save(["EURUSD"], config(tmp_path), AVAILABLE)
    assert [p.name for p in tmp_path.iterdir()] == ["wl.json"]


# ─── 엔진과 화면이 같은 목록을 쓴다 ──────────────────────────────

def test_the_engine_reads_the_choice_on_every_cycle(tmp_path, monkeypatch):
    from src.execution.journal import Journal
    from src.fx import mt5

    monkeypatch.setattr(mt5, "available_symbols", lambda only_forex=True: AVAILABLE)
    cfg = config(tmp_path)
    t = eng.FxTrader(cfg=cfg, journal=Journal(tmp_path / "j.jsonl"))
    assert [p.canonical for p in t.watchlist()] == EVERYTHING

    watchlist.save(["USDJPY"], cfg, AVAILABLE)          # 재시작 없이
    assert [p.canonical for p in t.watchlist()] == ["USDJPY"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.server import fx_api

    cfg = config(tmp_path)
    monkeypatch.setattr(fx_api, "load_config", lambda: cfg)
    monkeypatch.setattr(fx_api, "available_symbols", lambda only_forex=True: AVAILABLE)
    monkeypatch.setattr(fx_api, "_pairs_cache", None)
    app = FastAPI()
    app.include_router(fx_api.router)
    return TestClient(app), cfg


def watched(body) -> list[str]:
    return [p["canonical"] for p in body["pairs"] if p["watched"]]


def test_the_screen_lists_every_pair_and_marks_all_watched_by_default(client):
    c, _ = client
    body = c.get("/api/fx/pairs").json()
    assert body["all"] is True
    assert [p["canonical"] for p in body["pairs"]] == EVERYTHING
    assert watched(body) == EVERYTHING


def test_saving_from_the_screen_changes_what_the_engine_sees(client):
    c, cfg = client
    body = c.post("/api/fx/watchlist", json={"symbols": ["EURUSD", "NZDCHF"]}).json()
    assert body["all"] is False and watched(body) == ["EURUSD", "NZDCHF"]
    assert names(cfg) == ["EURUSD", "NZDCHF"]           # 엔진이 쓰는 함수와 같은 답

    back = c.post("/api/fx/watchlist", json={"symbols": EVERYTHING}).json()
    assert back["all"] is True and names(cfg) == EVERYTHING


def test_the_screen_refuses_bad_requests(client):
    c, cfg = client
    assert c.post("/api/fx/watchlist", json={"symbols": []}).status_code == 400
    assert c.post("/api/fx/watchlist", json={"symbols": ["EURTRY"]}).status_code == 400
    # 다른 출처의 웹페이지가 보낼 수 있는 형태(본문 없음·폼)는 받지 않는다
    assert c.post("/api/fx/watchlist").status_code == 422
    assert c.post("/api/fx/watchlist", data={"symbols": "EURUSD"}).status_code == 422
    assert names(cfg) == EVERYTHING
