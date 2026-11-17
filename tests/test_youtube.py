"""
유튜브 전략.

여기서 지키려는 것은 세 가지다.

    · **자막에서 온 문자열이 코드로 실행되지 않는다.** 수식은 화이트리스트
      문법만 통과하고, 저장되는 것은 JSON 뿐이다.
    · **연구용 전략 풀은 그대로다.** build_all() 에 섞이면 검증 숫자가 조용히
      바뀐다. 유튜브 전략은 FX 엔진의 투표에만 들어간다.
    · **사람이 켜야 들어간다.** 새로 저장하거나 덮어쓴 전략은 꺼져 있다.

LLM 과 유튜브는 전부 가짜로 갈아 끼운다. 테스트가 네트워크를 타면 안 된다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.config import Config
from src.fx import engine as eng
from src.strategies import all_strategies, build_all
from src.youtube import PREFIX, YoutubeStrategy, enabled_strategies, pipeline, spec, store
from src.youtube.expr import ExpressionError, evaluate, names_in

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def raw_spec(**over) -> dict:
    base = {
        "strategy_name": "sma_trend",
        "target_market": "FOREX",
        "timeframe": "1h",
        "indicators_needed": [{"name": "SMA", "params": {"period": 3}}],
        "entry_conditions": {"long": "close > sma_3 and close > prev_close",
                             "short": "close < sma_3 and close < prev_close"},
        "exit_conditions": {"take_profit_pct": 0.02, "stop_loss_pct": 0.01,
                            "indicator_exit": "RSI 70 이상"},
        "quant_assumptions": ["테스트용 가정"],
    }
    return {**base, **over}


def bars(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype="float64")
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1,
                         "close": closes, "volume": np.full(len(closes), 100.0)},
                        index=idx)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config({"youtube": {"dir": str(tmp_path / "yt")}})


# ─── 수식: 문법 밖은 실행되지 않는다 ─────────────────────────────

@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo x') > 0",
    "close.__class__ == 1",
    "open('x') > 1",
    "[c for c in close] > 1",
    "(close if 1 else 2) > 1",
    "'a' < 'b'",
    "close ** 99 > 1",
    "close[0] > 1",
])
def test_anything_outside_the_whitelist_is_rejected(expr):
    with pytest.raises(ExpressionError):
        names_in(expr)


@pytest.mark.parametrize("expr", [
    "close", "close + 1", "1", "True", "close and volume",   # 값 식 — 항상 참이 된다
    "1 > 0",                                                 # 변수가 없다 — 항상 참
    "abs() > 1", "max(close) > 1",                           # 함수 오용
])
def test_expressions_that_would_always_fire_are_rejected(expr):
    with pytest.raises(ExpressionError):
        names_in(expr)


def test_overlong_expressions_are_rejected():
    with pytest.raises(ExpressionError):
        names_in("+".join(["close"] * 5000) + " > 1")


def test_conditions_are_evaluated_per_bar():
    df = bars([1, 2, 3, 2, 5])
    cols = {"close": df["close"], "prev_close": df["close"].shift(1)}
    up = evaluate("close > prev_close", cols, df.index)
    assert up.tolist() == [False, True, True, False, True]
    both = evaluate("close > prev_close and not close > 4", cols, df.index)
    assert both.tolist() == [False, True, True, False, False]
    assert evaluate("max(close, prev_close) >= 3", cols, df.index).tolist() == \
        [False, False, True, True, True]


def test_warmup_bars_are_false_even_under_not():
    """`not (NaN > x)` 는 그대로 두면 참이다. 워밍업 구간이 통째로 진입 신호가 된다."""
    df = bars([1, 2, 3])
    cols = {"close": df["close"], "prev_close": df["close"].shift(1)}
    assert evaluate("not close > prev_close", cols, df.index).tolist() == \
        [False, False, False]


def test_division_by_zero_in_series_is_false_not_an_error():
    df = bars([1, 2, 3])
    cols = {"close": df["close"], "volume": df["volume"] * 0}
    assert not evaluate("close / volume < 1", cols, df.index).any()


def test_unknown_variables_are_an_error():
    df = bars([1, 2, 3])
    with pytest.raises(ExpressionError, match="mystery"):
        evaluate("close > mystery", {"close": df["close"]}, df.index)


# ─── 명세 검증 ───────────────────────────────────────────────────

def test_a_good_spec_passes_and_unknown_fields_are_dropped():
    out = spec.parse(json.dumps({**raw_spec(), "extra": "ignored"}))
    assert out["strategy_name"] == "sma_trend" and "extra" not in out


@pytest.mark.parametrize("name", ["../evil", "Has_Upper", "__init__", "for", "ab",
                                  "a/b", "a.b", "con\n", None, 3])
def test_the_name_cannot_carry_a_path(name):
    with pytest.raises(spec.SpecError):
        spec.normalize(raw_spec(strategy_name=name))


@pytest.mark.parametrize("expr, culprit", [
    ("close > ema_20", "ema_20"),
    ("abs > 1", "abs"),              # 함수 이름을 변수처럼 쓴 것도 모르는 변수다
])
def test_undeclared_variables_are_caught_before_saving(expr, culprit):
    bad = raw_spec(entry_conditions={"long": expr, "short": ""})
    with pytest.raises(spec.SpecError, match=culprit):
        spec.parse(json.dumps(bad))


def test_at_least_one_side_is_required():
    with pytest.raises(spec.SpecError, match="비어"):
        spec.parse(json.dumps(raw_spec(entry_conditions={"long": "", "short": " "})))


@pytest.mark.parametrize("indicator", [
    {"name": "ICHIMOKU"},
    {"name": "SMA", "params": {"period": 0}},
    {"name": "SMA", "params": {"period": 5000}},
    {"name": "SMA", "params": {"period": "20"}},
    {"name": "BBANDS", "params": {"std": float("inf")}},
])
def test_bad_indicators_are_rejected(indicator):
    with pytest.raises(spec.SpecError):
        spec.validate(spec.normalize(raw_spec(indicators_needed=[indicator])))


def test_non_finite_ratios_are_rejected():
    with pytest.raises(spec.SpecError):
        spec.normalize(raw_spec(exit_conditions={"take_profit_pct": float("nan")}))


def test_an_unsupported_timeframe_is_rejected():
    with pytest.raises(spec.SpecError):
        spec.normalize(raw_spec(timeframe="3h"))


# ─── 전략: 조건을 상태로 ─────────────────────────────────────────

def strategy(hold_bars: int = 2, **over) -> YoutubeStrategy:
    return YoutubeStrategy(spec.normalize(raw_spec(**over)), hold_bars=hold_bars)


def test_the_signal_follows_the_condition_then_holds():
    df = bars([10, 10, 10, 11, 12, 12, 12, 12, 12])
    out = strategy(hold_bars=2).generate(df, df)
    # 3·4번 봉에서 롱 조건이 참 → 마지막 참(4)에서 한 봉 더 유지하고 내려놓는다
    assert out["direction"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert out["strength"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]


def test_an_opposite_condition_flips_the_state():
    df = bars([10, 10, 10, 11, 12, 11, 10, 10, 10])
    out = strategy(hold_bars=3).generate(df, df)
    assert out["direction"].tolist() == [0, 0, 0, 1, 1, -1, -1, -1, -1]


def test_both_sides_true_on_one_bar_is_no_decision():
    df = bars([10, 11, 12, 13])
    same = {"long": "close > prev_close", "short": "close > prev_close"}
    out = strategy(entry_conditions=same).generate(df, df)
    assert (out["direction"] == 0).all()


def test_a_one_sided_strategy_never_goes_the_other_way():
    df = make_candles(500, "1h", derivatives=False)
    out = strategy(entry_conditions={"long": "close > sma_3", "short": ""}) \
        .generate(df, df)
    assert (out["direction"] >= 0).all() and (out["direction"] > 0).any()


def test_signals_never_look_ahead():
    """
    이 프로젝트에서 가장 중요한 성질. 봉 t 의 신호는 t 이후 봉이 무엇이든 같아야
    한다 — 뒤를 잘라 다시 계산했을 때 앞부분이 달라지면 미래를 본 것이다.
    """
    df = make_candles(600, "1h", derivatives=False)
    s = strategy(hold_bars=6, indicators_needed=[
        {"name": "EMA", "params": {"period": 20}}, {"name": "RSI"},
        {"name": "MACD"}, {"name": "STOCH"}, {"name": "BBANDS"},
        {"name": "ATR"}, {"name": "HIGHEST"}, {"name": "LOWEST"},
        {"name": "VOLUME_SMA"}],
        entry_conditions={
            "long": "close > prev_highest_20 - 2 * atr_14 and rsi_14 > 50 "
                    "and macd > macd_signal and volume > volume_sma_20",
            "short": "close < bb_lower_20 or (stoch_k_14 < stoch_d_14 and close < ema_20)"})
    full = s.generate(df, df)
    for cut in (200, 350, 599):
        part = s.generate(df.iloc[:cut], df.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_it_declares_every_regime_and_needs_no_feature_columns():
    s = strategy()
    assert s.name == PREFIX + "sma_trend"
    assert len(s.regimes) == 4 and s.needs == ()


# ─── 저장소 ──────────────────────────────────────────────────────

def test_a_saved_strategy_starts_switched_off(cfg):
    record = store.save(raw_spec(), URL, cfg=cfg)
    assert record["enabled"] is False
    assert enabled_strategies(cfg) == []


def test_only_enabled_strategies_reach_the_engine_list(cfg):
    store.save(raw_spec(), URL, cfg=cfg)
    store.save(raw_spec(strategy_name="other_one"), URL, cfg=cfg)
    store.set_enabled("sma_trend", True, cfg)
    assert [s.name for s in enabled_strategies(cfg)] == ["yt_sma_trend"]
    store.set_enabled("sma_trend", False, cfg)
    assert enabled_strategies(cfg) == []


def test_overwriting_needs_consent_and_switches_it_off_again(cfg):
    store.save(raw_spec(), URL, cfg=cfg)
    store.set_enabled("sma_trend", True, cfg)
    with pytest.raises(store.StoreError, match="이미"):
        store.save(raw_spec(), URL, cfg=cfg)
    # 다시 분석하면 내용이 달라질 수 있다 — 켜진 채 바뀌면 안 된다
    assert store.save(raw_spec(), URL, overwrite=True, cfg=cfg)["enabled"] is False


def test_a_strategy_from_another_video_is_not_overwritten_by_a_name_clash(cfg):
    store.save(raw_spec(), URL, cfg=cfg)
    other = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    with pytest.raises(store.StoreError, match="다른 영상"):
        store.save(raw_spec(), other, overwrite=True, cfg=cfg)
    store.save(raw_spec(), other, overwrite=True, allow_other_source=True, cfg=cfg)
    assert store.get("sma_trend", cfg)["source_url"] == other


def test_a_broken_file_is_skipped_not_fatal(cfg):
    store.save(raw_spec(), URL, cfg=cfg)
    folder = store.directory(cfg)
    (folder / "garbage.json").write_text("{not json", encoding="utf-8")
    hand_edited = json.loads((folder / "sma_trend.json").read_text(encoding="utf-8"))
    hand_edited["spec"]["entry_conditions"]["long"] = "__import__('os').system('x') > 0"
    hand_edited["spec"]["strategy_name"] = "tampered"
    hand_edited["enabled"] = True
    (folder / "tampered.json").write_text(json.dumps(hand_edited), encoding="utf-8")

    # ValueError 가 아닌 것으로 터지는 파일들 — 엔진이 매 바퀴 읽는 목록이라
    # 하나라도 새면 매매 루프 전체가 멈춘다
    (folder / "not_an_object.json").write_text("[]", encoding="utf-8")
    huge = json.loads((folder / "sma_trend.json").read_text(encoding="utf-8"))
    huge["spec"]["strategy_name"] = "huge_number"
    huge["spec"]["entry_conditions"]["long"] = "close > " + "9" * 400
    (folder / "huge_number.json").write_text(json.dumps(huge), encoding="utf-8")

    assert [r["spec"]["strategy_name"] for r in store.load_all(cfg)] == ["sma_trend"]
    assert not list(folder.glob("*.tmp"))
    with pytest.raises(store.StoreError):
        store.get("huge_number", cfg)


def test_a_huge_number_is_rejected_as_an_expression_error():
    df = bars([1, 2, 3])
    with pytest.raises(ExpressionError):
        evaluate("close > " + "9" * 400, {"close": df["close"]}, df.index)


def test_a_strategy_that_fails_at_runtime_abstains(caplog):
    """터진 전략 하나가 다른 전략의 판단과 포지션 관리를 막으면 안 된다."""
    s = strategy()
    df = bars([10, 11, 12, 13]).drop(columns=["volume"])      # 봉에 열이 빠졌다
    out = s.generate(df, df)
    assert (out["direction"] == 0).all() and (out["strength"] == 0).all()
    assert "기권" in caplog.text


@pytest.mark.parametrize("name", ["../x", "a/b", "..", ""])
def test_store_paths_cannot_escape_the_folder(cfg, name):
    for call in (lambda: store.get(name, cfg), lambda: store.delete(name, cfg),
                 lambda: store.set_enabled(name, True, cfg)):
        with pytest.raises(store.StoreError):
            call()


def test_delete_removes_the_file(cfg):
    store.save(raw_spec(), URL, cfg=cfg)
    store.delete("sma_trend", cfg)
    assert store.load_all(cfg) == []


# ─── 연구용 풀과 엔진 ────────────────────────────────────────────

def test_the_research_pool_is_untouched(cfg):
    """검증 숫자(129종목 7,496거래)가 나온 풀이다. 유튜브 전략이 섞이면 안 된다."""
    before = sorted(all_strategies())
    store.save(raw_spec(), URL, cfg=cfg)
    store.set_enabled("sma_trend", True, cfg)
    enabled_strategies(cfg)
    assert sorted(all_strategies()) == before
    assert not any(s.name.startswith(PREFIX) for s in build_all())


def trader_with(cfg_raw: dict, tmp_path) -> eng.FxTrader:
    from src.execution.journal import Journal
    return eng.FxTrader(cfg=Config(cfg_raw), journal=Journal(tmp_path / "j.jsonl"))


def test_the_engine_votes_with_enabled_youtube_strategies(tmp_path):
    raw = {"youtube": {"dir": str(tmp_path / "yt")}}
    cfg = Config(raw)
    store.save(raw_spec(), URL, cfg=cfg)
    t = trader_with(raw, tmp_path)
    assert "yt_sma_trend" not in {s.name for s in t.strategies()}

    store.set_enabled("sma_trend", True, cfg)
    names = {s.name for s in t.strategies()}      # 재시작 없이 다음 호출에 반영
    assert "yt_sma_trend" in names
    assert names - {"yt_sma_trend"}, "기존 전략이 밀려나면 안 된다"


def test_the_denylist_beats_the_toggle(tmp_path):
    raw = {"youtube": {"dir": str(tmp_path / "yt")},
           "fx": {"exclude_strategies": ["yt_sma_trend"]}}
    cfg = Config(raw)
    store.save(raw_spec(), URL, cfg=cfg)
    store.set_enabled("sma_trend", True, cfg)
    assert "yt_sma_trend" not in {s.name for s in trader_with(raw, tmp_path).strategies()}


def test_the_allowlist_does_not_silently_drop_an_enabled_strategy(tmp_path):
    raw = {"youtube": {"dir": str(tmp_path / "yt")},
           "fx": {"strategies": ["donchian_breakout"]}}
    cfg = Config(raw)
    store.save(raw_spec(), URL, cfg=cfg)
    store.set_enabled("sma_trend", True, cfg)
    assert {s.name for s in trader_with(raw, tmp_path).strategies()} == \
        {"donchian_breakout", "yt_sma_trend"}


def test_a_youtube_strategy_runs_through_the_real_signal_path(tmp_path):
    """가짜 신호가 아니라 진짜 signal_for 로 한 번 돌린다 — 배선이 맞는지."""
    from src.indicators import compute_all
    from src.live.signals import signal_for
    from src.config import load_config

    df = make_candles(1200, "1h", start="2025-01-01", seed=3, derivatives=False)
    strats = [s for s in build_all() if s.name == "donchian_breakout"] + [strategy()]
    sig = signal_for(load_config(use_local=False), "EURUSD", interval="1h",
                     use_derivatives=False, strategies=strats,
                     bars=df, feats=compute_all(df))
    vote = next(v for v in sig.votes if v.strategy == "yt_sma_trend")
    assert vote.korean.startswith("유튜브")
    assert vote.direction in (-1, 0, 1) and 0.0 <= vote.weight <= 1.0


# ─── 파이프라인 ──────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    URL, "https://youtu.be/dQw4w9WgXcQ?t=10",
    "https://m.youtube.com/watch?list=PL1&v=dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
])
def test_youtube_urls_are_recognised(url):
    assert pipeline.video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    "https://example.com/watch?v=dQw4w9WgXcQ",       # 남의 서버
    "http://127.0.0.1:8787/watch?v=dQw4w9WgXcQ",     # 내부 주소
    "https://youtube.com.evil.io/watch?v=dQw4w9WgXcQ",
    "file:///etc/passwd", "not a url", "https://youtu.be/short", "",
])
def test_only_youtube_is_ever_contacted(url):
    with pytest.raises(pipeline.PipelineError):
        pipeline.video_id(url)


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(pipeline, "fetch_captions", lambda vid, log: "자막 내용")
    answers: list[str] = []
    prompts: list[str] = []

    def llm(system, user, cfg):
        prompts.append(user)
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr(pipeline, "llm_json", llm)
    return answers, prompts


def test_the_pipeline_saves_a_disabled_strategy(cfg, fake_llm):
    fake_llm[0].append(json.dumps(raw_spec()))
    record = pipeline.run(URL, cfg=cfg)
    assert record["enabled"] is False and record["source_url"] == URL
    assert store.get("sma_trend", cfg)["spec"]["entry_conditions"]["long"]


def test_a_rejected_answer_is_fed_back_once(cfg, fake_llm):
    answers, prompts = fake_llm
    answers += [json.dumps(raw_spec(entry_conditions={"long": "close > ema_20",
                                                      "short": ""})),
                json.dumps(raw_spec())]
    pipeline.run(URL, cfg=cfg)
    assert len(prompts) == 2 and "ema_20" in prompts[1]


def test_nothing_is_saved_when_the_llm_never_gets_it_right(cfg, fake_llm):
    fake_llm[0].append("{not json")
    with pytest.raises(pipeline.PipelineError):
        pipeline.run(URL, cfg=cfg)
    assert store.load_all(cfg) == []


def test_a_malicious_answer_cannot_become_a_strategy(cfg, fake_llm):
    evil = raw_spec(strategy_name="../../evil",
                    entry_conditions={"long": "__import__('os').system('x') > 0",
                                      "short": ""})
    fake_llm[0].append(json.dumps(evil))
    with pytest.raises(pipeline.PipelineError):
        pipeline.run(URL, cfg=cfg)
    assert not store.directory(cfg).exists() or not list(store.directory(cfg).iterdir())


def test_a_given_name_replaces_the_llm_name(cfg, fake_llm):
    fake_llm[0].append(json.dumps(raw_spec(strategy_name="llm_chosen")))
    record = pipeline.run(URL, name="my_pick", cfg=cfg)
    assert record["spec"]["strategy_name"] == "my_pick"
    assert [r["spec"]["strategy_name"] for r in store.load_all(cfg)] == ["my_pick"]
    with pytest.raises(pipeline.PipelineError):
        pipeline.run(URL, name="../evil", cfg=cfg)


def test_it_stops_before_the_network_without_a_key(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "fetch_captions",
                        lambda *a: pytest.fail("키가 없는데 네트워크를 탔다"))
    with pytest.raises(pipeline.PipelineError, match="API_KEY"):
        pipeline.run(URL, cfg=cfg)


def test_no_captions_falls_back_to_audio_and_cleans_up(cfg, monkeypatch):
    seen = {}

    def fake_download(vid, folder):
        audio = folder / "audio.m4a"
        audio.write_bytes(b"fake")
        seen["audio"] = audio
        return audio

    monkeypatch.setattr(pipeline, "fetch_captions", lambda vid, log: None)
    monkeypatch.setattr(pipeline, "download_audio", fake_download)
    monkeypatch.setattr(pipeline, "transcribe",
                        lambda path, cfg: "전사문" if path.exists() else "")
    assert pipeline.transcript("dQw4w9WgXcQ", cfg, lambda m: None) == "전사문"
    assert not seen["audio"].exists() and not seen["audio"].parent.exists()


# ─── API ─────────────────────────────────────────────────────────

@pytest.fixture
def client(cfg, monkeypatch, fake_llm):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.server import youtube_api

    monkeypatch.setattr(youtube_api, "load_config", lambda: cfg)
    app = FastAPI()
    app.include_router(youtube_api.router)
    fake_llm[0].append(json.dumps(raw_spec()))
    return TestClient(app)


def test_the_card_flow_extract_enable_disable_delete(client):
    assert client.get("/api/youtube/strategies").json()["strategies"] == []

    made = client.post("/api/youtube/extract", json={"url": URL}).json()
    assert made["strategy_name"] == "sma_trend" and made["enabled"] is False
    assert made["engine_name"] == "yt_sma_trend"

    on = client.post("/api/youtube/strategies/sma_trend/enable", json={"on": True}).json()
    assert on["enabled"] is True
    listed = client.get("/api/youtube/strategies").json()
    assert listed["llm_ready"] is True
    assert [s["enabled"] for s in listed["strategies"]] == [True]

    off = client.post("/api/youtube/strategies/sma_trend/enable", json={"on": False}).json()
    assert off["enabled"] is False

    assert client.delete("/api/youtube/strategies/sma_trend").json() == \
        {"deleted": "sma_trend"}
    assert client.get("/api/youtube/strategies").json()["strategies"] == []


def test_the_api_reports_failures_as_messages(client):
    assert client.post("/api/youtube/extract",
                       json={"url": "https://example.com/x"}).status_code == 400
    client.post("/api/youtube/extract", json={"url": URL})
    again = client.post("/api/youtube/extract", json={"url": URL})
    assert again.status_code == 400 and "이미" in again.json()["detail"]
    assert client.post("/api/youtube/strategies/nope/enable",
                       json={"on": True}).status_code == 404
    assert client.delete("/api/youtube/strategies/nope").status_code == 404
    assert client.post("/api/youtube/strategies/..%2Fx/enable",
                       json={"on": True}).status_code in (404, 405)


def test_another_web_page_cannot_switch_a_strategy_on(client, cfg):
    """
    브라우저에 열린 남의 페이지가 보낼 수 있는 것은 본문 없는 POST·폼·text/plain 이다
    (사전 요청 없이 나가는 "단순 요청"). 그걸로 전략이 켜지면 사람이 본 적 없는
    규칙이 투표에 들어간다. JSON 본문이 아니면 거부해야 한다.
    """
    store.save(raw_spec(), URL, cfg=cfg)
    path = "/api/youtube/strategies/sma_trend/enable"
    for attempt in (client.post(path), client.post(path + "?on=true"),
                    client.post(path, data={"on": "true"}),
                    client.post(path, content='{"on": true}',
                                headers={"Content-Type": "text/plain"})):
        assert attempt.status_code == 422
    assert store.get("sma_trend", cfg)["enabled"] is False


def test_the_screen_is_told_how_each_timeframe_is_handled(client, cfg):
    for name, tf in [("four_hour", "4h"), ("one_hour", "1h"),
                     ("five_min", "5m"), ("unspoken", None)]:
        store.save(raw_spec(strategy_name=name, timeframe=tf), URL, cfg=cfg)
    store.save(raw_spec(strategy_name="one_min", timeframe="1m"), URL, cfg=cfg)
    rows = client.get("/api/youtube/strategies").json()["strategies"]
    modes = {s["strategy_name"]: s["timeframe_mode"] for s in rows}
    assert modes == {"four_hour": "resampled", "one_hour": "same",
                     "five_min": "unsupported", "one_min": "unsupported",
                     "unspoken": "same"}
    # 기본 군의 봉으로 계산할 수 없는 것 중 5분 이상은 단기 군으로 간다. 1분봉은 갈 곳이 없다.
    groups = {s["strategy_name"]: s["group"] for s in rows}
    assert groups == {"four_hour": "main", "one_hour": "main", "unspoken": "main",
                      "five_min": "short", "one_min": None}


# ─── 영상이 말한 시간봉으로 계산한다 ─────────────────────────────

RISING = {"long": "close > prev_close", "short": ""}


def on_4h(hold_bars: int = 1) -> YoutubeStrategy:
    return strategy(hold_bars=hold_bars, timeframe="4h", indicators_needed=[],
                    entry_conditions=RISING)


def test_a_four_hour_strategy_is_judged_on_four_hour_bars():
    # 1시간봉 12개 = 4시간봉 3개 (종가 10 → 20 → 30)
    df = bars([10, 10, 10, 10,   11, 12, 13, 20,   21, 22, 23, 30])
    out = on_4h().generate(df, df)
    # 두 번째 4시간봉은 07시 봉이 닫혀야 완성된다 → 거기서부터 롱.
    # 1시간봉으로 봤다면 04시부터 매 봉 롱이었을 것이다.
    assert out["direction"].tolist() == [0] * 7 + [1] * 5
    one_hour = strategy(hold_bars=1, timeframe="1h", indicators_needed=[],
                        entry_conditions=RISING).generate(df, df)
    assert one_hour["direction"].tolist() == [0, 0, 0, 0] + [1] * 8


def test_an_unfinished_higher_bar_is_not_used():
    """
    두 번째 4시간봉이 아직 세 시간치뿐이다. 그 봉의 '종가'는 아직 없다 —
    지금까지의 값으로 판단하면 봉이 닫힐 때 답이 바뀐다.
    """
    df = bars([10, 10, 10, 10,   11, 12, 13])
    assert (on_4h().generate(df, df)["direction"] == 0).all()


def test_the_hold_is_counted_in_the_videos_bars():
    # 4시간봉 종가 10 → 20 → 20 → 20: 두 번째 봉에서만 조건이 참
    df = bars([10] * 4 + [11, 12, 13, 20] + [20] * 8)
    out = on_4h(hold_bars=2).generate(df, df)
    # 4시간봉 두 개(8시간) 동안 유지한다. 1시간봉 두 개가 아니다.
    assert out["direction"].tolist() == [0] * 7 + [1] * 8 + [0]


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_higher_timeframe_signals_never_look_ahead(timeframe):
    """뒤를 잘라 다시 계산해도 앞부분이 같아야 한다 — 묶는 과정에서 미래가 새면 안 된다."""
    df = make_candles(900, "1h", derivatives=False)
    s = strategy(hold_bars=3, timeframe=timeframe, indicators_needed=[
        {"name": "EMA", "params": {"period": 5}}, {"name": "RSI", "params": {"period": 5}}],
        entry_conditions={"long": "close > ema_5 and rsi_5 > 55",
                          "short": "close < ema_5 and rsi_5 < 45"})
    full = s.generate(df, df)
    assert (full["direction"] != 0).any()
    for cut in (301, 502, 643, 899):                    # 긴 봉 중간에서 자른다
        part = s.generate(df.iloc[:cut], df.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_a_video_that_names_no_timeframe_uses_the_engine_bars():
    df = make_candles(400, "1h", derivatives=False)
    unspoken = strategy(timeframe=None).generate(df, df)
    one_hour = strategy(timeframe="1h").generate(df, df)
    pd.testing.assert_frame_equal(unspoken, one_hour)
    assert spec.normalize(raw_spec(timeframe=None))["timeframe"] is None


def test_a_shorter_timeframe_than_the_bars_stays_silent():
    """1시간봉으로는 5분봉 조건을 계산할 수 없다. 엉뚱한 봉으로 답하지 않는다."""
    df = make_candles(400, "1h", derivatives=False)
    out = strategy(timeframe="5m", entry_conditions=RISING).generate(df, df)
    assert (out["direction"] == 0).all()


def test_the_engine_leaves_out_strategies_it_cannot_compute(tmp_path):
    raw = {"youtube": {"dir": str(tmp_path / "yt")}, "fx": {"interval": "1h"}}
    cfg = Config(raw)
    for name, tf in [("five_min", "5m"), ("four_hour", "4h"), ("unspoken", None)]:
        store.save(raw_spec(strategy_name=name, timeframe=tf), URL, cfg=cfg)
        store.set_enabled(name, True, cfg)
    names = {s.name for s in trader_with(raw, tmp_path).strategies()}
    assert {"yt_four_hour", "yt_unspoken"} <= names
    assert "yt_five_min" not in names

    # 엔진 봉을 낮추면 들어온다
    low = {**raw, "fx": {"interval": "5m"}}
    assert "yt_five_min" in {s.name for s in trader_with(low, tmp_path).strategies()}


# ─── 영상이 말하지 않은 방향은 만들지 않는다 ─────────────────────

def test_a_long_only_video_stays_long_only():
    """롱만 설명한 영상에서 숏을 뒤집어 만들면 그건 영상의 전략이 아니다."""
    one_sided = raw_spec(entry_conditions={"long": "close > sma_3", "short": ""})
    out = spec.parse(json.dumps(one_sided))
    assert out["entry_conditions"] == {"long": "close > sma_3", "short": ""}

    df = make_candles(400, "1h", derivatives=False)
    signals = YoutubeStrategy(out).generate(df, df)
    assert (signals["direction"] >= 0).all()


def test_the_prompt_forbids_inventing_the_other_side():
    prompt = pipeline.SYSTEM_PROMPT
    assert "전혀 언급하지 않으면" in prompt and "지어내거나 뒤집어 만들지 않는다" in prompt
    assert "언급했을 때만 대칭으로" in prompt


def test_the_main_app_serves_the_youtube_routes():
    from src.server.app import app
    assert {"/api/youtube/strategies", "/api/youtube/extract"} <= set(app.openapi()["paths"])
