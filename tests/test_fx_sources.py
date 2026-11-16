"""
거래의 출처를 앱별로 가른다.

**왜 이 파일이 있나.** 화면이 거래를 `엔진 / 수동` 두 칸으로만 갈랐다.
그 라벨을 글자 그대로 읽고 "손으로 연 거래 76건에 손절이 하나도 없다 —
처분효과다"라는 진단을 냈는데 틀렸다. 그 76건은 네 개 앱이 낸 것이었고,
가장 큰 손실을 낸 앱은 **열린 포지션에 손절을 걸어 두고 있었다.**

틀린 진단의 원인은 데이터가 아니라 **화면의 두 칸짜리 라벨**이었다.
여기서 지키는 것:

    · 등록된 앱은 magic 으로 정확히 갈린다
    · **모르는 magic 이 '손클릭'으로 떨어지지 않는다** (이게 핵심이다)
    · magic 이 0 일 때만 손클릭이다
    · 한 왕복 안에서 코멘트가 덮어써져도 출처가 안 바뀐다
    · 요약이 앱별로 따로 선다
"""

from __future__ import annotations

import pytest

from src.fx import sources as S


def _deal(magic: int = 0, comment: str = "") -> dict:
    return {"magic": magic, "comment": comment}


# ─── 등록된 앱 ───────────────────────────────────────────────────

@pytest.mark.parametrize("magic,key,label", [
    (S.MAGIC,  "engine",     "레짐 퀀트"),
    (20260819, "larry",      "래리 윌리엄스"),
    (26081900, "pair",       "페어트레이딩"),
    (770316,   "allweather", "전천후 트레이더"),
    (940715,   "gates",      "4관문 수동주문"),
])
def test_a_registered_app_is_named_by_its_magic(magic, key, label):
    """실측한 번호들이다 — 데모 계좌 한 개, 최근 120일 체결 158건."""
    s = S.classify(magic, "")
    assert s.key == key
    assert s.label == label


def test_no_two_apps_claim_the_same_number():
    """
    한 번호를 두 앱이 주장하면 REGISTRY 순서가 귀속을 정한다 — 즉 **줄
    순서를 바꾸는 것만으로 과거 성적이 바뀐다.** 조용히 일어나면 못 잡는다.
    """
    seen: dict[int, str] = {}
    for src in S.REGISTRY:
        for magic in src.magics:
            assert magic not in seen, f"magic {magic}: {seen[magic]} vs {src.key}"
            seen[magic] = src.key


def test_the_labels_match_the_account_registry():
    """
    이름은 지어내지 않는다 — 계좌 등록부(trading-tools/mt5-who.py)에 적힌
    이름을 쓴다. 코멘트의 `gsv` 를 앱 이름으로 읽고 '그록데스크'라 부른
    적이 있다. `gsv` 는 래리 앱의 **전략 이름**이다.
    """
    labels = {s.magics[0]: s.label for s in S.REGISTRY}
    assert labels[20260819] == "래리 윌리엄스"
    assert labels[940715] == "4관문 수동주문"
    assert labels[S.MAGIC] == "레짐 퀀트"


def test_the_engine_number_is_not_duplicated_anywhere():
    """
    주문을 내는 쪽과 내역을 읽는 쪽이 같은 번호를 봐야 한다. 양쪽에 따로
    적어 두면 한쪽만 바뀌었을 때 과거 거래가 조용히 남의 것이 된다.
    """
    from src.fx import broker, mt5
    assert mt5.MAGIC is S.MAGIC and broker.MAGIC is S.MAGIC
    assert mt5.PREFIX is S.PREFIX and broker.PREFIX is S.PREFIX


# ─── 모르는 번호 ─────────────────────────────────────────────────

def test_an_unknown_magic_is_not_called_hand_placed():
    """
    **이 파일의 존재 이유다.** 등록 안 된 번호를 '손클릭'으로 떨어뜨리면,
    새 앱이 붙은 걸 아무도 모른 채 그 거래를 사람 손거래로 읽는다.
    이미 한 번 그렇게 읽고 틀린 진단을 냈다.
    """
    s = S.classify(123456789, "")
    assert s.key != S.HAND.key
    assert "미등록" in s.label
    assert "123456789" in s.label, "어떤 번호인지 모르면 등록할 수가 없다"


def test_only_a_zero_magic_is_hand_placed():
    """MT5 창에서 사람이 직접 낸 주문에만 번호가 없다."""
    assert S.classify(0, "").key == "hand"
    assert S.classify(0, "").label == "사람이 직접"


def test_an_unknown_magic_keeps_its_own_identity():
    """번호가 다르면 다른 앱이다 — 한 칸에 뭉뚱그리면 또 같은 실수가 난다."""
    assert S.classify(111, "").key != S.classify(222, "").key


# ─── 코멘트 단서 ─────────────────────────────────────────────────

def test_the_tag_prefix_identifies_us_when_magic_is_missing():
    """일부 브로커는 magic 을 안 실어 준다. 태그 접두사가 두 번째 단서다."""
    assert S.classify(0, S.PREFIX + "EURUSD2609160100").key == "engine"


def test_magic_wins_over_a_comment():
    """
    코멘트는 브로커가 `[sl 1.2345]` 같은 걸로 덮어쓴다 — 믿을 수 없다.
    번호가 있으면 번호가 먼저다.
    """
    assert S.classify(20260819, "rq-looks-like-ours").key == "larry"


def test_a_loose_prefix_does_not_swallow_other_apps():
    """
    '종목이름으로 시작하니까' 같은 느슨한 규칙을 넣으면 남의 거래를 빨아들인다.
    **느슨한 귀속이 바로 이 파일이 고치려는 문제다.**
    """
    assert S.classify(0, "US30Cash manual entry").key == "hand"
    assert S.classify(0, "GOLD scalp").key == "hand"


# ─── 왕복 한 건 ──────────────────────────────────────────────────

def test_an_overwritten_exit_comment_does_not_split_a_trade():
    """
    진입은 `rq...`, 청산은 브로커가 `[sl 1.34263]` 으로 덮어썼다 — 실제로
    계좌에 있는 모양이다. 줄 하나만 보면 같은 거래가 두 출처로 갈린다.
    """
    g = [_deal(0, S.PREFIX + "GBPUSD2609160100"), _deal(0, "[sl 1.34263]")]
    assert S.of_deals(g).key == "engine"
    assert S.of_deals(list(reversed(g))).key == "engine"


def test_a_bot_trade_with_one_blank_deal_is_still_that_bot():
    """magic 이 빈 체결이 섞였다고 남의 앱 거래가 손클릭이 되면 안 된다."""
    assert S.of_deals([_deal(20260819, "GOLD|gsv|L|2026-08-27"),
                       _deal(0, "reversal exit")]).key == "larry"


def test_a_trade_with_no_magic_at_all_is_hand_placed():
    assert S.of_deals([_deal(0, ""), _deal(0, "")]).key == "hand"


def test_an_empty_group_does_not_crash():
    assert S.of_deals([]).key == "hand"


# ─── 연 쪽이 주인이다 ────────────────────────────────────────────

def test_the_app_that_opened_the_position_owns_the_trade():
    """
    실측(2026-11-14): 80건 중 2건에서 **다른 앱이 남의 포지션을 닫았다.**
    그록데스크가 연 US100Cash 를 추세알림이 닫았고, 페어트레이딩이 연
    EURJPY 도 추세알림이 닫았다.

    손익은 **연 앱의 판단**에 달렸으므로 진입 체결이 주인을 정한다. 체결
    순서대로 아무거나 고르면 같은 계좌를 두 번 재면서 답이 달라진다.
    """
    ins = [_deal(20260819, "US100Cash|gsv|L|2026-08-27")]
    outs = [_deal(940715, "forex-alert close")]
    assert S.of_deals(ins + outs, ins).key == "larry"


def test_a_blank_entry_falls_back_to_the_whole_trade():
    """
    진입 체결에 magic 도 태그도 없는 경우가 있다(실측 7건: 진입은 비어
    있고 청산만 `forex-alert close`). 그때 '모른다'고 하면 남의 봇 거래가
    **손클릭으로 둔갑한다** — 바로 그 오독을 고치려고 만든 모듈이다.
    """
    ins = [_deal(0, "")]
    outs = [_deal(940715, "forex-alert close")]
    assert S.of_deals(ins + outs, ins).key == "gates"


def test_a_genuinely_hand_placed_trade_stays_hand_placed():
    """되돌림 방지 — 전부 비어 있으면 그건 정말 손클릭이다."""
    ins = [_deal(0, "")]
    assert S.of_deals(ins + [_deal(0, "")], ins).key == "hand"


def test_a_cross_app_close_is_reported(monkeypatch):
    """
    앱들이 서로의 포지션을 건드리는 건 조용히 지나갈 일이 아니다.
    화면이 손익을 **연 앱 성적**으로 세고 있으므로 더욱 그렇다.
    """
    from tests.test_fx_history import deal
    from types import SimpleNamespace
    from src.fx import mt5 as m

    ds = [deal(1, entry=0, magic=20260819, comment="US100Cash|gsv|L|2026-08-27",
               symbol="US100Cash"),
          deal(1, entry=1, kind=1, magic=940715, comment="forex-alert close",
               profit=-39.86, symbol="US100Cash")]
    monkeypatch.setattr(m, "initialise", lambda: True)
    monkeypatch.setattr(m, "_mt5", lambda: SimpleNamespace(
        history_deals_get=lambda a, b: tuple(ds)))

    t = m.closed_trades()[0]
    assert t["source"] == "larry"              # 연 앱이 주인
    assert t["closed_by"] == "4관문 수동주문"       # 닫은 앱도 밝힌다


def test_a_normal_trade_has_nothing_to_report(monkeypatch):
    """같은 앱이 열고 닫았으면 빈칸이어야 한다 — 매 줄에 뜨면 신호가 죽는다."""
    from tests.test_fx_history import deal
    from types import SimpleNamespace
    from src.fx import mt5 as m

    ds = [deal(1, entry=0, magic=20260819, comment="GOLD|gsv|L|2026-08-27"),
          deal(1, entry=1, kind=1, magic=20260819, comment="", profit=5.0)]
    monkeypatch.setattr(m, "initialise", lambda: True)
    monkeypatch.setattr(m, "_mt5", lambda: SimpleNamespace(
        history_deals_get=lambda a, b: tuple(ds)))
    assert m.closed_trades()[0]["closed_by"] is None


# ─── 내역에 실리는가 ─────────────────────────────────────────────

def test_closed_trades_carry_the_app_name(monkeypatch):
    """화면이 읽을 수 있게 거래마다 앱 이름이 실려야 한다."""
    from tests.test_fx_history import deal         # 같은 가짜 체결을 쓴다
    from types import SimpleNamespace
    from src.fx import mt5 as m

    ds = [deal(1, entry=0, magic=20260819, comment="GOLD|gsv|L|2026-08-27",
               symbol="GOLD"),
          deal(1, entry=1, kind=1, magic=20260819, comment="", profit=-215.47,
               symbol="GOLD")]
    monkeypatch.setattr(m, "initialise", lambda: True)
    monkeypatch.setattr(m, "_mt5", lambda: SimpleNamespace(
        history_deals_get=lambda a, b: tuple(ds)))

    t = m.closed_trades()[0]
    assert t["source"] == "larry"
    assert t["source_label"] == "래리 윌리엄스"
    assert t["magic"] == 20260819
    # 그리고 **우리 것이 아니다** — 이 구분은 그대로여야 한다.
    assert t["ours"] is False


# ─── 요약 ────────────────────────────────────────────────────────

def _t(source: str, label: str, profit: float) -> dict:
    return {"ours": source == "engine", "source": source, "source_label": label,
            "profit": profit, "commission": 0.0, "swap": 0.0,
            "r_multiple": None, "closed_at": "2026-11-05T10:00:00+00:00"}


def test_the_summary_stands_each_app_up_separately():
    """
    '엔진 / 그 외' 두 칸만으로는 그 외 −598 이 **누구 손실인지** 알 수 없다.
    앱별로 서야 "어느 앱이 돈을 잃는가"가 한눈에 보인다.
    """
    from src.server import fx_api
    rows = fx_api._history_summary([
        _t("larry", "래리 윌리엄스", -215.47),
        _t("larry", "래리 윌리엄스", -212.0),
        _t("engine", "엔진", -535.0),
        _t("hand", "손클릭", -1.0),
    ])["by_source"]

    got = {r["key"]: r for r in rows}
    assert got["larry"]["trades"] == 2
    assert got["larry"]["net"] == pytest.approx(-427.47)
    assert got["engine"]["net"] == pytest.approx(-535.0)
    assert got["hand"]["trades"] == 1
    # 활동이 많은 앱이 위에 온다
    assert rows[0]["key"] == "larry"


def test_the_app_totals_add_up_to_the_whole():
    """
    앱별 합이 전체와 어긋나면 어느 쪽이 맞는지 알 수 없고, 둘 다 못 믿는다.
    """
    from src.server import fx_api
    trades = [_t("larry", "래리 윌리엄스", -7.78), _t("pair", "페어트레이딩", 9.85),
              _t("engine", "엔진", -601.45), _t("magic999", "미등록(magic=999)", 3.0)]
    s = fx_api._history_summary(trades)
    assert sum(r["trades"] for r in s["by_source"]) == s["all"]["trades"]
    assert (pytest.approx(sum(r["net"] for r in s["by_source"]), abs=0.01)
            == s["all"]["net"])


def test_an_unregistered_app_shows_up_in_the_summary_instead_of_hiding():
    """미등록 앱이 '손클릭' 칸에 섞여 들어가면 아무도 눈치채지 못한다."""
    from src.server import fx_api
    rows = fx_api._history_summary([_t("magic999", "미등록(magic=999)", -5.0)])
    keys = {r["key"] for r in rows["by_source"]}
    assert "magic999" in keys and "hand" not in keys


def test_an_empty_history_has_no_app_rows():
    from src.server import fx_api
    assert fx_api._history_summary([])["by_source"] == []
