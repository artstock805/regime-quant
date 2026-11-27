"""
FX 자동매매 루프 — 신호를 보고 MT5 로 진입·청산한다.

    봉 마감 → 신호 계산 → 가드 통과 → 주문 → 장부 기록 → 차트 이동

**검증 결과를 여기에 적어 둔다.** 이 루프가 돌리는 신호는 FX 에서 우위가
확인되지 않았다. 캔들만 쓰는 전략 10개를 14개 통화쌍 · 4.8년으로 검증했고,
**비용을 하나도 빼지 않은 총우위조차** 1시간봉 +0.0019R (군집 t=0.46),
4시간봉 +0.0035R (t=0.52), 일봉 +0.0049R (t=0.51) 로 전부 판정불가였다.
비용을 넣으면 1시간봉은 -0.0525R (t=-12.29) 로 확실한 마이너스다.

그래서 기본값이 이렇다.
    · 엔진은 **꺼진 채로 시작한다.** 사람이 켜야 돈다.
    · 주문은 **데모 계좌에만** 나간다 (broker.assert_tradable).
    · 화면이 이 판정을 숨기지 않는다.

이 셋을 다 넘기고 실계좌에서 켜는 것은 사람의 결정이다. 다만 그 결정을 하기
전에 무엇을 아는 상태였는지는 분명해야 한다.

**봉 마감에만 판단한다.** 진행 중인 봉으로 판단하면 신호가 매초 흔들리고
백테스트 규칙과 어긋난다. 같은 봉에서 두 번 진입하지 않는 것은 broker.tag 가
막는다.

**MT5 가 진실이다.** 매 회 시작할 때 포지션을 대조한다. 손절이 브로커 쪽에서
체결됐거나 사람이 터미널에서 닫았을 수 있고, 그걸 모르면 이미 닫힌 것을
들고 있다고 착각한 채 동시 보유 한도를 채운다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..backtest import ExecParams, WeightParams
from ..config import Config, Costs, load_config
from ..execution.guards import RiskGuard, RiskLimits
from ..execution.journal import Journal
from ..indicators import compute_all
from ..live.signals import SymbolSignal, signal_for
from ..strategies import build_all
from ..youtube import enabled_strategies as youtube_strategies
from ..youtube import short_interval, short_strategies
from . import broker
from . import defense as fx_defense
from . import keepawake
from . import watchlist as fx_watchlist
from .mt5 import MT5Error, bars, account, cash_flows, positions, symbol_info
from .symbols import Pair, parse

# 검증 결과. 화면과 장부가 같은 문장을 쓰도록 한 곳에 둔다.
VERDICT = {
    "passed": False,
    "headline": "FX 에서 우위가 확인되지 않았습니다",
    "detail": (
        "캔들 전략 10개 · 14개 통화쌍 · 2021-11~2026-09 검증 결과, "
        "비용을 빼지 않은 총우위조차 판정불가였습니다 "
        "(1시간봉 +0.0019R, 군집 t=0.46). 비용을 넣으면 -0.0525R (t=-12.29) 입니다. "
        "데모에서 동작을 확인하는 용도로 쓰세요."
    ),
}

# 지표가 데워지는 데 필요한 여유분. 학습 구간 앞에 이만큼이 더 있어야
# 첫 봉부터 지표가 유효하다.
WARMUP_BARS = 300

# FX 는 주 5일 24시간 돈다 — 주말이 비므로 달력 일수에 5/7 을 곱한다.
# 봉이 부족하면 **엔진이 영원히 거래하지 않는다.** 학습 구간이 짧아 레짐별
# 거래 수가 min_trades(20)에 못 미치고, 그러면 모든 전략의 가중치가 0 이 되어
# 점수가 항상 0.000 이 나온다. 화면은 멀쩡해 보이는데 아무 일도 안 일어난다.
_BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48,
                 "1h": 24, "4h": 6, "1d": 1}

# 새 봉이 떴는지 확인하는 주기(초). 봉 두 개만 받아 보는 값싼 호출이다.
# **시계 계산으로 봉 마감을 맞추지 않는다.** MT5 는 서버 시간을 주고(GMT+2/+3,
# 서머타임으로 바뀐다) 주말에는 장이 닫힌다. "다음 봉 마감까지 몇 초"를 계산하면
# 그 셋을 다 맞춰야 하고 하나만 틀려도 같은 봉을 두 번 보거나 건너뛴다.
# **새 봉이 실제로 나타났는지 보는 쪽**이 그 전부를 원천 회피한다.
POLL_SECONDS = 30.0

# 입금액을 다시 읽는 주기(초). 계좌 손실 한도(_seed_stop)의 기준값이다.
SEED_REFRESH_SECONDS = 600.0


@dataclass
class EngineState:
    running: bool = False
    mode: str = "demo"
    last_run: str | None = None
    last_error: str | None = None
    scanned: int = 0
    # 루프가 실제로 돌고 있는지 화면이 알아야 한다. "시작"을 눌렀는데
    # cycles 가 0 에서 안 늘면 뭔가 잘못된 것이다.
    cycles: int = 0
    checks: int = 0
    last_bar: str | None = None
    next_check: str | None = None
    signals: list[dict] = field(default_factory=list)
    opened: list[dict] = field(default_factory=list)
    closed: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Group:
    """
    투표 군. 한 군 안의 전략끼리만 투표하고, 그 군만의 포지션을 열고 닫는다.

        기본 군   기존 전략 + 엔진 봉 이상의 유튜브 전략. fx.interval 로 돈다.
        단기 군   엔진 봉보다 짧은 유튜브 전략(5·15·30분). 그중 가장 짧은 봉으로 돈다.

    **왜 가르는가.** 1시간봉으로는 15분봉 조건을 계산할 수 없고, 기본 군을 15분봉으로
    내리면 1시간봉으로 검증한 기존 전략이 전부 다른 것이 된다. 그래서 짧은 전략은
    자기 봉으로 따로 돌린다.

    **두 군은 같은 계좌를 쓴다.** 그래서
      · 포지션의 주인은 주문 태그(code)에서 읽는다. 각 군은 자기 것만 세고 닫는다.
        같은 종목에 두 군이 따로, 반대 방향으로도 들 수 있다(헤지 계좌).
      · 위험 가드는 **함께 쓴다.** 킬 스위치·일일 손실 한도·분당 주문 한도는
        계좌의 것이다. 군마다 따로 두면 한 군이 한도를 넘겨도 다른 군이 계속 진입한다.
      · 동시 보유 한도(max_positions)만 군마다 따로 센다.

    단기 군은 **검증된 적이 없다.** 이 프로젝트는 짧은 봉일수록 비용이 1R 을 크게
    먹는다는 것을 실측했다(5분봉 13~27%). 발언권은 기본 군과 같은 방식(최근 학습
    구간의 비용 차감 성적)으로 정하므로, 비용을 못 이기는 전략은 가중치 0 이 된다.
    """
    name: str
    code: str
    label: str


MAIN = Group("main", "", "기본 군")
SHORT = Group("short", broker.SHORT_CODE, "단기 군")
GROUPS = {g.name: g for g in (MAIN, SHORT)}


def entry_votes(sig: Any) -> list[dict]:
    """
    이 진입에 **찬성한 전략과 기여도.** 기여도가 큰 순이다 — 첫 줄이 이 진입을
    가장 세게 민 전략이다.

    진입은 한 전략이 아니라 투표가 정한다. 그래서 "어떤 전략으로 들어갔나"의 답은
    목록이다. 발언권이 있고(가중치 > 0) 진입 방향과 같은 쪽으로 말한 것만 담는다.
    반대하거나 침묵한 전략, 발언권이 없는 전략은 이 진입의 이유가 아니다.

    기여도는 가중치 × 확신도 × 방향이다. 엔진이 확신도를 백분위로 보정하면
    (calibrate_strength) 실제 점수에 들어간 값과 조금 다를 수 있다. 순서를 읽는
    데 쓰고, 합이 점수와 정확히 맞는다고 읽지 않는다.
    """
    direction = int(getattr(sig, "direction", 0) or 0)
    out = []
    for vote in getattr(sig, "votes", None) or []:
        contribution = float(vote.contribution)
        if not vote.has_voice or contribution * direction <= 0:
            continue
        out.append({"strategy": vote.strategy, "korean": vote.korean,
                    "weight": float(vote.weight), "strength": float(vote.strength),
                    "contribution": round(contribution, 4)})
    return sorted(out, key=lambda row: -abs(row["contribution"]))


class _GroupJournal:
    """장부에 군 이름을 붙여 쓴다. 두 군이 한 장부를 쓰므로 누가 한 일인지 남아야 한다."""

    def __init__(self, inner: Journal, group: str) -> None:
        self._inner, self._group = inner, group

    def write(self, kind: str, **fields: Any) -> dict:
        return self._inner.write(kind, group=self._group, **fields)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class FxTrader:
    """
    한 번에 한 스레드만 돈다. MT5 는 스레드 안전하지 않고, 루프가 겹치면
    같은 신호로 두 번 주문할 수 있다.
    """

    # 클래스 기본값으로도 둔다. __init__ 을 거치지 않는 대역(테스트의 FakeEngine)이
    # 군 없이 만들어져도 기본 군으로 동작해야 한다.
    group: Group = MAIN
    # 루프가 절전 방지를 쥐고 있는가 (루프 스레드가 세우고 내린다)
    _awake: bool = False
    # 입금액(입금 − 출금)과 읽은 시각. 계좌의 것이라 군이 함께 쓴다.
    _seed_cache: tuple[float, float] | None = None

    def __init__(self, cfg: Config | None = None,
                 journal: Journal | None = None,
                 guard: RiskGuard | None = None,
                 group: Group = MAIN) -> None:
        self.cfg = cfg or load_config()
        self.group = group
        journal = journal or Journal()
        self.journal = journal if group is MAIN else _GroupJournal(journal, group.name)
        self.guard = guard or RiskGuard(RiskLimits(
            risk_per_trade=float(self.cfg.get("risk.risk_per_trade", 0.01)),
            max_positions=int(self.cfg.get("live.max_positions", 3)),
            daily_loss_limit=float(self.cfg.get("risk.daily_loss_limit", 0.03)),
            min_balance=float(self.cfg.get("live.min_balance", 100.0)),
            # FX 는 명목가가 잔고를 넘는 것이 정상이다. 0.1랏 EUR/USD 는
            # 11,000 달러어치인데 그게 1만 달러 계좌의 평범한 크기다.
            # 암호화폐 기본값(잔고의 1배)을 그대로 쓰면 아무것도 못 산다.
            # 그래도 상한은 둔다 — 랏 계산 버그가 레버리지 사고로 번지는
            # 것을 막는 것이 이 한도의 목적이고, 그건 FX 에서도 같다.
            max_notional_ratio=float(self.cfg.get("fx.max_leverage", 30.0)),
        ))
        self.state = EngineState()
        self._lock = threading.Lock()
        self._known: dict[str, dict] = {}
        # 종목별로 마지막 마감봉이 급변동이었는지 (+1 상승, -1 하락, 0 아님)
        self._shock: dict[str, int] = {}
        # 종목별로 마지막 대응 주문을 낸 봉. 한 봉에 한 번만 대응한다.
        self._defended: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ─── 설정 ────────────────────────────────────────────────────

    @property
    def interval(self) -> str | None:
        """
        이 군이 도는 봉. 기본 군은 설정값(fx.interval)이다.

        단기 군은 **켜 둔 단기 전략 중 가장 짧은 시간봉**이고, 하나도 없으면
        None 이다 — 돌 것이 없으므로 루프가 아무것도 하지 않는다.
        """
        if self.group is SHORT:
            return short_interval(self.cfg)
        return str(self.cfg.get("fx.interval", "1h"))

    @property
    def allow_live(self) -> bool:
        return bool(self.cfg.get("fx.allow_live", False))

    def defense(self) -> fx_defense.Rules:
        """역행 대응 규칙 (마팅게일·양방 대응). 매 바퀴 설정에서 다시 읽는다."""
        return fx_defense.rules(self.cfg)

    def watchlist(self) -> list[Pair]:
        """
        감시할 통화쌍. **기본은 전체**이고 화면에서 고르면 그것만 본다
        (src/fx/watchlist.py). 매 바퀴 다시 읽으므로 화면에서 바꾼 선택이
        재시작 없이 다음 바퀴에 반영된다.
        """
        from .mt5 import available_symbols
        return fx_watchlist.pairs(self.cfg, available_symbols(only_forex=True))

    def weight_params(self) -> WeightParams:
        """
        FX 는 **자기 임계값을 쓴다.** 암호화폐 값(0.85)을 그대로 가져오면
        점수가 한 번도 그 문턱을 넘지 못한다.

        이유는 구조적이다. 암호화폐에서는 taker_flow_momentum 한 전략이 가중치를
        거의 독식했고, 그 하나가 강하게 말하면 합산 점수가 곧바로 높아졌다.
        FX 에서는 가중치가 서너 전략에 갈리는데(실측 0.41/0.18/0.41) 이들이
        동시에 같은 방향으로 강하게 말하는 일이 드물다. EUR/USD 3,488봉(203일)
        실측에서 |점수|의 95분위가 0.17 이었고 0.85 를 넘은 봉은 **0개**였다.

        그래서 이 값은 수익률을 맞추려고 고른 것이 아니라 **신호가 나오기는
        하는 수준**으로 고른 것이다. 0.35 에서 1.89% (사흘에 한 번꼴).
        어떤 임계값도 이 신호를 수익성 있게 만들지는 못한다 — 우위 자체가
        확인되지 않았다 (VERDICT 참고).
        """
        return WeightParams(
            entry_threshold=float(self.cfg.get("fx.entry_threshold", 0.35)),
            calibrate_strength=bool(self.cfg.get("fx.calibrate_strength", True)),
        )

    def bars_needed(self) -> int:
        """
        받아 올 봉 개수. **학습 구간을 다 덮어야 한다.**

        signal_for 는 최근 train_months(기본 6개월)에서 전략을 돌려 가중치를
        뽑는다. 그보다 적게 받아 오면 레짐별 거래 수가 min_trades 에 못 미쳐
        모든 가중치가 0 이 되고, 엔진은 조용히 아무것도 하지 않는다.
        고장 난 것처럼 보이지 않아서 더 나쁘다.
        """
        train_days = int(self.cfg.get("backtest.walk_forward.train_months", 6)) * 31
        per_day = _BARS_PER_DAY.get(self.interval, 24)
        # 주말은 장이 닫힌다. 달력 일수를 그대로 곱하면 실제보다 많이 잡는다.
        return int(train_days * (5 / 7) * per_day) + WARMUP_BARS

    def strategies(self):
        """
        FX 에서 돌릴 수 있는 전략만. 파생 데이터를 요구하는 것은 제외한다 —
        FX 에는 자금조달률도 미결제약정도 테이커 비율도 없고, 없는 열을 0 으로
        채우면 신호가 아니라 잡음이 된다.

        **거름망이 둘인 이유.** `fx.strategies` 는 허용목록이라 비워 두면 전부
        쓴다. 특정 전략 하나를 빼려고 나머지 열을 적으면, 나중에 전략을 추가할
        때 목록에 넣는 것을 잊고 **조용히 빠진다**. 그래서 금지목록
        `fx.exclude_strategies` 를 따로 둔다 — 새 전략은 기본으로 들어오고,
        빼는 것만 이름으로 명시된다.
        """
        names = set(self.cfg.get("fx.strategies") or [])
        banned = set(self.cfg.get("fx.exclude_strategies") or [])
        if self.group is SHORT:
            # 단기 군에는 기존 전략을 넣지 않는다. 그것들은 1시간봉으로 만들고
            # 검증한 것이라, 5분봉 위에서 돌리면 같은 이름의 다른 전략이 된다.
            return [s for s in short_strategies(self.cfg) if s.name not in banned]
        out = []
        for s in build_all():
            if any(str(n).startswith(("perp_", "funding", "oi_", "basis"))
                   for n in getattr(s, "needs", ())):
                continue
            if names and s.name not in names:
                continue
            if s.name in banned:
                continue
            out.append(s)
        # 화면에서 켠 유튜브 전략. **허용목록(fx.strategies)은 적용하지 않는다** —
        # 켜는 것 자체가 사람의 선택이고, 허용목록에 이름을 또 적게 하면 켜 놓고도
        # 조용히 빠진다. 금지목록은 그대로 이긴다. 매 바퀴 저장소를 다시 읽으므로
        # 토글이 재시작 없이 반영된다.
        out.extend(s for s in youtube_strategies(self.cfg) if s.name not in banned)
        return out

    # ─── 시작·정지 ───────────────────────────────────────────────

    def start(self) -> dict:
        with self._lock:
            if self.state.running:
                return self.status()
            acct = account()
            self.state.running = True
            self.state.mode = "demo" if acct.is_demo else "live"
            self.journal.write("engine", event="start", venue="MT5",
                               login=acct.login, server=acct.server,
                               is_demo=acct.is_demo, interval=self.interval,
                               allow_live=self.allow_live,
                               verdict=VERDICT["headline"])
            self._spawn()
            return self.status()

    def stop(self, reason: str = "사용자 정지") -> dict:
        with self._lock:
            self.state.running = False
            self._stop.set()
            self.journal.write("engine", event="stop", reason=reason)
            return self.status()

    def kill(self, reason: str = "킬 스위치") -> dict:
        """
        **멈추라고 했으면 조건 없이 멈춘다.** 진입을 막고 루프를 세운다.
        열린 포지션은 건드리지 않는다 — 청산은 flatten 이 따로 한다.
        """
        self.guard.engage_kill(reason)
        self.state.running = False
        self._stop.set()
        self.journal.write("engine", event="kill", reason=reason)
        return self.status()

    def release(self) -> dict:
        self.guard.release_kill()
        self.journal.write("engine", event="release")
        return self.status()

    def flatten(self, reason: str = "전량 청산") -> list[dict]:
        """우리가 연 포지션 전부 청산. 사람이 연 것은 건드리지 않는다."""
        out = broker.close_all(allow_live=self.allow_live, reason="rqflat")
        for row in out:
            self._journal_close(reason, row)
        return out

    def _journal_close(self, why: str, row: dict) -> None:
        """
        청산을 장부에 남긴다.

        broker 가 돌려주는 행에도 `reason` 이 있다 — 청산 주문 comment 에 넣은 표식
        ("rqexit")이거나 실패 사유다. 우리 사유와 이름이 겹치므로 그쪽을
        `broker_reason` 으로 옮긴다.

        **실제로 있었던 일이다.** 행을 그대로 풀어 넣자 키워드가 겹쳐 TypeError 가
        났다. 포지션은 이미 닫혔는데 장부에는 남지 않았고, 그 바퀴에서 뒤 순서의
        종목들이 통째로 건너뛰어졌다. 테스트의 가짜 브로커가 `reason` 을 돌려주지
        않아 실제 청산이 처음 일어난 날에야 드러났다.
        """
        fields = {k: v for k, v in row.items() if k != "reason"}
        self.journal.write("close", reason=why, broker_reason=row.get("reason"), **fields)

    # ─── 루프 ────────────────────────────────────────────────────

    def _spawn(self) -> None:
        """루프 스레드를 띄운다. 이미 살아 있으면 두 번 띄우지 않는다."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name=f"fx-engine-{self.group.name}",
                                        daemon=True)
        self._thread.start()

    def _newest_bar(self) -> str | None:
        """
        기준 종목의 **마지막 마감봉** 시각. 못 읽으면 None.

        `bars()` 가 진행 중인 봉을 버리고 주므로, 이 값이 바뀌면 새 봉이
        마감된 것이다. 별도의 settle 대기가 필요 없다.
        """
        interval = self.interval
        if interval is None:          # 단기 군인데 켜 둔 단기 전략이 없다
            return None
        pairs = self.watchlist()
        if not pairs:
            return None
        df = bars(pairs[0].mt5, interval, 3)
        if df.empty:
            return None
        return df.index[-1].isoformat()

    def _loop(self) -> None:
        """
        루프 스레드의 몸통. **도는 동안 PC 가 절전에 들어가지 않게 한다**
        (src/fx/keepawake.py). 절전 중에는 엔진이 서고, 그러면 진입도 신호 청산도
        역행 대응도 멈춘다. 요청은 이 스레드에 묶여 있어 루프가 끝나면 풀린다.
        """
        self._awake = keepawake.hold()
        try:
            self._poll()
        finally:
            keepawake.release()
            self._awake = False

    def _poll(self) -> None:
        """
        새 봉이 마감될 때마다 한 바퀴 돈다.

        **한 바퀴가 실패해도 루프는 죽지 않는다.** MT5 가 잠깐 끊기거나 한
        종목이 말썽인 것으로 매매가 통째로 멈추면, 열린 포지션을 관리할
        주체가 사라진다. 손절은 브로커에 걸려 있어 지켜지지만 목표가 이동이나
        재평가가 멈춘다.

        주말에는 새 봉이 안 뜨므로 조용히 아무것도 하지 않는다 — 오류를
        장부에 쌓지 않는다.
        """
        while not self._stop.is_set():
            try:
                if self._seed_stop():       # 계좌 손실 한도 — 봉 마감을 기다리지 않는다
                    break
                newest = self._newest_bar()
                self.state.checks += 1
                if newest and newest != self.state.last_bar:
                    first = self.state.last_bar is None
                    self.state.last_bar = newest
                    # **처음 한 바퀴는 건너뛴다.** 시작 시점의 마지막 봉은
                    # 이미 지나간 봉이고, 그걸로 진입하면 백테스트가 가정한
                    # "봉 마감 직후 체결"보다 한참 늦은 가격에 들어간다.
                    if not first:
                        self.run_once()
                        self.state.cycles += 1
            except Exception as exc:
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.journal.write("error", where="loop",
                                   message=self.state.last_error)
            self.state.next_check = datetime.now(timezone.utc).isoformat()
            if self._stop.wait(POLL_SECONDS):
                break

    # ─── 계좌 손실 한도 ──────────────────────────────────────────

    def _seed(self) -> float:
        """
        입금액(입금 − 출금). 손실 한도의 기준이다. 모르면 0.

        거래 내역 전체를 훑어야 나오는 값이라 10분에 한 번만 다시 읽는다 — 그 사이
        입금액이 바뀌는 일은 없다시피 하고, 30초마다 훑으면 MT5 가 느려진다.
        """
        now = time.monotonic()
        cached = FxTrader._seed_cache
        if cached is not None and now - cached[1] < SEED_REFRESH_SECONDS:
            return cached[0]
        value = float(cash_flows().get("net_funding") or 0.0)
        FxTrader._seed_cache = (value, now)
        return value

    def _seed_stop(self) -> bool:
        """
        평가 자산이 입금액의 (1 − seed_stop) 아래로 내려갔으면 **앱이 연 포지션을 전부
        닫고 엔진을 멈춘다.** 그렇게 했으면 True.

        최종 손절을 꺼 두면 묶음 하나의 손실에 끝이 없다. 이 한도가 그 끝이다.
        평가 자산(미실현 손익 포함)으로 잰다 — 잔고는 닫기 전까지 그대로라서 열린
        포지션이 계좌를 깎고 있어도 알 수 없다. 사람이 손으로 연 포지션의 손익도
        평가 자산에 들어가지만, 닫는 것은 앱이 연 것뿐이다.

        킬 스위치를 **먼저** 켠다. 닫는 도중에 다른 군이 새로 들어가면 안 된다.
        풀려면 사람이 화면에서 킬을 해제해야 한다.
        """
        limit = self.defense().seed_stop
        if limit <= 0 or self.guard.kill_switch:
            return False
        seed = self._seed()
        if seed <= 0:
            return False
        equity, floor = account().equity, seed * (1.0 - limit)
        if equity > floor:
            return False

        reason = (f"평가 자산 {equity:.2f} 이 입금액 {seed:.2f} 의 "
                  f"-{limit:.0%} 선({floor:.2f}) 아래로 내려갔습니다")
        self.guard.engage_kill(reason)
        try:
            # **닫는 것이 먼저다.** 기록이 실패해도 포지션은 닫혀 있어야 한다.
            self.flatten("계좌 손실 한도")
        finally:
            self.journal.write("engine", event="seed_stop", equity=equity, seed=seed,
                               floor=floor, reason=reason)
            for other in {id(t): t for t in (self, *traders())}.values():
                other.kill(reason)
        return True

    # ─── 한 회 ───────────────────────────────────────────────────

    def run_once(self) -> dict:
        """
        한 바퀴. 스케줄러가 봉 마감마다 부른다.

        순서가 중요하다. **대조를 먼저 한다** — 손절이 이미 체결됐는데 그걸
        모르고 진입 판단을 하면 한도 계산이 틀린다.
        """
        with self._lock:
            self.state.signals, self.state.opened, self.state.closed = [], [], []
            try:
                return self._run_once()
            except MT5Error as exc:
                self.state.last_error = str(exc)
                self.journal.write("error", where="run_once", message=str(exc))
                return self.status()

    def _run_once(self) -> dict:
        acct = account()
        rec = broker.reconcile(self._known)
        if rec["closed_elsewhere"]:
            self.journal.write("reconcile", closed_elsewhere=rec["closed_elsewhere"],
                               note="브로커 쪽에서 닫혔습니다 (손절 체결·수동 청산·재시작)")
        # 두 군이 같은 계좌를 쓴다. **자기 군이 연 것만 자기 것으로 센다.** 남의 군
        # 포지션까지 세면 그쪽 신호로 내 포지션을 닫거나, 내 한도를 남의 것으로 채운다.
        self._known = {k: v for k, v in rec["open"].items()
                       if broker.group_of(v) == self.group.name}

        ep = ExecParams(
            risk_per_trade=float(self.cfg.require("risk.risk_per_trade")),
            atr_stop_mult=float(self.cfg.require("risk.atr_stop_mult")),
            reward_ratio=float(self.cfg.require("risk.reward_ratio")),
            daily_loss_limit=float(self.cfg.require("risk.daily_loss_limit")),
        )
        strats = self.strategies()
        if not strats:
            # 투표할 전략이 없다(단기 군에서 전략을 전부 껐을 때). 들고 있는 것은
            # 브로커의 손절·목표가 지킨다 — 신호가 없으니 여기서 판단할 것이 없다.
            self.state.scanned = 0
            self.state.last_run = datetime.now(timezone.utc).isoformat()
            self.state.last_error = None
            return self.status()
        pairs = self.watchlist()
        self.state.scanned = len(pairs)
        # **감시 목록에서 뺀 종목이라도 들고 있으면 계속 돌본다.** 목록을 줄인 순간
        # 열려 있던 포지션이 버려지면 신호 청산도 역행 대응도 멈춘다. 그런 종목은
        # 닫을 때까지만 훑고, 새로 들어가지는 않는다.
        watched = {p.mt5 for p in pairs}
        leftover = [p for p in (parse(symbol) for symbol in sorted(
            {pos["symbol"] for pos in self._known.values()} - watched)) if p is not None]

        for pair in pairs + leftover:
            try:
                sig = self._signal(pair, strats, ep)
            except (MT5Error, ValueError) as exc:
                self.journal.write("error", symbol=pair.canonical, message=str(exc))
                continue
            if sig is None:
                continue

            self.state.signals.append({**sig.as_dict(),
                                       "tradingview": pair.tradingview,
                                       "mt5": pair.mt5})
            if sig.action in ("롱진입", "숏진입", "청산"):
                self.journal.write("signal", symbol=pair.canonical,
                                   action=sig.action, direction=sig.direction,
                                   score=sig.score, bar_time=sig.bar_time)
                self._follow(pair)

            self._act(pair, sig, acct, ep, allow_entry=pair.mt5 in watched)

        self.state.last_run = datetime.now(timezone.utc).isoformat()
        self.state.last_error = None
        return self.status()

    def _signal(self, pair: Pair, strats, ep: ExecParams) -> SymbolSignal | None:
        """
        **주문을 낼 브로커의 호가로** 신호를 계산한다. 트레이딩뷰 시세로
        계산하면 화면은 맞아 보여도 체결과 어긋난다 (제공자가 다르다).
        """
        df = bars(pair.mt5, self.interval, self.bars_needed())
        if df.empty:
            return None
        self._shock[pair.mt5] = fx_defense.shock(df, self.defense().shock_mult)
        feats = compute_all(df)

        # 이 통화쌍의 실제 비용. 봉마다 그때의 스프레드가 저장돼 있다.
        if "spread" in df.columns:
            si = symbol_info(pair.mt5)
            half = float((df["spread"] * si.point / df["close"]).median()) / 2.0
        else:
            half = 0.0
        costs = Costs(fee_rate=half, slippage_bps=0.5)

        return signal_for(self.cfg, pair.canonical, interval=self.interval,
                          use_derivatives=False, strategies=strats,
                          weight_params=self.weight_params(),
                          exec_params=ep, costs=costs, bars=df, feats=feats)

    def _follow(self, pair: Pair) -> None:
        """신호가 난 종목으로 차트를 옮긴다. 사람이 골라 둔 것은 건드리지 않는다."""
        try:
            from ..server.fx_api import follow
            follow(pair.canonical)
        except Exception:            # 화면이 없어도 매매는 돌아야 한다
            pass

    # ─── 판단 → 주문 ─────────────────────────────────────────────

    def _act(self, pair: Pair, sig: SymbolSignal, acct, ep: ExecParams,
             allow_entry: bool = True) -> None:
        held = [p for p in self._known.values() if p["symbol"] == pair.mt5]
        rules = self.defense()
        # 역행 대응 중이면 한 종목에 여러 포지션이(양방 대응은 반대 방향으로도)
        # 있다. 묶음의 방향은 첫 진입의 방향이다.
        basket = fx_defense.basket_of(
            held, rules, symbol_info(pair.mt5).point if held and rules.enabled else 0.0)
        held_side = basket.side if basket else (held[0]["side"] if held else None)

        # 1) 청산이 먼저다. 자리를 비워야 다음 진입을 볼 수 있다.
        if held and (sig.action == "청산" or
                     (sig.direction != 0 and
                      held_side != ("LONG" if sig.direction > 0 else "SHORT"))):
            for p in held:
                self._close(p, "신호 청산")
            # 닫힌 것만 장부에서 빠진다. **다 닫히지 않았으면 이번 바퀴는 여기까지** —
            # 남은 것을 두고 새로 들어가면 한 종목에 주인이 다른 포지션이 섞인다.
            # 묶음은 포지션이 여럿이라 하나가 안 닫히는 일이 전보다 흔하다.
            if any(p["symbol"] == pair.mt5 for p in self._known.values()):
                return
            held = []

        if not self.state.running:
            return
        if held:
            # 이미 들고 있다. 새로 들어가지 않고, 묶음이 역행했으면 대응한다.
            if basket is not None:
                self._defend(pair, basket, sig, acct, ep)
            return
        if not allow_entry or sig.action not in ("롱진입", "숏진입"):
            return                    # 감시 목록 밖의 종목에는 새로 들어가지 않는다

        side = "LONG" if sig.direction > 0 else "SHORT"
        if sig.stop_price is None or sig.risk_per_unit is None:
            self._blocked(pair, "손절 거리를 계산할 수 없습니다 (ATR 없음)")
            return

        risk_money = acct.balance * self.guard.limits.risk_per_trade
        lots, why = broker.lots_for_risk(pair.mt5, side, sig.close,
                                         sig.stop_price, risk_money)
        if lots <= 0:
            self._blocked(pair, why or "랏이 0 입니다")
            return

        si = symbol_info(pair.mt5)
        notional = si.notional(lots, sig.close)      # 계좌 통화로 — 한도가 그 단위다
        decision = self.guard.check_entry(
            balance=acct.balance, open_positions=self._baskets(),
            notional=notional, symbol=pair.canonical)
        if not decision:
            self._blocked(pair, decision.reason)
            return

        tag_value = broker.tag(pair.mt5, sig.bar_time, self.group.code)
        stop = sig.stop_price
        if rules.enabled:
            # 수량은 손절 거리 1배에 맞춘 그대로 두고, **손절만 멀리 건다** —
            # 추가 진입을 다 한 뒤 한 단계 더 간 자리다. 그 사이는 _defend 가 맡는다.
            # 최종 손절을 꺼 두었으면(final_stop) 손절 없이 들어간다.
            sign = 1 if side == "LONG" else -1
            stop = (sig.close - sign * (rules.max_adds + 1) * sig.risk_per_unit
                    if rules.final_stop else None)
            # 단계 간격을 태그에 적는다. 손절이 없어도, 재시작해도 읽을 수 있게.
            tag_value += fx_defense.base_suffix(sig.risk_per_unit, si.point)
        try:
            fill = broker.open_position(
                pair.mt5, side, lots, stop, sig.target_price,
                tag_value=tag_value, allow_live=self.allow_live)
        except MT5Error as exc:
            self.journal.write("reject", symbol=pair.canonical, side=side,
                               lots=lots, message=str(exc))
            return

        self.guard.record_order()
        self._known[str(fill.ticket)] = {
            "ticket": fill.ticket, "symbol": pair.mt5, "side": side,
            "volume": fill.volume, "entry_price": fill.price,
            "stop_loss": fill.stop_loss, "take_profit": fill.take_profit,
            "comment": tag_value, "magic": broker.MAGIC,
        }
        row = {**fill.as_dict(), "canonical": pair.canonical,
               "tradingview": pair.tradingview, "score": sig.score,
               "regime": sig.regime,
               # 1R 의 가격(손절 거리 1배). 거래 내역이 R 을 이것으로 계산한다 —
               # 손절 가격으로 계산하면 손절을 멀리 걸거나 안 건 거래의 R 이 틀린다.
               "risk_per_unit": sig.risk_per_unit,
               # 이 진입을 만든 전략들. 진입하는 순간에 남겨야 한다 — 가중치는
               # 매 바퀴 다시 계산되므로 나중에는 그때의 투표를 되살릴 수 없다.
               "strategies": entry_votes(sig)}
        self.state.opened.append(row)
        self.journal.write("order", **row)

    # ─── 역행 대응 ───────────────────────────────────────────────

    def _baskets(self) -> int:
        """
        들고 있는 **종목 수.** 동시 보유 한도는 종목을 세는 것이다 — 역행 대응으로
        한 종목에 포지션이 여럿 쌓여도 한 자리만 쓴다.
        """
        return len({p["symbol"] for p in self._known.values()})

    def _defend(self, pair: Pair, basket: fx_defense.Basket, sig: SymbolSignal,
                acct, ep: ExecParams) -> None:
        """
        묶음이 다음 단계까지 역행했으면 대응한다 (src/fx/defense.py).

        양방 대응이 열려 있으면 손실이 묶여 있으므로 더 하지 않는다. 추가 진입을
        다 썼으면 브로커에 걸린 최종 손절이 맡는다.
        """
        if basket.hedge is not None or basket.level >= basket.rules.max_adds:
            return
        if not basket.reached(sig.close):
            return
        # **한 봉에 한 번만.** 봉이 두 단계를 건너뛰었을 때 화면의 "한 바퀴"를 같은
        # 봉에서 또 누르면, 같은 마감 가격으로 다음 단계까지 닿았다고 읽어 4배를
        # 더 싣는다. 다음 단계는 다음 봉이 마감한 뒤에 본다.
        if self._defended.get(pair.mt5) == str(sig.bar_time):
            return
        si = symbol_info(pair.mt5)
        if self._shock.get(pair.mt5, 0) == -basket.sign:
            # 급변동으로 밀렸다. 물량을 늘리지 않고 반대로 같은 수량을 열어 묶는다.
            # 한 단계 되돌리면 이 포지션이 손절되고 묶음만 남는다. 최종 손절까지
            # 가면 묶음의 손실을 이 포지션의 이익이 상쇄한다.
            lots = fx_defense.round_step(min(basket.volume, si.volume_max), si.volume_step)
            self._defense_order(
                pair, basket, sig, acct, si, kind="hedge", level=basket.level,
                side="SHORT" if basket.side == "LONG" else "LONG", lots=lots,
                stop=sig.close + basket.sign * basket.step, target=basket.final_stop,
                tag_value=basket.tag(fx_defense.HEDGE))
            return

        level = basket.level + 1
        lots = min(basket.next_volume(si.volume_step), si.volume_max)
        fill = self._defense_order(
            pair, basket, sig, acct, si, kind="martingale", level=level,
            side=basket.side, lots=lots, stop=basket.final_stop, target=None,
            tag_value=basket.tag(f"m{level}"))
        if fill is not None:
            self._retarget(basket, fill, ep)

    def _defense_order(self, pair: Pair, basket: fx_defense.Basket, sig: SymbolSignal,
                       acct, si, *, kind: str, level: int, side: str, lots: float,
                       stop: float, target: float | None, tag_value: str):
        """대응 주문 하나. 가드를 지나고, 장부에 무엇에 대한 대응인지 남긴다."""
        if lots < si.volume_min:
            self._blocked(pair, f"역행 대응 랏 {lots} 이 최소 {si.volume_min} 보다 작습니다")
            return None
        # 이 종목은 이미 한 자리를 쓰고 있다 — 자기 자신은 한도에서 뺀다.
        decision = self.guard.check_entry(
            balance=acct.balance, open_positions=self._baskets() - 1,
            notional=si.notional(lots, sig.close), symbol=pair.canonical)
        if not decision:
            self._blocked(pair, f"역행 대응 보류 — {decision.reason}")
            return None
        try:
            fill = broker.open_position(pair.mt5, side, lots, stop, target,
                                        tag_value=tag_value, allow_live=self.allow_live)
        except MT5Error as exc:
            self.journal.write("reject", symbol=pair.canonical, side=side, lots=lots,
                               defense=kind, message=str(exc))
            return None

        self.guard.record_order()
        self._defended[pair.mt5] = str(sig.bar_time)
        self._known[str(fill.ticket)] = {
            "ticket": fill.ticket, "symbol": pair.mt5, "side": side,
            "volume": fill.volume, "entry_price": fill.price,
            "stop_loss": fill.stop_loss, "take_profit": fill.take_profit,
            "comment": tag_value, "magic": broker.MAGIC,
        }
        row = {**fill.as_dict(), "canonical": pair.canonical,
               "tradingview": pair.tradingview, "score": sig.score, "regime": sig.regime,
               "risk_per_unit": basket.step,
               "strategies": [], "defense": kind, "level": level,
               "basket": str(basket.base.get("comment", ""))}
        self.state.opened.append(row)
        self.journal.write("order", **row)
        return fill

    def _retarget(self, basket: fx_defense.Basket, fill, ep: ExecParams) -> None:
        """
        추가 진입 뒤 묶음 전체의 목표가를 옮긴다. 평균 단가가 내려왔으므로 처음
        노렸던 수익 금액은 훨씬 가까운 가격에서 난다. 손절은 그대로 둔다.
        """
        target = basket.target(ep.reward_ratio, (fill.volume, fill.price))
        for ticket in [int(p["ticket"]) for p in basket.legs] + [int(fill.ticket)]:
            try:
                broker.move_stop(ticket, basket.final_stop, target)
            except MT5Error as exc:
                # 목표가를 못 옮겨도 손절은 걸려 있다. 다음 추가 때 다시 시도된다.
                self.journal.write("error", where="retarget", ticket=ticket,
                                   message=str(exc))
                continue
            if str(ticket) in self._known:
                self._known[str(ticket)]["take_profit"] = target

    def _close(self, pos: dict, reason: str) -> None:
        try:
            out = broker.close_position(int(pos["ticket"]),
                                        allow_live=self.allow_live,
                                        reason="rqexit")
        except MT5Error as exc:
            self.journal.write("error", where="close", ticket=pos.get("ticket"),
                               message=str(exc))
            return
        self._known.pop(str(pos["ticket"]), None)
        if out.get("closed"):
            self.guard.record_realised(float(out.get("profit") or 0.0))
            self.state.closed.append(out)
            self._journal_close(reason, out)

    def _blocked(self, pair: Pair, reason: str) -> None:
        self.journal.write("blocked", symbol=pair.canonical, reason=reason)

    # ─── 상태 ────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        try:
            acct = account()
            balance, acct_d = acct.balance, acct.as_dict()
        except MT5Error as exc:
            balance, acct_d = 0.0, {"error": str(exc)}
        rules = self.defense()
        try:
            seed = self._seed() if rules.seed_stop > 0 else 0.0
        except MT5Error:
            seed = 0.0
        return {
            # 계좌 손실 한도. floor 아래로 평가 자산이 내려가면 전부 닫고 멈춘다.
            # 입금액을 모르면(seed 0) 한도가 듣지 않는다 — 화면이 그것을 알려야 한다.
            "seed_stop": {"limit": rules.seed_stop, "seed": seed,
                          "floor": round(seed * (1.0 - rules.seed_stop), 2) if seed > 0 else None},
            "group": self.group.name,
            "group_label": self.group.label,
            "running": self.state.running,
            "mode": self.state.mode,
            # 단기 군은 켜 둔 단기 전략이 없으면 None — 돌 것이 없다는 뜻이다
            "interval": self.interval,
            "allow_live": self.allow_live,
            "verdict": VERDICT,
            # 역행 대응(마팅게일·양방 대응) 설정. 화면이 켜져 있는지 보여 준다.
            "defense": self.defense().as_dict(),
            "account": acct_d,
            "guard": self.guard.status(balance),
            "last_run": self.state.last_run,
            "last_error": self.state.last_error,
            "scanned": self.state.scanned,
            # **루프가 실제로 도는지 보여 준다.** running=True 인데 cycles 가
            # 0 에서 안 늘면 스케줄러가 죽은 것이다 — 예전에는 스케줄러가
            # 아예 없어서 "시작"을 눌러도 아무 일이 없었다.
            "loop_alive": bool(self._thread and self._thread.is_alive()),
            # 루프가 도는 동안 PC 의 유휴 절전을 막고 있는가. Windows 가 아니거나
            # 요청이 거부되면 False — 그때는 PC 가 잠들면 엔진도 선다.
            "keep_awake": self._awake,
            "cycles": self.state.cycles,
            "checks": self.state.checks,
            "last_bar": self.state.last_bar,
            "next_check": self.state.next_check,
            "poll_seconds": POLL_SECONDS,
            "signals": self.state.signals,
            "opened": self.state.opened,
            "closed": self.state.closed,
            "held": list(self._known.values()),
        }


_traders: dict[str, FxTrader] = {}
_trader_lock = threading.Lock()


def trader(group: str = "main") -> FxTrader:
    """
    **군마다 엔진 하나.** 같은 군에 둘이 돌면 같은 신호로 두 번 주문한다.

    단기 군은 기본 군의 **가드와 장부를 함께 쓴다.** 킬 스위치·일일 손실 한도·
    분당 주문 한도는 계좌의 것이고, 장부가 둘이면 같은 파일에 따로 써서 줄이 섞인다.
    """
    if group not in GROUPS:
        raise KeyError(f"모르는 투표 군: {group} (가능: {', '.join(GROUPS)})")
    with _trader_lock:
        if "main" not in _traders:
            _traders["main"] = FxTrader()
        if group not in _traders:
            main = _traders["main"]
            _traders[group] = FxTrader(cfg=main.cfg, journal=main.journal,
                                       guard=main.guard, group=GROUPS[group])
        return _traders[group]


def traders() -> list[FxTrader]:
    """지금까지 만들어진 엔진 전부 — 킬 스위치처럼 군을 가리지 않는 동작에 쓴다."""
    with _trader_lock:
        return list(_traders.values())
