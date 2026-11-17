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
from . import broker
from . import watchlist as fx_watchlist
from .mt5 import MT5Error, bars, account, positions, symbol_info
from .symbols import Pair

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


class FxTrader:
    """
    한 번에 한 스레드만 돈다. MT5 는 스레드 안전하지 않고, 루프가 겹치면
    같은 신호로 두 번 주문할 수 있다.
    """

    def __init__(self, cfg: Config | None = None,
                 journal: Journal | None = None,
                 guard: RiskGuard | None = None) -> None:
        self.cfg = cfg or load_config()
        self.journal = journal or Journal()
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ─── 설정 ────────────────────────────────────────────────────

    @property
    def interval(self) -> str:
        return str(self.cfg.get("fx.interval", "1h"))

    @property
    def allow_live(self) -> bool:
        return bool(self.cfg.get("fx.allow_live", False))

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
            self.journal.write("close", reason=reason, **row)
        return out

    # ─── 루프 ────────────────────────────────────────────────────

    def _spawn(self) -> None:
        """루프 스레드를 띄운다. 이미 살아 있으면 두 번 띄우지 않는다."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="fx-engine",
                                        daemon=True)
        self._thread.start()

    def _newest_bar(self) -> str | None:
        """
        기준 종목의 **마지막 마감봉** 시각. 못 읽으면 None.

        `bars()` 가 진행 중인 봉을 버리고 주므로, 이 값이 바뀌면 새 봉이
        마감된 것이다. 별도의 settle 대기가 필요 없다.
        """
        pairs = self.watchlist()
        if not pairs:
            return None
        df = bars(pairs[0].mt5, self.interval, 3)
        if df.empty:
            return None
        return df.index[-1].isoformat()

    def _loop(self) -> None:
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
        self._known = dict(rec["open"])

        ep = ExecParams(
            risk_per_trade=float(self.cfg.require("risk.risk_per_trade")),
            atr_stop_mult=float(self.cfg.require("risk.atr_stop_mult")),
            reward_ratio=float(self.cfg.require("risk.reward_ratio")),
            daily_loss_limit=float(self.cfg.require("risk.daily_loss_limit")),
        )
        strats = self.strategies()
        pairs = self.watchlist()
        self.state.scanned = len(pairs)

        for pair in pairs:
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

            self._act(pair, sig, acct, ep)

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

    def _act(self, pair: Pair, sig: SymbolSignal, acct, ep: ExecParams) -> None:
        held = [p for p in self._known.values() if p["symbol"] == pair.mt5]

        # 1) 청산이 먼저다. 자리를 비워야 다음 진입을 볼 수 있다.
        if held and (sig.action == "청산" or
                     (sig.direction != 0 and
                      held[0]["side"] != ("LONG" if sig.direction > 0 else "SHORT"))):
            for p in held:
                self._close(p, "신호 청산")
            held = []

        if not self.state.running or sig.action not in ("롱진입", "숏진입"):
            return
        if held:
            return                    # 이미 들고 있다

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
        notional = lots * si.contract_size * sig.close
        decision = self.guard.check_entry(
            balance=acct.balance, open_positions=len(self._known),
            notional=notional, symbol=pair.canonical)
        if not decision:
            self._blocked(pair, decision.reason)
            return

        tag_value = broker.tag(pair.mt5, sig.bar_time)
        try:
            fill = broker.open_position(
                pair.mt5, side, lots, sig.stop_price, sig.target_price,
                tag_value=tag_value, allow_live=self.allow_live)
        except MT5Error as exc:
            self.journal.write("reject", symbol=pair.canonical, side=side,
                               lots=lots, message=str(exc))
            return

        self.guard.record_order()
        self._known[str(fill.ticket)] = {
            "ticket": fill.ticket, "symbol": pair.mt5, "side": side,
            "volume": fill.volume, "entry_price": fill.price,
            "stop_loss": fill.stop_loss, "comment": tag_value,
            "magic": broker.MAGIC,
        }
        row = {**fill.as_dict(), "canonical": pair.canonical,
               "tradingview": pair.tradingview, "score": sig.score,
               "regime": sig.regime}
        self.state.opened.append(row)
        self.journal.write("order", **row)

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
            self.journal.write("close", reason=reason, **out)

    def _blocked(self, pair: Pair, reason: str) -> None:
        self.journal.write("blocked", symbol=pair.canonical, reason=reason)

    # ─── 상태 ────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        try:
            acct = account()
            balance, acct_d = acct.balance, acct.as_dict()
        except MT5Error as exc:
            balance, acct_d = 0.0, {"error": str(exc)}
        return {
            "running": self.state.running,
            "mode": self.state.mode,
            "interval": self.interval,
            "allow_live": self.allow_live,
            "verdict": VERDICT,
            "account": acct_d,
            "guard": self.guard.status(balance),
            "last_run": self.state.last_run,
            "last_error": self.state.last_error,
            "scanned": self.state.scanned,
            # **루프가 실제로 도는지 보여 준다.** running=True 인데 cycles 가
            # 0 에서 안 늘면 스케줄러가 죽은 것이다 — 예전에는 스케줄러가
            # 아예 없어서 "시작"을 눌러도 아무 일이 없었다.
            "loop_alive": bool(self._thread and self._thread.is_alive()),
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


_trader: FxTrader | None = None
_trader_lock = threading.Lock()


def trader() -> FxTrader:
    """앱 전체가 하나의 엔진을 쓴다. 두 개가 돌면 같은 신호로 두 번 주문한다."""
    global _trader
    with _trader_lock:
        if _trader is None:
            _trader = FxTrader()
        return _trader
