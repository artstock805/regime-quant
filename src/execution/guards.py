"""
리스크 가드 — 주문이 나가기 전에 통과해야 하는 관문.

자동매매에서 계좌를 날리는 건 대개 전략이 아니라 **제동장치 없음**이다.
버그 하나로 같은 주문이 100번 나가거나, 손실이 나는 날 계속 진입하거나,
멈추고 싶은데 멈출 방법이 없거나.

여기 있는 것은 전부 **거부 사유를 문장으로 돌려준다.** 조용히 막으면
사용자는 "왜 진입 안 하지"를 알 수 없고, 그러면 가드를 꺼 버린다.

기본값은 보수적으로 둔다. 검증 1/5 인 전략을 돌리는 중이라는 사실을
기본값이 기억하고 있어야 한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone


@dataclass
class RiskLimits:
    # 트레이드 1회당 계좌의 몇 %를 위험에 두는가 (1R)
    risk_per_trade: float = 0.01
    # 동시에 몇 종목까지
    max_positions: int = 3
    # 하루 실현손실이 계좌의 이 비율을 넘으면 그날은 진입 중단
    daily_loss_limit: float = 0.03
    # 한 종목의 명목가 상한 (계좌 대비). 레버리지 사고를 막는다
    max_notional_ratio: float = 1.0
    # 분당 주문 상한. 루프가 폭주해도 여기서 걸린다
    max_orders_per_minute: int = 6
    # 계좌 잔고 하한 (USDT). 이 아래로 내려가면 진입하지 않는다
    min_balance: float = 20.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOW = Decision(True)


class RiskGuard:
    """
    상태를 들고 있는 관문. 하루 실현손익과 최근 주문 시각을 기억한다.

    **킬 스위치가 최우선이다.** 켜져 있으면 다른 조건을 보지 않고 전부 막는다.
    사람이 "멈춰"라고 했을 때 조건부로 멈추면 안 된다.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch: bool = False
        self.kill_reason: str = ""
        self._day: date | None = None
        self.realised_today: float = 0.0
        self.orders: list[float] = []          # 최근 주문 시각(epoch)
        self.blocks: list[dict] = []           # 무엇을 왜 막았는지 (화면용)

    # ─── 킬 스위치 ───────────────────────────────────────────────

    def engage_kill(self, reason: str) -> None:
        self.kill_switch = True
        self.kill_reason = reason

    def release_kill(self) -> None:
        self.kill_switch = False
        self.kill_reason = ""

    # ─── 일일 손익 ───────────────────────────────────────────────

    def _roll_day(self, now: datetime) -> None:
        today = now.astimezone(timezone.utc).date()
        if self._day != today:
            self._day = today
            self.realised_today = 0.0

    def record_realised(self, pnl: float, now: datetime | None = None) -> None:
        self._roll_day(now or datetime.now(timezone.utc))
        self.realised_today += pnl

    def record_order(self, now: datetime | None = None) -> None:
        t = (now or datetime.now(timezone.utc)).timestamp()
        self.orders = [x for x in self.orders if t - x < 60] + [t]

    # ─── 판정 ────────────────────────────────────────────────────

    def check_entry(self, *, balance: float, open_positions: int,
                    notional: float, symbol: str,
                    now: datetime | None = None) -> Decision:
        now = now or datetime.now(timezone.utc)
        self._roll_day(now)
        d = self._check(balance=balance, open_positions=open_positions,
                        notional=notional, now=now)
        if not d.allowed:
            self.blocks.append({"at": now.isoformat(), "symbol": symbol,
                                "reason": d.reason})
            del self.blocks[:-50]
        return d

    def _check(self, *, balance: float, open_positions: int,
               notional: float, now: datetime) -> Decision:
        # 1) 킬 스위치가 최우선. 사람이 멈추라고 한 것은 조건부가 아니다.
        if self.kill_switch:
            return Decision(False, f"킬 스위치가 켜져 있습니다 — {self.kill_reason}")

        # 2) 잔고
        if balance < self.limits.min_balance:
            return Decision(False,
                            f"잔고 부족 ({balance:.2f} < {self.limits.min_balance} USDT)")

        # 3) 일일 손실 한도. 나는 날 계속 들어가는 것을 막는다.
        limit = -abs(self.limits.daily_loss_limit) * balance
        if self.realised_today <= limit:
            return Decision(False,
                            f"일일 손실 한도 도달 (오늘 {self.realised_today:+.2f} USDT, "
                            f"한도 {limit:.2f})")

        # 4) 동시 보유
        if open_positions >= self.limits.max_positions:
            return Decision(False,
                            f"동시 보유 한도 ({open_positions}/{self.limits.max_positions})")

        # 5) 명목가 상한. 수량 계산 버그가 레버리지 사고로 번지는 것을 막는다.
        cap = self.limits.max_notional_ratio * balance
        if notional > cap:
            return Decision(False,
                            f"명목가 상한 초과 ({notional:.2f} > {cap:.2f} USDT)")

        # 6) 주문 속도. 루프가 폭주해도 여기서 걸린다.
        recent = [x for x in self.orders if now.timestamp() - x < 60]
        if len(recent) >= self.limits.max_orders_per_minute:
            return Decision(False,
                            f"분당 주문 한도 ({len(recent)}/"
                            f"{self.limits.max_orders_per_minute})")

        return ALLOW

    # ─── 수량 ────────────────────────────────────────────────────

    def position_size(self, *, balance: float, entry: float,
                      stop: float) -> tuple[float, str]:
        """
        1R = 계좌 x risk_per_trade 가 되도록 수량을 정한다.

        손절폭으로 나누는 것이 핵심이다. 고정 수량으로 하면 변동성이 큰 종목에서
        같은 1R 이 몇 배가 된다 — 백테스트의 R 배수와 실전이 어긋난다.
        """
        risk_amount = balance * self.limits.risk_per_trade
        per_unit = abs(entry - stop)
        if per_unit <= 0:
            return 0.0, "손절 거리가 0 입니다"
        qty = risk_amount / per_unit
        if qty <= 0:
            return 0.0, "계산된 수량이 0 입니다"
        return qty, ""

    # ─── 상태 ────────────────────────────────────────────────────

    def status(self, balance: float = 0.0) -> dict:
        limit = -abs(self.limits.daily_loss_limit) * balance if balance else None
        return {
            "kill_switch": self.kill_switch,
            "kill_reason": self.kill_reason,
            "limits": self.limits.as_dict(),
            "realised_today": round(self.realised_today, 4),
            "daily_loss_limit_usdt": None if limit is None else round(limit, 2),
            "orders_last_minute": len([
                x for x in self.orders
                if datetime.now(timezone.utc).timestamp() - x < 60]),
            "recent_blocks": self.blocks[-10:][::-1],
        }
