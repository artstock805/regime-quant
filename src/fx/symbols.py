"""
종목 이름 맞추기 — MT5 와 트레이딩뷰는 같은 통화쌍을 다르게 부른다.

    MT5          EURUSD, EURUSD., EURUSDm, EURUSD.raw   (브로커마다 접미사가 다르다)
    트레이딩뷰    FX_IDC:EURUSD, OANDA:EURUSD, FX:EURUSD (데이터 제공자마다 다르다)

**두 시세는 같지 않다.** 트레이딩뷰의 FX_IDC 는 은행 간 집계 호가고, MT5 는
우리가 실제로 체결하는 브로커 호가다. 스프레드도 다르고 값도 몇 핍 어긋난다.

그래서 이 앱은 역할을 나눈다.
    · **차트(보는 것)** — 트레이딩뷰. 익숙한 인터페이스와 지표를 그대로 쓴다.
    · **신호·주문(하는 것)** — MT5. 우리가 실제로 체결하는 호가로 계산한다.

화면에 그 차이를 숨기지 않는다. 차트의 진입 마커와 실제 체결가가 몇 핍 다른 것은
버그가 아니라 서로 다른 피드이기 때문이고, 그 사실을 모르면 버그로 오해한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 트레이딩뷰 FX 데이터 제공자. FX_IDC 가 가장 널리 무료로 열려 있다.
PROVIDERS = {
    "FX_IDC": "은행 간 집계 (무료·가장 널리 열림)",
    "OANDA": "OANDA 호가",
    "FX": "트레이딩뷰 통합",
    "FOREXCOM": "FOREX.com 호가",
}
DEFAULT_PROVIDER = "FX_IDC"

# 주요 통화. 이 조합만 FX 로 인정한다 — 브로커 심볼 목록에는 CFD·지수도 섞여 있다.
CURRENCIES = {"EUR", "USD", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"}

# 유동성 상위 조합. 스프레드가 좁아 비용/1R 이 낮다.
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
CROSSES = ["EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD", "CADJPY", "CHFJPY"]

# 브로커가 붙이는 접미사. 예: EURUSDm(마이크로), EURUSD.(ECN), EURUSD_raw
# 앞 6글자가 통화쌍이고 뒤는 짧은 꼬리여야 한다. 꼬리를 무제한 허용하면
# EURUSDJPY 같은 이름이 EURUSD 로 읽혀 **엉뚱한 종목에 주문이 나간다.**
_SYMBOL = re.compile(r"^([A-Z]{6})([.\-_]?[A-Z0-9]{0,6})$")


@dataclass(frozen=True)
class Pair:
    mt5: str                 # 주문을 낼 이름 (브로커 것)
    base: str                # EUR
    quote: str               # USD
    provider: str = DEFAULT_PROVIDER

    @property
    def canonical(self) -> str:
        """접미사를 뗀 표준 이름. EURUSDm → EURUSD"""
        return f"{self.base}{self.quote}"

    @property
    def tradingview(self) -> str:
        """트레이딩뷰 위젯에 넣을 심볼."""
        return f"{self.provider}:{self.canonical}"

    @property
    def display(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def is_jpy(self) -> bool:
        """엔화 페어는 호가 자릿수가 다르다 (3자리 vs 5자리)."""
        return self.quote == "JPY"

    def as_dict(self) -> dict:
        return {"mt5": self.mt5, "canonical": self.canonical,
                "tradingview": self.tradingview, "display": self.display,
                "base": self.base, "quote": self.quote}


def parse(mt5_name: str, provider: str = DEFAULT_PROVIDER) -> Pair | None:
    """
    브로커 심볼을 통화쌍으로 읽는다. FX 가 아니면 None.

    접미사를 떼는 이유: 브로커마다 EURUSDm, EURUSD., EURUSD_raw 처럼 다르게
    부르는데 트레이딩뷰는 EURUSD 하나뿐이다. 이걸 안 맞추면 차트가 안 뜬다.
    """
    m = _SYMBOL.match(mt5_name.strip().upper())
    if m is None:
        return None
    core, tail = m.group(1), m.group(2)
    # 꼬리가 통화 코드면 이건 접미사가 아니라 다른 이름이다 (EURUSDJPY 등)
    if tail.lstrip("._-") in CURRENCIES:
        return None
    base, quote = core[:3], core[3:]
    if base not in CURRENCIES or quote not in CURRENCIES or base == quote:
        return None
    return Pair(mt5=mt5_name, base=base, quote=quote, provider=provider)


def resolve(wanted: str, available: list[str],
            provider: str = DEFAULT_PROVIDER) -> Pair | None:
    """
    표준 이름(EURUSD)으로 **이 브로커의 실제 심볼**을 찾는다.

    접미사가 붙은 이름을 못 찾으면 주문이 나가지 않는다. 정확히 일치하는 것을
    먼저 보고, 없으면 접미사가 붙은 것을 찾는다.
    """
    target = wanted.upper()
    if target in available:
        return parse(target, provider)
    for name in available:
        p = parse(name, provider)
        if p and p.canonical == target:
            return p
    return None


def catalogue(available: list[str], provider: str = DEFAULT_PROVIDER,
              majors_only: bool = False) -> list[Pair]:
    """
    이 브로커에서 쓸 수 있는 FX 쌍 목록. 중복(접미사 변형)은 하나만 남긴다.
    주요 통화쌍을 앞에 둔다 — 스프레드가 좁아 비용/1R 이 낮기 때문이다.
    """
    seen: dict[str, Pair] = {}
    for name in available:
        p = parse(name, provider)
        if p is None or p.canonical in seen:
            continue
        seen[p.canonical] = p

    order = {c: i for i, c in enumerate(MAJORS + CROSSES)}
    pairs = list(seen.values())
    if majors_only:
        pairs = [p for p in pairs if p.canonical in set(MAJORS + CROSSES)]
    return sorted(pairs, key=lambda p: (order.get(p.canonical, 999), p.canonical))


def pip_size(pair: Pair, digits: int) -> float:
    """
    1핍의 가격 단위. 엔화는 0.01, 나머지는 0.0001.
    5자리/3자리 호가에서 1포인트는 그 1/10 이다.
    """
    return 0.01 if pair.is_jpy else 0.0001


def to_pips(pair: Pair, price_delta: float, digits: int) -> float:
    """가격 차이를 핍으로. 화면과 로그가 사람 단위로 읽히게 한다."""
    return price_delta / pip_size(pair, digits)
