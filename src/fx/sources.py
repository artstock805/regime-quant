"""
거래의 **출처**를 가른다 — 어느 앱이 낸 주문인가.

**왜 이게 필요한가.** 이 계좌에는 여러 앱이 동시에 주문을 낸다. 화면이
`엔진 / 수동` 두 가지로만 가르면 나머지가 전부 "수동"으로 묶이는데, 그
대부분은 사람이 손으로 낸 게 아니라 **다른 봇**이다.

실제로 그 라벨을 글자 그대로 읽고 "손으로 연 거래 76건에 손절이 하나도
없다 — 처분효과다"라는 진단을 냈다가 틀렸다. 그 76건은 네 개 앱이 낸
것이었고, 가장 큰 손실(GOLD −215)을 낸 앱은 **열린 포지션에 손절을 걸어
두고 있었다.** 화면이 두 칸으로만 갈라서 생긴 오독이다.

그래서 출처를 **번호로 확인해서** 가른다. 추측하지 않는다.

**어떻게 가르나.** MT5 는 주문에 `magic`(앱이 스스로 찍는 번호)과
`comment`(태그 문자열)를 싣는다. magic 이 먼저다 — 코멘트는 브로커가
`[sl 1.2345]` 같은 걸로 덮어쓰는 경우가 있어서 사라질 수 있다.

**모르는 번호를 '사람이 직접'으로 떨어뜨리지 않는다.** 그게 이 파일이 고치려는
바로 그 실수다. 등록 안 된 magic 은 `미등록(magic=N)`으로 그대로 보여 줘서,
새 앱이 붙으면 화면에 드러나고 여기 등록하게 만든다. magic 이 **0일 때만**
손클릭이다 — MT5 창에서 사람이 직접 낸 주문에는 번호가 없다.

**이름은 지어내지 않는다.** 등록부가 이미 있다 —
`trading-tools/mt5-who.py` 의 `APPS`. 아래는 거기서 가져온
이름이다. 코멘트만 보고 이름을 붙이면 틀린다: 래리 앱의 `GOLD|gsv|L|...`
에서 `gsv` 는 **전략 이름**이지 앱 이름이 아닌데, 그걸 앱 이름으로 읽고
한 번 틀리게 보고했다.

실측 근거 (2026-11-14, 데모 계좌 한 개, 최근 120일 체결 158건):

    magic=20260819   99건  'GOLD|gsv|L|...'      래리 윌리엄스
    magic=26081900   32건  'PT|EURJPY~USDJPY'    페어트레이딩
    magic=0          14건  (코멘트 없음)          사람이 직접
    magic=940715      7건  'forex-alert 4gates'  4관문 수동주문
    magic=770315      4건  'rqGBPUSD2609160100'  레짐 퀀트 (이 프로젝트)
    magic=770316      2건  'aw|m|1'              전천후 트레이더

앱을 새로 붙이면 **등록부와 여기 둘 다** 한 줄 추가한다. 깜빡해도 조용히
섞이지는 않는다 — '미등록'으로 뜬다.
"""

from __future__ import annotations

from dataclasses import dataclass

# 이 프로젝트의 엔진이 주문에 찍는 번호와 태그.
#
# **여기가 유일한 정의 자리다.** 주문을 내는 쪽(broker.py)과 내역을 읽는
# 쪽(mt5.py)이 각자 적어 두면, 한쪽만 바뀌었을 때 과거 거래가 조용히 남의
# 것이 된다. 두 모듈 모두 여기서 가져간다.
MAGIC = 770315
PREFIX = "rq"

# magic 을 안 쓰는 수동 주문. MT5 는 0 을 넣는다.
NO_MAGIC = 0


@dataclass(frozen=True)
class Source:
    """
    거래를 낸 주체 하나.

    `magics` 가 1차 단서, `prefixes` 는 magic 이 비었을 때의 2차 단서다.
    """

    key: str
    label: str
    magics: tuple[int, ...] = ()
    prefixes: tuple[str, ...] = ()

    def has_magic(self, magic: int) -> bool:
        return bool(magic) and magic in self.magics

    def has_prefix(self, comment: str) -> bool:
        text = (comment or "").strip()
        return bool(text and self.prefixes) and text.startswith(self.prefixes)


# 등록된 앱들. **엔진이 맨 앞이다** — 한 포지션에 여러 단서가 섞여 있으면
# 엔진 쪽이 이긴다(아래 of_deals 참고).
REGISTRY: tuple[Source, ...] = (
    Source("engine", "레짐 퀀트", (MAGIC,), (PREFIX,)),
    # 래리 앱의 코멘트는 'GOLD|gsv|L|...' 다. gsv·nbo·vbo 는 **전략 이름**이지
    # 앱 이름이 아니다 — 코멘트만 보고 이름을 지으면 틀린다(실제로 틀렸다).
    # 접두사는 등록하지 않는다: 종목 이름이 앞에 와서 접두사로 쓰려면 'US'
    # 같은 토막을 넣어야 하는데, 그러면 남의 지수 거래까지 빨아들인다.
    # **느슨한 규칙은 틀린 귀속을 만들고, 그게 이 파일이 고치려는 문제다.**
    Source("larry", "래리 윌리엄스", (20260819,)),
    Source("pair", "페어트레이딩", (26081900,), ("PT|",)),
    Source("allweather", "전천후 트레이더", (770316,), ("aw|",)),
    # 텔레그램 버튼으로 **사람이 눌러서** 나가는 주문이다. 봇이 아니다.
    Source("gates", "4관문 수동주문", (940715,), ("forex-alert",)),
)

HAND = Source("hand", "사람이 직접", (NO_MAGIC,))


def unregistered(magic: int) -> Source:
    """
    모르는 번호. **'수동'으로 숨기지 않고 번호를 그대로 들고 올라간다.**

    이걸 화면에서 보면 REGISTRY 에 한 줄 추가하라는 뜻이다.
    """
    return Source(f"magic{magic}", f"미등록(magic={magic})", (magic,))


def classify(magic: int, comment: str = "") -> Source:
    """
    체결 하나의 출처.

    **번호를 전부 본 다음에야 코멘트를 본다.** 한 앱씩 '번호 아니면 코멘트'로
    훑으면 REGISTRY 순서가 우선순위가 돼 버린다 — 앞에 선 앱의 접두사가
    뒤에 선 앱의 **정확한 번호**를 이긴다. 코멘트는 브로커가 `[sl 1.2345]`
    로 덮어쓰기도 하는 약한 단서이므로, 순서가 뒤집히면 귀속이 틀린다.
    """
    for s in REGISTRY:
        if s.has_magic(magic):
            return s
    for s in REGISTRY:
        if s.has_prefix(comment):
            return s
    return HAND if not magic else unregistered(magic)


def _pick(deals: list[dict]) -> Source:
    """
    체결 여러 줄 중 하나를 고른다.

    **엔진이 하나라도 걸리면 엔진이다.** 진입 체결의 코멘트가 `rq...` 인데
    청산 체결은 브로커가 `[sl 1.2345]` 로 덮어쓰는 경우가 실제로 있다.
    엔진이 아니면 **손클릭이 아닌 첫 출처**를 쓴다 — magic 이 빈 체결이
    섞여 있다고 해서 남의 앱 거래가 손클릭이 되면 안 된다.
    """
    found = [classify(d.get("magic", 0), d.get("comment", "")) for d in deals]
    for s in found:
        if s.key == "engine":
            return s
    for s in found:
        if s.key != HAND.key:
            return s
    return HAND


def of_deals(deals: list[dict], entries: list[dict] | None = None) -> Source:
    """
    왕복 한 건의 출처 — **연 쪽이 주인이다.**

    실측(2026-11-14): 한 포지션 안에서 magic 이 **갈리는** 경우가 80건 중
    2건 있었다. 그록데스크가 연 US100Cash 를 추세알림이 닫았고, 페어
    트레이딩이 연 EURJPY 도 추세알림이 닫았다. 앱들이 서로의 포지션을
    건드리고 있다.

    이때 손익은 **연 앱의 판단**에 달렸으므로 진입 체결이 주인을 정한다.
    체결 순서대로 아무거나 고르면 같은 계좌를 두 번 재면서 다른 답이 나온다.

    다만 진입 체결에 magic 도 태그도 없는 경우가 있다(실측 7건: 진입은
    비어 있고 청산만 `forex-alert close`). 그때는 모른다고 할 게 아니라
    왕복 전체를 본다 — 안 그러면 남의 봇 거래가 '사람이 직접'으로 둔갑한다.
    """
    if entries:
        s = _pick(entries)
        if s.key != HAND.key:
            return s
    return _pick(deals)
