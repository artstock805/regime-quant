"""
바이낸스 USDT-M 선물 클라이언트 — 서명이 필요한 호출을 담당한다.

**기본은 테스트넷이다.** 실전으로 바꾸려면 설정에서 명시적으로 켜야 하고,
그때 경고가 뜬다. 기본값이 실전이면 실수 한 번이 돈이 된다.

키를 다루는 방식
    키는 이 저장소에 절대 들어가지 않는다. 두 곳에서만 읽는다.
        · 환경변수  BINANCE_API_KEY / BINANCE_API_SECRET
        · config/secrets.yaml  (git 에서 제외됨)
    키가 없으면 조회조차 하지 않고 바로 알려 준다 — 주문이 조용히 실패하는 것보다
    시작하지 않는 편이 낫다.

주문에서 반드시 지키는 것 세 가지
    1) **멱등성**: 모든 주문에 (종목, 봉시각, 동작)에서 만든 clientOrderId 를 붙인다.
       루프가 두 번 돌거나 재시작해도 같은 주문이 두 번 나가지 않는다. 자동매매에서
       가장 흔하고 가장 비싼 사고가 중복 주문이다.
    2) **거래소가 진실이다**: 포지션은 매번 거래소에서 읽는다. 내 기록을 믿지 않는다.
    3) **수량·가격 반올림**: 거래소가 정한 tickSize/stepSize/최소명목을 지킨다.
       안 지키면 주문이 거부되는데, 그 거부를 놓치면 "냈다고 생각했는데 없는" 상태가 된다.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
SECRETS = ROOT / "config" / "secrets.yaml"

TESTNET = "https://testnet.binancefuture.com"
LIVE = "https://fapi.binance.com"
RECV_WINDOW = 5000
RETRY_STATUS = (500, 502, 503, 504)
MAX_RETRIES = 3


class ExchangeError(RuntimeError):
    """거래소가 거부했거나 통신이 실패했다."""


class MissingCredentials(ExchangeError):
    def __init__(self) -> None:
        super().__init__(
            "API 키가 없습니다. 둘 중 하나로 넣으세요.\n"
            "  · 환경변수 BINANCE_API_KEY / BINANCE_API_SECRET\n"
            f"  · {SECRETS} 에 api_key / api_secret\n"
            "테스트넷 키는 https://testnet.binancefuture.com 에서 무료로 만듭니다.")


def load_credentials(testnet: bool = True) -> tuple[str, str]:
    """
    환경변수 → secrets.yaml 순으로 찾는다.
    테스트넷과 실전 키를 따로 둘 수 있게 접두사를 지원한다.
    """
    prefix = "BINANCE_TESTNET" if testnet else "BINANCE"
    key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("BINANCE_API_KEY")
    sec = os.environ.get(f"{prefix}_API_SECRET") or os.environ.get("BINANCE_API_SECRET")
    if key and sec:
        return key, sec

    if SECRETS.exists():
        import yaml
        raw = yaml.safe_load(SECRETS.read_text(encoding="utf-8")) or {}
        block = raw.get("testnet" if testnet else "live", raw)
        key = block.get("api_key") or raw.get("api_key")
        sec = block.get("api_secret") or raw.get("api_secret")
        if key and sec:
            return str(key), str(sec)

    raise MissingCredentials()


@dataclass(frozen=True)
class SymbolRules:
    """거래소가 정한 반올림 규칙. 안 지키면 주문이 거부된다."""
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def round_price(self, price: float) -> float:
        q = (Decimal(str(price)) / self.tick_size).to_integral_value(ROUND_DOWN)
        return float(q * self.tick_size)

    def round_qty(self, qty: float) -> float:
        q = (Decimal(str(qty)) / self.step_size).to_integral_value(ROUND_DOWN)
        return float(q * self.step_size)

    def is_tradable(self, qty: float, price: float) -> tuple[bool, str]:
        if Decimal(str(qty)) < self.min_qty:
            return False, f"최소 수량 미달 ({qty} < {self.min_qty})"
        notional = Decimal(str(qty)) * Decimal(str(price))
        if notional < self.min_notional:
            return False, f"최소 명목가 미달 ({notional:.2f} < {self.min_notional})"
        return True, ""


class BinanceFutures:
    """USDT-M 선물. 조회와 주문 모두 여기를 지난다."""

    def __init__(self, *, testnet: bool = True, key: str | None = None,
                 secret: str | None = None, timeout: int = 15) -> None:
        self.testnet = testnet
        self.base = TESTNET if testnet else LIVE
        self.timeout = timeout
        self._key, self._secret = (key, secret) if key and secret else (None, None)
        self._session = requests.Session()
        self._rules: dict[str, SymbolRules] = {}
        self._time_offset = 0

    # ─── 인증 ────────────────────────────────────────────────────

    def _creds(self) -> tuple[str, str]:
        if not self._key or not self._secret:
            self._key, self._secret = load_credentials(self.testnet)
        return self._key, self._secret

    def _sign(self, params: dict) -> str:
        _, secret = self._creds()
        query = urllib.parse.urlencode(params, doseq=True)
        sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    # ─── 호출 ────────────────────────────────────────────────────

    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False) -> Any:
        params = dict(params or {})
        headers = {}
        if signed:
            key, _ = self._creds()
            headers["X-MBX-APIKEY"] = key
            params["timestamp"] = int(time.time() * 1000) + self._time_offset
            params["recvWindow"] = RECV_WINDOW
            body = self._sign(params)
        else:
            body = urllib.parse.urlencode(params, doseq=True)

        url = f"{self.base}{path}" + (f"?{body}" if body else "")
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.request(method, url, headers=headers,
                                             timeout=self.timeout)
                if resp.status_code in RETRY_STATUS:
                    raise ExchangeError(f"{resp.status_code} (일시적)")
                if resp.status_code >= 400:
                    # 거래소가 준 이유를 그대로 올린다. 삼키면 원인을 못 찾는다.
                    try:
                        err = resp.json()
                        raise ExchangeError(
                            f"{path} 거부 [{err.get('code')}] {err.get('msg')}")
                    except ValueError:
                        raise ExchangeError(f"{path} 실패 {resp.status_code}: {resp.text[:200]}")
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, ExchangeError) as exc:
                # 거래소가 명시적으로 거부한 것(4xx)은 재시도하지 않는다
                if isinstance(exc, ExchangeError) and "일시적" not in str(exc):
                    raise
                last = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.0 * (2 ** attempt))
        raise ExchangeError(f"{MAX_RETRIES}회 재시도 실패: {path} ({last})")

    # ─── 시간 동기 ───────────────────────────────────────────────

    def sync_time(self) -> int:
        """
        로컬 시계가 어긋나면 서명이 거부된다(-1021). 시작할 때 한 번 맞춘다.
        윈도우에서 흔한 실패라 조용히 넘기지 않는다.
        """
        server = int(self._request("GET", "/fapi/v1/time")["serverTime"])
        self._time_offset = server - int(time.time() * 1000)
        return self._time_offset

    # ─── 조회 ────────────────────────────────────────────────────

    def rules(self, symbol: str) -> SymbolRules:
        if symbol not in self._rules:
            self._load_rules()
        if symbol not in self._rules:
            raise ExchangeError(f"{symbol} 은 이 거래소에 없습니다")
        return self._rules[symbol]

    def _load_rules(self) -> None:
        info = self._request("GET", "/fapi/v1/exchangeInfo")
        for s in info.get("symbols", []):
            if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
                continue
            f = {x["filterType"]: x for x in s.get("filters", [])}
            try:
                self._rules[s["symbol"]] = SymbolRules(
                    symbol=s["symbol"],
                    tick_size=Decimal(f["PRICE_FILTER"]["tickSize"]),
                    step_size=Decimal(f["LOT_SIZE"]["stepSize"]),
                    min_qty=Decimal(f["LOT_SIZE"]["minQty"]),
                    min_notional=Decimal(
                        f.get("MIN_NOTIONAL", {}).get("notional", "0")),
                )
            except (KeyError, TypeError):
                continue

    def balance(self) -> dict[str, float]:
        """USDT 잔고. 포지션 크기를 정하는 근거다."""
        rows = self._request("GET", "/fapi/v2/balance", signed=True)
        for r in rows:
            if r.get("asset") == "USDT":
                return {"wallet": float(r["balance"]),
                        "available": float(r["availableBalance"])}
        return {"wallet": 0.0, "available": 0.0}

    def positions(self) -> dict[str, dict]:
        """
        **열린 포지션. 이것이 진실이다.** 내부 기록과 다르면 이쪽을 믿는다.
        수량 0 인 종목은 버린다.
        """
        rows = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        out = {}
        for r in rows:
            amt = float(r.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            out[r["symbol"]] = {
                "symbol": r["symbol"],
                "qty": abs(amt),
                "side": "LONG" if amt > 0 else "SHORT",
                "entry_price": float(r.get("entryPrice", 0) or 0),
                "mark_price": float(r.get("markPrice", 0) or 0),
                "unrealised": float(r.get("unRealizedProfit", 0) or 0),
                "leverage": float(r.get("leverage", 1) or 1),
            }
        return out

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    def mark_price(self, symbol: str) -> float:
        r = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(r["markPrice"])

    # ─── 주문 ────────────────────────────────────────────────────

    def place_market(self, symbol: str, side: str, qty: float, *,
                     client_id: str, reduce_only: bool = False) -> dict:
        """
        시장가 주문. client_id 로 멱등성을 보장한다 — 같은 id 로 두 번 보내면
        거래소가 두 번째를 거부한다(-4015). 그 거부는 성공으로 간주한다.
        """
        params = {
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": self._fmt_qty(symbol, qty),
            "newClientOrderId": client_id,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._order(params, client_id, symbol)

    def place_stop(self, symbol: str, side: str, stop_price: float, *,
                   client_id: str) -> dict:
        """
        손절. **reduce-only 시장가 스톱**으로 건다.

        진입 직후 바로 걸어야 한다. 프로그램이 죽거나 네트워크가 끊겨도 포지션이
        맨몸으로 남지 않게 하는 유일한 장치다. closePosition=true 라 수량을
        따로 관리하지 않아도 된다.
        """
        params = {
            "symbol": symbol, "side": side, "type": "STOP_MARKET",
            "stopPrice": self._fmt_price(symbol, stop_price),
            "closePosition": "true", "workingType": "MARK_PRICE",
            "newClientOrderId": client_id,
        }
        return self._order(params, client_id, symbol)

    def place_target(self, symbol: str, side: str, stop_price: float, *,
                     client_id: str) -> dict:
        params = {
            "symbol": symbol, "side": side, "type": "TAKE_PROFIT_MARKET",
            "stopPrice": self._fmt_price(symbol, stop_price),
            "closePosition": "true", "workingType": "MARK_PRICE",
            "newClientOrderId": client_id,
        }
        return self._order(params, client_id, symbol)

    def cancel_all(self, symbol: str) -> dict:
        return self._request("DELETE", "/fapi/v1/allOpenOrders",
                             {"symbol": symbol}, signed=True)

    def _order(self, params: dict, client_id: str, symbol: str) -> dict:
        try:
            return self._request("POST", "/fapi/v1/order", params, signed=True)
        except ExchangeError as exc:
            # -4015 / "Duplicate order sent" — 같은 clientOrderId 를 이미 받았다.
            # 재시도나 재시작으로 생기는 정상적인 경로다. 실패가 아니다.
            if "-4015" in str(exc) or "duplicate" in str(exc).lower():
                return {"clientOrderId": client_id, "symbol": symbol,
                        "status": "DUPLICATE_IGNORED", "duplicate": True}
            raise

    # ─── 포맷 ────────────────────────────────────────────────────

    def _fmt_qty(self, symbol: str, qty: float) -> str:
        r = self.rules(symbol)
        return f"{r.round_qty(qty):f}".rstrip("0").rstrip(".") or "0"

    def _fmt_price(self, symbol: str, price: float) -> str:
        r = self.rules(symbol)
        return f"{r.round_price(price):f}".rstrip("0").rstrip(".") or "0"

    # ─── 상태 ────────────────────────────────────────────────────

    def describe(self) -> dict:
        return {
            "venue": "바이낸스 USDT-M 선물",
            "mode": "테스트넷" if self.testnet else "실전",
            "base": self.base,
            "has_credentials": bool(self._key and self._secret),
        }
