"""
유튜브 전략의 과거 검증 — 켜기 전에 이 전략이 과거에 어땠는지 본다.

방법은 scripts/validate_fx.py 와 같다.

    · 전략을 **단독으로** 과거 봉에 돌린다. 체결 규칙은 엔진과 같다
      (마감봉 신호, 다음 봉 시가 체결, ATR 손절, 설정의 손익비).
    · 비용은 봉마다 기록된 **실제 스프레드**다 (편도 = 스프레드/2) + 슬리피지 0.5bp.
      비용을 뺀 것과 빼지 않은 것을 둘 다 낸다 — 우위가 애초에 없는 것인지,
      있는데 비용이 먹는 것인지는 다른 이야기다.
    · 통화쌍은 함께 움직인다. 주 단위 군집으로 검정한다 — 종목 수로 t 를 부풀리지 않는다.
    · 연도별 성적을 따로 낸다. 한두 해가 전체를 끌고 간 것인지 보려는 것이다.

**읽을 때 조심할 것.** 진입 조건만 영상에서 온 것이고 청산은 이 앱의 방식이다.
영상의 익절·손절 규칙으로는 결과가 다를 수 있다. 지원하지 않는 지표는 LLM 이
근사했다. 그리고 이것은 한 번의 시험이다 — 여러 전략을 검증해 좋은 것만 고르면
그 고르는 행위가 결과를 부풀린다.

**별도 프로세스에서, 낮은 우선순위로 돈다.** 실제로 있었던 일이다: 검증을 엔진과
같은 PC 에서 그냥 돌렸더니 CPU 를 다 먹어 엔진의 한 바퀴가 몇 시간씩 밀렸다.
같은 프로세스의 스레드로 돌리면 더 나쁘다(GIL). 그래서 서버는 이 모듈을 자식
프로세스로 띄우고(`python -m src.youtube.validation <이름>`), 진행 상황은 파일로
주고받는다. 한 번에 하나만 돈다.

결과는 전략 파일에 저장된다. 같은 이름으로 다시 분석해 덮어쓰면 그 결과는
새 전략의 것이 아니므로 버려진다.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import ROOT, Config, load_config
from . import store
from .strategy import SHORT_GROUP, YoutubeStrategy, strategy_group

STATUS_FILE = "runs/youtube_validation.json"
# 받아 볼 봉 개수. 1시간봉 6만 개가 약 9년 8개월이다. 브로커가 그만큼 주지 않으면
# 절반, 그 절반으로 줄여 다시 받는다.
TARGET_BARS = {"5m": 100_000, "15m": 80_000, "30m": 60_000,
               "1h": 60_000, "4h": 20_000, "1d": 5_000}
# 이보다 거래가 적으면 판정하지 않는다. 몇 건의 평균은 우연이다.
MIN_TRADES = 30

Fetch = Callable[[str, str, int], pd.DataFrame]


class ValidationError(Exception):
    """사용자에게 그대로 보여 줄 수 있는 실패."""


def interval_for(spec: dict, cfg: Config) -> str:
    """
    검증에 쓸 봉. 기본 군 전략은 엔진 봉(더 긴 봉은 전략이 묶어서 계산한다),
    단기 군 전략은 자기 시간봉이다 — 엔진이 그 전략을 돌리는 방식과 같아야 한다.
    """
    base = str(cfg.get("fx.interval", "1h"))
    group = strategy_group(spec["timeframe"], base)
    if group is None:
        raise ValidationError(
            f"{spec['timeframe']}봉 전략은 검증할 수 없습니다 — 그 봉의 과거 데이터를 "
            "충분히 받을 수 없습니다")
    return spec["timeframe"] if group == SHORT_GROUP else base


def _history(fetch: Fetch, symbol: str, interval: str) -> pd.DataFrame:
    want = TARGET_BARS.get(interval, 60_000)
    error: Exception | None = None
    for count in (want, want // 2, want // 4):
        try:
            return fetch(symbol, interval, count)
        except Exception as exc:          # 브로커가 그만큼의 과거를 주지 않는다
            error = exc
    raise ValidationError(f"{symbol} {interval} 봉을 받지 못했습니다: {error}")


def summarise(trades: pd.DataFrame) -> dict[str, Any]:
    """
    거래 목록(symbol, mode, r, exit_time)을 판정으로 줄인다.
    mode 는 net(비용 차감) 또는 gross(비용 제외)다.
    """
    from ..validation.stats import clustered_mean_test, time_clusters

    def test(frame: pd.DataFrame):
        r = frame["r"].to_numpy("float64")
        return r, clustered_mean_test(r, time_clusters(frame["exit_time"], freq="W"))

    net = trades[trades["mode"] == "net"]
    gross = trades[trades["mode"] == "gross"]
    if len(net) < MIN_TRADES:
        return {"trades": int(len(net)), "verdict": "거래부족",
                "net_r": None, "gross_r": None, "win_rate": None, "t": None,
                "p": None, "gross_t": None, "clusters": 0, "per_year": None,
                "positive_pairs": 0, "by_year": []}

    r, result = test(net)
    gross_r, gross_result = test(gross)
    when = pd.to_datetime(net["exit_time"], utc=True)
    years = max((when.max() - when.min()).days / 365.25, 0.01)
    yearly = net.assign(year=when.dt.year).groupby("year")["r"].agg(["mean", "count"])

    def number(value: float) -> float | None:
        return round(float(value), 4) if np.isfinite(value) else None

    return {
        "trades": int(len(r)),
        "per_year": round(len(r) / years),
        "net_r": number(r.mean()),
        "gross_r": number(gross_r.mean()),
        "win_rate": number((r > 0).mean()),
        "t": number(result.t),
        "p": number(result.p),
        "gross_t": number(gross_result.t),
        "clusters": int(result.n_clusters),
        "verdict": result.verdict,
        "positive_pairs": int((net.groupby("symbol")["r"].mean() > 0).sum()),
        "by_year": [{"year": int(year), "r": round(float(row["mean"]), 4),
                     "trades": int(row["count"])} for year, row in yearly.iterrows()],
    }


def run(spec: dict, cfg: Config | None = None, *,
        symbols: list[tuple[str, str]] | None = None,
        fetch: Fetch | None = None,
        point: Callable[[str], float] | None = None,
        progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """
    전략 하나를 과거 봉에 돌려 판정을 돌려준다. 주문은 내지 않는다.

    symbols 는 (표준 이름, 브로커 심볼) 목록이다. 주지 않으면 감시 통화쌍 전체.
    fetch·point 는 봉과 호가 단위를 주는 함수다. 주지 않으면 MT5 에서 읽는다.
    """
    from ..backtest.engine import ExecParams
    from ..backtest.walkforward import run_standalone
    from ..config import Costs
    from ..indicators import compute_all
    from ..regime import classify

    cfg = cfg or load_config()
    interval = interval_for(spec, cfg)
    if fetch is None or point is None or symbols is None:
        from ..fx import watchlist
        from ..fx.mt5 import available_symbols, bars, symbol_info
        fetch = fetch or bars
        point = point or (lambda name: symbol_info(name).point)
        if symbols is None:
            symbols = [(p.canonical, p.mt5)
                       for p in watchlist.pairs(cfg, available_symbols(only_forex=True))]

    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    strategy = YoutubeStrategy(spec)
    started = time.time()
    rows: list[dict] = []
    first, last, done, failed = None, None, 0, []

    for index, (canonical, broker_symbol) in enumerate(symbols, 1):
        if progress:
            progress(index - 1, len(symbols), canonical)
        try:
            df = _history(fetch, broker_symbol, interval)
        except ValidationError:
            failed.append(canonical)
            continue
        feats = compute_all(df)
        regimes = classify(feats)["regime"]
        half = float((df["spread"] * point(broker_symbol) / df["close"]).median()) / 2.0 \
            if "spread" in df.columns else 0.0
        for mode, costs in (("net", Costs(fee_rate=half, slippage_bps=0.5)),
                            ("gross", Costs(fee_rate=0.0, slippage_bps=0.0))):
            result = run_standalone(
                df, feats, [strategy], interval=interval, costs=costs, params=params,
                initial_equity=10_000.0, minute_df=None, regimes=regimes,
                symbol=canonical)[strategy.name]
            for _, trade in result.trades.iterrows():
                rows.append({"symbol": canonical, "mode": mode,
                             "r": trade.get("r_multiple", np.nan),
                             "exit_time": trade.get("exit_time")})
        first = df.index[0] if first is None else min(first, df.index[0])
        last = df.index[-1] if last is None else max(last, df.index[-1])
        done += 1

    if not done:
        raise ValidationError("어느 통화쌍의 봉도 받지 못했습니다 — MT5 연결을 확인하세요")
    trades = pd.DataFrame(rows, columns=["symbol", "mode", "r", "exit_time"]).dropna(subset=["r"])
    return {
        **summarise(trades),
        "interval": interval,
        "pairs": done,
        "pairs_failed": failed,
        "from": f"{first:%Y-%m-%d}",
        "to": f"{last:%Y-%m-%d}",
        "stop_mult": params.atr_stop_mult,
        "reward_ratio": params.reward_ratio,
        "seconds": round(time.time() - started),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── 시간봉을 말하지 않은 영상 ───────────────────────────────────
#
# 영상이 봉을 말하지 않으면 엔진 봉(1시간)으로 검증한다. 거기서 거래가 너무 적어
# 판정하지 못하면 다른 봉으로 차례로 다시 돌린다 — 그 규칙이 어느 봉을 두고 한
# 말인지 모르기 때문이다. **처음으로 판정이 나온 봉에서 멈춘다.** 성적이 가장 좋은
# 봉을 고르지 않는다: 여러 봉을 돌려 좋은 것을 고르면 그 고르는 행위가 결과를
# 부풀린다. 판정이 나온 봉은 전략의 시간봉으로 저장한다 — 엔진이 검증한 것과 같은
# 봉으로 그 전략을 돌려야 검증이 의미가 있다.
FALLBACK_TIMEFRAMES = ("1d", "30m", "15m", "5m")


def run_with_fallback(spec: dict, cfg: Config, *,
                      progress: Callable[[int, int, str, str | None], None] | None = None
                      ) -> tuple[dict[str, Any], str | None]:
    """
    (판정, 고른 시간봉). 영상이 봉을 말했거나 첫 검증에서 판정이 났으면 고른 봉은 None.
    progress 의 마지막 인자는 지금 시도하는 봉이다 (첫 검증은 None).
    """
    def attempt(candidate: dict, trying: str | None) -> dict[str, Any]:
        return run(candidate, cfg, progress=(
            (lambda i, n, symbol: progress(i, n, symbol, trying)) if progress else None))

    first = attempt(spec, None)
    if spec["timeframe"] is not None or first.get("net_r") is not None:
        return first, None
    tried = [{"timeframe": first["interval"], "trades": first["trades"]}]
    for timeframe in FALLBACK_TIMEFRAMES:
        try:
            trial = attempt({**spec, "timeframe": timeframe}, timeframe)
        except ValidationError:          # 그 봉의 과거 데이터를 받지 못했다
            tried.append({"timeframe": timeframe, "trades": None})
            continue
        tried.append({"timeframe": timeframe, "trades": trial["trades"]})
        if trial.get("net_r") is not None:
            return {**trial, "tried": tried, "timeframe_chosen": timeframe}, timeframe
    return {**first, "tried": tried}, None


# ─── 별도 프로세스로 돌리기 ──────────────────────────────────────

def _status_path(cfg: Config) -> Path:
    path = Path(str(cfg.get("youtube.validation_status", STATUS_FILE)))
    return path if path.is_absolute() else ROOT / path


def write_status(cfg: Config, **fields: Any) -> dict:
    path = _status_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return row


def read_status(cfg: Config) -> dict:
    try:
        row = json.loads(_status_path(cfg).read_text(encoding="utf-8"))
        return row if isinstance(row, dict) else {"state": "idle"}
    except (OSError, ValueError):
        return {"state": "idle"}


_process: subprocess.Popen | None = None
_start_lock = threading.Lock()
logger = logging.getLogger(__name__)


def spawn(name: str) -> subprocess.Popen:
    """
    검증을 자식 프로세스로 띄운다. **우선순위를 낮춘다** — CPU 가 모자라면 엔진이
    먼저 써야 한다. 창은 띄우지 않는다.
    """
    flags = (getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
             | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return subprocess.Popen(
        [sys.executable, "-m", "src.youtube.validation", name], cwd=str(ROOT),
        creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1"})


def running() -> bool:
    return _process is not None and _process.poll() is None


def start(name: str, cfg: Config | None = None) -> dict:
    """검증을 시작한다. 이미 하나가 돌고 있으면 거부한다 — 둘이 돌면 엔진이 밀린다."""
    global _process
    cfg = cfg or load_config()
    record = store.get(name, cfg)
    interval_for(record["spec"], cfg)          # 검증할 수 없는 전략이면 여기서 멈춘다
    with _start_lock:                          # 사람의 클릭과 대기 줄이 동시에 띄우지 않게
        if running():
            raise ValidationError("이미 검증이 돌고 있습니다. 끝난 뒤 다시 시도하세요")
        status = write_status(cfg, name=name, state="running", done=0, total=0, symbol="")
        _process = spawn(name)
    return status


# ─── 대기 줄 ─────────────────────────────────────────────────────
#
# 전략을 대입하면 바로 검증한다. 검증은 한 번에 하나만 돌기 때문에, 이미 도는
# 중이면 줄을 세우고 앞의 것이 끝나는 대로 이어서 돌린다. 줄은 메모리에만 있다 —
# 앱을 다시 켜면 사라지고, 그 전략은 "검증" 버튼으로 다시 돌리면 된다.
_queue: list[str] = []
_queue_lock = threading.Lock()
_drainer: threading.Thread | None = None


def enqueue(name: str, cfg: Config | None = None) -> None:
    """검증을 줄에 세운다. 검증할 수 없는 전략이면 ValidationError."""
    global _drainer
    cfg = cfg or load_config()
    interval_for(store.get(name, cfg)["spec"], cfg)
    with _queue_lock:
        if name not in _queue:
            _queue.append(name)
        if _drainer is None:
            _drainer = threading.Thread(target=_drain, args=(cfg,), daemon=True,
                                        name="youtube-validation-queue")
            _drainer.start()


def queued() -> list[str]:
    with _queue_lock:
        return list(_queue)


def _drain(cfg: Config) -> None:
    global _drainer
    while True:
        process = _process
        if process is not None and process.poll() is None:
            process.wait()
            continue
        with _queue_lock:
            if not _queue:
                _drainer = None
                return
            name = _queue.pop(0)
        try:
            start(name, cfg)
        except (ValidationError, store.StoreError) as exc:
            if running():        # 그 사이 사람이 다른 검증을 눌렀다 — 다시 줄 맨 앞에
                with _queue_lock:
                    _queue.insert(0, name)
            else:                # 그 사이 지워졌거나 바뀐 전략 — 건너뛴다
                logger.warning("대기하던 검증을 건너뜁니다: %s (%s)", name, exc)


def status(cfg: Config | None = None) -> dict:
    """
    지금 검증 상태. 파일에는 '도는 중'인데 프로세스가 없으면 죽은 것이다 —
    그대로 두면 화면이 영원히 '검증 중'이라고 말한다.
    """
    cfg = cfg or load_config()
    row = read_status(cfg)
    if row.get("state") == "running" and not running():
        row = {**row, "state": "failed",
               "error": "검증 프로세스가 결과 없이 끝났습니다 (앱을 다시 켰거나 중단됨)"}
    return row


def main(argv: list[str] | None = None) -> int:
    """자식 프로세스의 진입점. 결과를 전략 파일에 쓰고 상태 파일로 진행을 알린다."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("사용법: python -m src.youtube.validation <전략 이름>", file=sys.stderr)
        return 2
    name = args[0]
    cfg = load_config()
    try:
        record = store.get(name, cfg)
        result, chosen = run_with_fallback(
            record["spec"], cfg, progress=lambda i, n, symbol, trying: write_status(
                cfg, name=name, state="running", done=i, total=n, symbol=symbol,
                trying=trying))
        result["spec_created_at"] = record["created_at"]
        if not store.set_validation(name, result, cfg, timeframe=chosen):
            raise ValidationError("검증하는 동안 전략이 바뀌어 결과를 버렸습니다")
        # **통과하지 못했으면 보관함으로 옮긴다.** 손실이든, 거래가 너무 적어 판정이
        # 안 났든 켤 수 없기는 마찬가지다. 목록에는 켤 수 있는 후보만 남긴다.
        net = result.get("net_r")
        passed = isinstance(net, (int, float)) and not isinstance(net, bool) and net >= 0
        if not passed:
            store.archive(name, cfg)
        write_status(cfg, name=name, state="done", done=result["pairs"],
                     total=result["pairs"], symbol="", archived=not passed,
                     net_r=net, trades=result.get("trades"))
        return 0
    except Exception as exc:
        write_status(cfg, name=name, state="failed", error=f"{type(exc).__name__}: {exc}"
                     if not isinstance(exc, (ValidationError, store.StoreError)) else str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
