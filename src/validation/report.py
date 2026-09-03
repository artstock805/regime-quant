"""
검증 리포트 — 결과를 파일로 고정한다.

콘솔 출력만 남기면 두 가지가 불가능해진다.
  · 실행 간 비교 — "지난번보다 나아졌나"를 손으로 표를 만들어 비교하게 된다
  · 재현 — 어떤 설정으로 그 숫자가 나왔는지 기록이 없다

그래서 실행마다 `runs/<타임스탬프>/` 에 설정 스냅샷과 전 결과를 JSON 으로 남기고,
사람이 읽는 HTML 요약을 함께 만든다. 판정은 JSON 이 원본이고 HTML 은 그것을 그린다.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def _jsonable(x: Any) -> Any:
    """numpy·pandas·datetime 을 JSON 이 먹을 수 있게 바꾼다."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not np.isfinite(v) else v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, pd.DataFrame):
        return x.to_dict("records")
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "as_dict"):
        return _jsonable(x.as_dict())
    return x


def environment() -> dict:
    """재현에 필요한 환경 정보. 버전이 바뀌면 숫자가 달라질 수 있다."""
    def _v(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:      # noqa: BLE001
            return None

    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:          # noqa: BLE001
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": _v("pandas"), "numpy": _v("numpy"),
        "git_commit": git,
    }


class ValidationReport:
    """검증 결과를 모아 파일로 내보낸다."""

    def __init__(self, title: str, config: dict, meta: dict | None = None):
        self.title = title
        self.created = datetime.now(timezone.utc)
        self.config = config
        self.meta = meta or {}
        self.sections: dict[str, Any] = {}
        self.checks: dict[str, bool] = {}
        self.notes: list[str] = []

    def add(self, name: str, payload: Any, *, note: str | None = None) -> None:
        self.sections[name] = _jsonable(payload)
        if note:
            self.notes.append(f"{name}: {note}")

    def check(self, name: str, passed: bool) -> None:
        """합격 판정 항목. 리포트 맨 위에 O/X 로 요약된다."""
        self.checks[name] = bool(passed)

    @property
    def passed(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    @property
    def verdict(self) -> str:
        n = len(self.checks)
        if not n:
            return "판정 항목 없음"
        if self.passed == n:
            return f"{self.passed}/{n} 전부 통과"
        if self.passed >= max(1, n - 1):
            return f"{self.passed}/{n} 통과 — 한 항목만 미달"
        return f"{self.passed}/{n} 통과 — 우연과 구분되지 않는다"

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "title": self.title,
            "created": self.created.isoformat(),
            "environment": environment(),
            "config": _jsonable(self.config),
            "meta": _jsonable(self.meta),
            "checks": self.checks,
            "passed": self.passed,
            "total_checks": len(self.checks),
            "verdict": self.verdict,
            "notes": self.notes,
            "sections": self.sections,
        }

    # ─── 저장 ────────────────────────────────────────────────────

    def save(self, runs_dir: str | Path, name: str | None = None,
             trades: "pd.DataFrame | None" = None) -> Path:
        stamp = self.created.strftime("%Y%m%d-%H%M%S")
        out = Path(runs_dir) / (name or stamp)
        out.mkdir(parents=True, exist_ok=True)
        # 거래 원본을 함께 남긴다. 리포트의 요약 숫자만으로는 나중에 다른 질문
        # (군집 보정, 종목 수 확장 곡선 등)을 물을 수 없다. JSON 은 결론이고
        # 이 파일이 근거다.
        if trades is not None and not trades.empty:
            trades.to_parquet(out / "trades.parquet")
        (out / "report.json").write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "report.html").write_text(self.render_html(), encoding="utf-8")
        # 최신 실행을 가리키는 포인터 — 비교 스크립트가 여기서 시작한다
        (Path(runs_dir) / "latest.txt").write_text(out.name, encoding="utf-8")
        return out

    # ─── HTML ────────────────────────────────────────────────────

    def render_html(self) -> str:
        d = self.as_dict()
        rows = "".join(
            f'<div class="chk {"ok" if v else "no"}">'
            f'<span class="mark">{"O" if v else "X"}</span>{k}</div>'
            for k, v in self.checks.items()
        )
        body = "".join(self._section_html(k, v) for k, v in self.sections.items())
        cfg = json.dumps(d["config"], ensure_ascii=False, indent=2)
        env = " · ".join(f"{k} {v}" for k, v in d["environment"].items() if v)
        return _HTML.format(
            title=self.title, created=d["created"][:19].replace("T", " "),
            verdict=self.verdict, checks=rows, sections=body,
            config=cfg, env=env,
        )

    def _section_html(self, name: str, payload: Any) -> str:
        inner = ""
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            inner = _table_html(payload)
        elif isinstance(payload, dict):
            table_key = next(
                (k for k, v in payload.items()
                 if isinstance(v, list) and v and isinstance(v[0], dict)), None
            )
            scalars = {k: v for k, v in payload.items()
                       if not isinstance(v, (list, dict))}
            if scalars:
                inner += _kv_html(scalars)
            if table_key:
                inner += _table_html(payload[table_key])
            nested = {k: v for k, v in payload.items()
                      if isinstance(v, dict) and k != table_key}
            for k, v in nested.items():
                flat = {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                if flat:
                    inner += f'<div class="sub">{_esc(k)}</div>' + _kv_html(flat)
        else:
            inner = f"<pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>"
        return f'<section><h2>{_esc(name)}</h2>{inner}</section>'


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v: Any) -> str:
    if v is None:
        return '<span class="dim">—</span>'
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        s = f"{v:+.4f}" if abs(v) < 100 else f"{v:,.2f}"
        cls = "up" if v > 0 else ("down" if v < 0 else "dim")
        return f'<span class="{cls}">{s}</span>'
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, (list, tuple)):
        # 중첩 리스트(누락 구간·이상치 시각 등)를 칸에 펼치면 표가 무너진다.
        # 개수만 보여주고 전체는 JSON 에 남긴다 — 마우스를 올리면 앞부분이 보인다.
        if not v:
            return '<span class="dim">—</span>'
        peek = _esc("; ".join(str(x) for x in v[:3]))
        return f'<span class="dim" title="{peek}">{len(v)}건</span>'
    if isinstance(v, dict):
        return f'<span class="dim">{len(v)}항목</span>'
    return _esc(v)


def _table_html(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list({k: None for r in rows for k in r})
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(r.get(c))}</td>" for c in cols) + "</tr>"
        for r in rows[:200]
    )
    more = (f'<div class="dim">… {len(rows) - 200}행 생략</div>'
            if len(rows) > 200 else "")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}"


def _kv_html(d: dict) -> str:
    items = "".join(
        f'<div class="kv"><dt>{_esc(k)}</dt><dd>{_fmt(v)}</dd></div>'
        for k, v in d.items()
    )
    return f'<div class="kvs">{items}</div>'


_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--ground:#0A0E14;--panel:#111823;--rule:#1E2836;--ink:#C9D5E4;--ink-hi:#EAF1F8;
--dim:#6B7A8F;--faint:#41505F;--up:#26D98B;--down:#FF5C6E;--acc:#4DD9E8;
--mono:ui-monospace,"SFMono-Regular",Consolas,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font:14px/1.55 system-ui,-apple-system,sans-serif;
font-variant-numeric:tabular-nums}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px;color:var(--ink-hi);letter-spacing:-.01em}}
.sub{{color:var(--dim);font-size:11px;letter-spacing:.1em;text-transform:uppercase;margin:14px 0 6px}}
.meta{{color:var(--faint);font-family:var(--mono);font-size:11px;margin-bottom:18px}}
.verdict{{font-family:var(--mono);font-size:17px;font-weight:700;padding:10px 14px;
border:1px solid var(--rule);border-radius:3px;background:var(--panel);margin:0 0 14px}}
.checks{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:26px}}
.chk{{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px;
padding:6px 11px;border:1px solid var(--rule);border-radius:3px;background:var(--panel)}}
.chk .mark{{font-weight:700}}
.chk.ok{{color:var(--up);border-color:rgba(38,217,139,.32)}}
.chk.no{{color:var(--down);border-color:rgba(255,92,110,.32)}}
section{{margin:0 0 26px;border:1px solid var(--rule);border-radius:3px;background:var(--panel);
padding:14px 16px;overflow-x:auto}}
h2{{font-size:12px;letter-spacing:.11em;text-transform:uppercase;color:var(--dim);
margin:0 0 10px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px}}
th{{text-align:right;color:var(--faint);font-weight:600;padding:5px 9px;
border-bottom:1px solid var(--rule);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
td{{text-align:right;padding:4px 9px;border-bottom:1px solid #161F2B;white-space:nowrap}}
tbody tr:last-child td{{border-bottom:0}}
.kvs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:1px;
background:#161F2B;margin-bottom:10px}}
.kv{{background:var(--panel);padding:7px 10px;display:flex;justify-content:space-between;gap:12px}}
.kv dt{{color:var(--faint);font-size:11.5px;margin:0}}
.kv dd{{margin:0;font-family:var(--mono);font-size:12px;color:var(--ink-hi)}}
.up{{color:var(--up)}} .down{{color:var(--down)}} .dim{{color:var(--faint)}}
pre{{font-family:var(--mono);font-size:11px;color:var(--dim);white-space:pre-wrap;margin:0}}
details{{margin-top:8px}} summary{{cursor:pointer;color:var(--dim);font-size:12px}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<div class="meta">{created} UTC · {env}</div>
<div class="verdict">{verdict}</div>
<div class="checks">{checks}</div>
{sections}
<section><h2>설정 스냅샷</h2>
<details><summary>펼치기 — 이 숫자들이 나온 설정</summary><pre>{config}</pre></details>
</section>
</div></body></html>
"""


# ─── 실행 간 비교 ────────────────────────────────────────────────

def load_report(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "report.json"
    return json.loads(p.read_text(encoding="utf-8"))


def rerender(path: str | Path) -> Path:
    """
    저장된 JSON 에서 HTML 을 다시 만든다.

    JSON 이 원본이고 HTML 은 그것을 그린 것이므로, 표현이 바뀌었을 때 비싼 계산을
    다시 돌릴 필요가 없어야 한다. 실제로 리포트 렌더링을 고친 뒤 이미 돌아간
    검증(순열 검정 포함 30분)의 HTML 만 갱신해야 하는 상황이 있었다.
    """
    p = Path(path)
    d = load_report(p)
    r = ValidationReport(d["title"], d.get("config", {}), d.get("meta", {}))
    r.created = datetime.fromisoformat(d["created"])
    r.sections = d.get("sections", {})
    r.checks = d.get("checks", {})
    r.notes = d.get("notes", [])
    out = (p if p.is_dir() else p.parent) / "report.html"
    out.write_text(r.render_html(), encoding="utf-8")
    return out


def compare_reports(paths: list[str | Path]) -> pd.DataFrame:
    """
    여러 실행의 판정 항목과 핵심 숫자를 나란히 놓는다.
    "지난번보다 나아졌나"를 손으로 표 만들지 않고 답하기 위한 것.
    """
    rows = []
    for p in paths:
        d = load_report(p)
        row = {"실행": Path(p).name, "생성": d["created"][:16].replace("T", " "),
               "판정": d["verdict"]}
        row.update({f"[{k}]": ("O" if v else "X") for k, v in d["checks"].items()})
        pooled = d["sections"].get("통합 검정", {})
        if isinstance(pooled, dict):
            for k in ("net_mean", "net_t", "gross_mean", "gross_t", "trades"):
                if k in pooled:
                    row[k] = pooled[k]
        rows.append(row)
    return pd.DataFrame(rows)
