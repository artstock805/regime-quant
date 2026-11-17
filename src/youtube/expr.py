"""
진입 조건 수식의 안전한 평가기.

LLM 이 만든 수식 문자열을 `eval()` 로 실행하지 않는다. 허용하는 것은 숫자·변수·
사칙연산·비교·and/or/not·괄호와 abs/min/max 호출뿐이고, 속성 접근이나 그 밖의
호출은 문법 단계에서 거부한다. 자막은 남이 쓴 글이고 그 글에서 나온 문자열이
주문 판단까지 간다 — 여기가 그 경계다.

**봉 전체를 한 번에 계산한다.** 변수는 봉마다 값이 있는 Series 이고 결과는
봉마다 참/거짓이다. 참조한 값이 하나라도 NaN 인 봉(지표 워밍업)은 거짓으로
둔다. 그러지 않으면 `not (close > ema_200)` 이 워밍업 구간에서 참이 된다.

**값 식은 조건으로 받지 않는다.** `close` 나 `close + 1` 은 0 이 아니므로 항상
참이고, 그대로 두면 매 봉 진입하는 전략이 된다. 최상위는 비교여야 한다.
"""

from __future__ import annotations

import ast
import operator
from typing import Mapping

import numpy as np
import pandas as pd

MAX_EXPRESSION_LENGTH = 500

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
_COMPARE = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCTIONS = ("abs", "min", "max")
_PLAIN = (ast.Expression, ast.Name, ast.Load, ast.BoolOp, ast.And, ast.Or,
          ast.Compare, ast.Not, ast.USub, *_BINARY, *_COMPARE)


class ExpressionError(ValueError):
    """허용되지 않는 문법이거나 모르는 변수를 쓴 수식."""


def _is_allowed(node: ast.AST) -> bool:
    if isinstance(node, _PLAIN):
        return True
    if isinstance(node, ast.BinOp):
        return type(node.op) in _BINARY
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.Not, ast.USub))
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float)
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS) \
                or node.keywords:
            return False
        return len(node.args) == 1 if node.func.id == "abs" else len(node.args) >= 2
    return False


def _is_condition(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.BoolOp):
        return all(_is_condition(v) for v in node.values)
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            and _is_condition(node.operand))


def _names(tree: ast.AST) -> set[str]:
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and id(n) not in called}


def _parse(expr: str) -> ast.Expression:
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"수식이 너무 깁니다 ({len(expr)}자, 최대 {MAX_EXPRESSION_LENGTH}자)")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"수식 문법 오류: {expr!r} ({exc.msg})") from exc
    for node in ast.walk(tree):
        if not _is_allowed(node):
            raise ExpressionError(
                f"허용되지 않는 수식 요소 {type(node).__name__}: {expr!r}")
    if not _is_condition(tree.body):
        raise ExpressionError(f"조건식이 아닙니다 (비교 연산이 필요): {expr!r}")
    if not _names(tree):
        raise ExpressionError(f"변수가 하나도 없는 수식입니다: {expr!r}")
    return tree


def names_in(expr: str) -> set[str]:
    """수식이 참조하는 변수 이름. 허용 범위를 벗어난 수식이면 ExpressionError."""
    return _names(_parse(expr))


def _eval(node: ast.AST, columns: Mapping[str, pd.Series]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return columns[node.id]
    if isinstance(node, ast.BoolOp):
        join = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = _eval(node.values[0], columns)
        for value in node.values[1:]:
            out = join(out, _eval(value, columns))
        return out
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, columns)
        return np.logical_not(operand) if isinstance(node.op, ast.Not) else -operand
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_eval(node.left, columns),
                                      _eval(node.right, columns))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, columns)
        out = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, columns)
            out = np.logical_and(out, _COMPARE[type(op)](left, right))
            left = right
        return out
    args = [_eval(a, columns) for a in node.args]
    if node.func.id == "abs":
        return np.abs(args[0])
    pick = np.minimum if node.func.id == "min" else np.maximum
    out = args[0]
    for arg in args[1:]:
        out = pick(out, arg)
    return out


def evaluate(expr: str, columns: Mapping[str, pd.Series], index: pd.Index) -> pd.Series:
    """수식을 봉마다 평가한 참/거짓 Series. 참조 값이 NaN 인 봉은 거짓."""
    tree = _parse(expr)
    names = _names(tree)
    unknown = names - set(columns)
    if unknown:
        raise ExpressionError(f"모르는 변수: {sorted(unknown)}")
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = _eval(tree.body, columns)
    except (ArithmeticError, TypeError) as exc:
        # 상수끼리 0 으로 나누기, 400자리 정수(float 변환 오버플로) 등.
        # ExpressionError 로 바꿔야 저장 전 검증과 LLM 재시도가 이걸 잡는다.
        raise ExpressionError(f"계산할 수 없는 수식입니다: {expr!r} ({exc})") from exc
    result = pd.Series(raw, index=index).fillna(False).astype(bool)
    ready = pd.Series(True, index=index)
    for name in names:
        ready &= columns[name].notna()
    return result & ready
