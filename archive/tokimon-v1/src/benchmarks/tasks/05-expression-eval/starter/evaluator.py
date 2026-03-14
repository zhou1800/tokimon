"""Expression evaluator."""

from __future__ import annotations


def evaluate(expr: str) -> int:
    tokens = expr.split()
    if not tokens:
        raise ValueError("empty expression")
    total = int(tokens[0])
    idx = 1
    while idx < len(tokens):
        op = tokens[idx]
        rhs = int(tokens[idx + 1])
        if op == "+":
            total += rhs
        elif op == "-":
            total -= rhs
        elif op == "*":
            total *= rhs
        elif op == "/":
            total //= rhs
        idx += 2
    return total
