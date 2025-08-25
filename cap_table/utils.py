from __future__ import annotations
from datetime import datetime, date
from typing import Any


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(d: str) -> date:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return date.today()


def years_between(d0: str, d1: date) -> float:
    if not d0:
        return 0.0
    return max(0.0, (d1 - _parse_date(d0)).days / 365.25)


def money_fmt(x: float, currency: str = "€") -> str:
    if x != x:  # NaN
        return "–"
    return f"{currency}{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def shares_fmt(x: float) -> str:
    if x != x:
        return "–"
    return f"{int(x):,}".replace(",", ".")
