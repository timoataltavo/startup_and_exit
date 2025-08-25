from __future__ import annotations
from typing import Dict, Any, List
from datetime import date
from .utils import _as_float, years_between, _parse_date
from .models import RoundSummary


def extract_liquidation_terms(raw: Dict[str, Any]) -> Dict[str, Any]:
    terms = (raw or {}).get("liquidation_terms", {})
    classes = list(terms.get("classes", []))
    events = (raw or {}).get("events", [])
    round_dates: Dict[str, date] = {}
    for ev in events:
        rn = ev.get("name"); d = ev.get("date")
        if rn and d:
            try:
                round_dates[rn] = _parse_date(d)
            except (TypeError, ValueError):  # pragma: no cover
                pass

    def latest_date_for_class(c: Dict[str, Any]) -> date:
        latest: date | None = None
        for rn in c.get("applies_to_round_names", []) or []:
            rd = round_dates.get(rn)
            if rd and (latest is None or rd > latest):
                latest = rd
        return latest or date(1900, 1, 1)

    classes_sorted = sorted(classes, key=latest_date_for_class, reverse=True)
    return {"classes": classes_sorted}


def build_investment_tranches(_events: List[RoundSummary], raw: Dict[str, Any], liq_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
    name_to_class = {}
    for c in liq_terms.get("classes", []):
        for rn in c.get("applies_to_round_names", []):
            name_to_class[rn] = c
    tranches: List[Dict[str, Any]] = []
    raw_events = (raw or {}).get("events", [])
    for ev_raw in raw_events:
        if (ev_raw.get("kind") == "investment_round") and ev_raw.get("name") in name_to_class:
            c = name_to_class[ev_raw.get("name")]; rdate = ev_raw.get("date")
            for investor, amt in (ev_raw.get("amounts_invested") or {}).items():
                tranches.append({
                    "investor": investor.strip(),
                    "round_name": ev_raw.get("name"),
                    "date": rdate,
                    "principal": _as_float(amt),
                    "class_name": c.get("name"),
                    "rate": float(c.get("simple_interest_rate")) if c.get("simple_interest_rate") is not None else 0.0,
                    "cap_multiple_total": c.get("cap_multiple_total"),
                    "received": 0.0,
                })
    order_of_class = {c.get("name"): idx for idx, c in enumerate(liq_terms.get("classes", []))}
    tranches.sort(key=lambda t: order_of_class.get(t["class_name"], 9999))
    return tranches


def simulate_exit_proceeds(total_proceeds: float, exit_date: date, cap_table_after: Dict[str, float], events: List[RoundSummary], raw_data: Dict[str, Any], liq_terms: Dict[str, Any]) -> Dict[str, Any]:
    proceeds_left = max(0.0, float(total_proceeds))
    tranches = build_investment_tranches(events, raw_data, liq_terms)
    shares_by_holder = {h: float(s) for h, s in cap_table_after.items()}
    total_shares = sum(shares_by_holder.values()) or 1.0
    payouts_lp: Dict[str, float] = {}
    payouts_participation: Dict[str, float] = {}
    lp_tranche_records: List[Dict[str, Any]] = []
    for tr in tranches:
        yrs = years_between(tr["date"], exit_date)
        accrued = tr["principal"] * (1.0 + tr["rate"] * yrs)
        cap_total = tr["cap_multiple_total"]
        if cap_total is not None:
            accrued = min(accrued, cap_total * tr["principal"])
        tr["lp_claim"] = max(0.0, accrued)
    for tr in tranches:
        if proceeds_left <= 0:
            break
        pay = min(tr["lp_claim"], proceeds_left)
        if pay > 0:
            payouts_lp[tr["investor"]] = payouts_lp.get(tr["investor"], 0.0) + pay
            tr["received"] += pay
            proceeds_left -= pay
            lp_tranche_records.append({
                "investor": tr["investor"],
                "round_name": tr["round_name"],
                "class_name": tr["class_name"],
                "lp_claim": tr["lp_claim"],
                "lp_paid": pay,
            })
    if proceeds_left > 0 and total_shares > 0:
        for h, s in shares_by_holder.items():
            if s <= 0:
                continue
            amt = proceeds_left * (s / total_shares)
            if amt <= 0:
                continue
            payouts_participation[h] = payouts_participation.get(h, 0.0) + amt
        proceeds_left = 0.0
    totals: Dict[str, float] = {}
    for h, v in payouts_lp.items():
        totals[h] = totals.get(h, 0.0) + v
    for h, v in payouts_participation.items():
        totals[h] = totals.get(h, 0.0) + v
    if -1e-6 < proceeds_left < 0:
        proceeds_left = 0.0
    by_class: Dict[str, float] = {}
    class_of_investor = {tr["investor"]: tr["class_name"] for tr in tranches}
    for h, v in totals.items():
        c = class_of_investor.get(h, "Common/Other")
        by_class[c] = by_class.get(c, 0.0) + v
    return {
        "payouts_lp": payouts_lp,
        "payouts_participation": payouts_participation,
        "totals": totals,
        "by_class": by_class,
        "unallocated": max(0.0, proceeds_left),
        "lp_by_tranche": lp_tranche_records,
    }


def compute_total_invested(raw: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for ev in (raw or {}).get("events", []):
        if ev.get("kind") == "investment_round":
            for investor, amt in (ev.get("amounts_invested") or {}).items():
                k = str(investor).strip()
                totals[k] = totals.get(k, 0.0) + _as_float(amt)
    return totals
