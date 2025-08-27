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


def build_lp_rounds(raw: Dict[str, Any], liq_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
    name_to_class = {}
    for c in liq_terms.get("classes", []):
        for rn in c.get("applies_to_round_names", []):
            name_to_class[rn] = c
    lp_rounds = []
    raw_events = (raw or {}).get("events", [])
    for ev_raw in raw_events:
        if (ev_raw.get("kind") == "investment_round") and ev_raw.get("name") in name_to_class:
            c = name_to_class[ev_raw.get("name")]; rdate = ev_raw.get("date")
            
            lp_round = {
                    "name": ev_raw.get("name"),
                    "date": rdate,
                    "rate": float(c.get("simple_interest_rate")) if c.get("simple_interest_rate") is not None else 0.0,
                    "cap_multiple_total": c.get("cap_multiple_total"),
                    "tranches": []                 
            }
            
            total_invested = 0.0
            for investor, amt in (ev_raw.get("amounts_invested") or {}).items():
                lp_round["tranches"].append({
                    "investor": investor.strip(),
                    "principal": _as_float(amt),
                    "received": 0.0
                })
                total_invested += _as_float(amt)
                
            lp_round["total_invested"] = total_invested
            
            lp_rounds.append(lp_round)

    # Sort by date, later first
    lp_rounds.sort(key=lambda t: t["date"], reverse=True)

    return lp_rounds


def simulate_exit_proceeds(total_proceeds: float, exit_date: date, cap_table_after: Dict[str, float], raw_data: Dict[str, Any], liq_terms: Dict[str, Any]) -> Dict[str, Any]:
    proceeds_left = max(0.0, float(total_proceeds))
    lp_rounds = build_lp_rounds(raw_data, liq_terms)
    shares_by_holder = {h: float(s) for h, s in cap_table_after.items()}
    totals_by_holder = compute_total_invested(raw_data)
    total_shares = sum(shares_by_holder.values()) or 1.0
    
    payouts_lp: Dict[str, dict] = {}
    payouts_participation: Dict[str, float] = {}

    # First LPs are satisfied before any participation

    for lp_round in lp_rounds:
        # Compute the LP interest of the complete round
        yrs = years_between(lp_round["date"], exit_date)
        accrued = lp_round["total_invested"] * (1.0 + lp_round["rate"] * yrs)

        # Max at the proceeds left in the round
        accrued = min(accrued, proceeds_left)

        # Split the accrued pro-rata invested in this round
        for tr in lp_round["tranches"]:
            investor_lp_claim = tr["principal"] / lp_round["total_invested"] * accrued if lp_round["total_invested"] > 0 else 0

            # Check if the claim is not capped
            if lp_round["cap_multiple_total"] is not None:
                total_invest = totals_by_holder.get(tr["investor"], 0.0)
                pro_rata_proceeds = (shares_by_holder[tr["investor"]] / total_shares) * total_proceeds
                if pro_rata_proceeds >= total_invest * lp_round["cap_multiple_total"] + total_invest * lp_round["rate"] * yrs:
                    investor_lp_claim = 0.0  # void the LP claim

            # Payout
            tr["payout"] = max(0.0, investor_lp_claim)
            proceeds_left -= tr["payout"]

            # Add to payouts for that investor
            investor_lp_dict = payouts_lp.get(tr["investor"], {})
            investor_lp_dict[lp_round["name"]] = tr["payout"]
            payouts_lp[tr["investor"]] = investor_lp_dict

    # Now handle participation payouts
    for holder, shares in shares_by_holder.items():
        if shares <= 0:
            continue
        amt = proceeds_left * (shares / total_shares)
        if amt <= 0:
            continue
        payouts_participation[holder] = payouts_participation.get(holder, 0.0) + amt
    proceeds_left = 0.0
    totals: Dict[str, float] = {}

    for h, v in payouts_lp.items():
        totals[h] = totals.get(h, 0.0) + sum(v.values())
    for h, v in payouts_participation.items():
        totals[h] = totals.get(h, 0.0) + v
    if -1e-6 < proceeds_left < 0:
        proceeds_left = 0.0

    return {
        "payouts_lp": payouts_lp,
        "payouts_participation": payouts_participation,
        "totals": totals,
        "unallocated": max(0.0, proceeds_left)
    }


def compute_total_invested(raw: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for ev in (raw or {}).get("events", []):
        if ev.get("kind") == "investment_round":
            for investor, amt in (ev.get("amounts_invested") or {}).items():
                k = str(investor).strip()
                totals[k] = totals.get(k, 0.0) + _as_float(amt)
    return totals
