from __future__ import annotations
from typing import Dict, List, Tuple, Any
from .models import RoundSummary
from .utils import _as_float
import pandas as pd


def normalize_event(ev: Dict[str, Any]) -> RoundSummary:
    kind = ev.get("kind", "investment_round")
    name = ev.get("name", "Unnamed")
    date_s = ev.get("date", "")
    rs = RoundSummary(kind=kind, name=name, date=date_s)
    if kind == "investment_round":
        rs.amounts_invested = {k.strip(): _as_float(v) for k, v in (ev.get("amounts_invested") or {}).items()}
        rs.shares_received = {k.strip(): _as_float(v) for k, v in (ev.get("shares_received") or {}).items()}
        rs.shares_to_vsp = {k.strip(): _as_float(v) for k, v in (ev.get("shares_to_vsp") or {}).items()}
        rs.new_money = sum(rs.amounts_invested.values())
        rs.new_shares = sum(rs.shares_received.values())
        rs.price_per_share = (rs.new_money / rs.new_shares) if rs.new_shares else float("nan")
    elif kind == "vsp_issue":
        rs.vsp_issued = {k.strip(): _as_float(v) for k, v in (ev.get("vsp_received") or {}).items()}
    return rs


def compute_cumulative_states(events: List[RoundSummary]) -> Tuple[List[RoundSummary], List[Dict[str, float]]]:
    cap_tables: List[Dict[str, float]] = []
    holders: Dict[str, float] = {}
    vsp_pool_name = "VSP Pool"
    for ev in events:
        if ev.kind == "investment_round":
            for holder, sh in ev.shares_received.items():
                holders[holder] = holders.get(holder, 0.0) + sh
            if ev.shares_to_vsp:
                holders[vsp_pool_name] = holders.get(vsp_pool_name, 0.0)
                for holder, sh in ev.shares_to_vsp.items():
                    holders[holder] = holders.get(holder, 0.0) - sh
                    holders[vsp_pool_name] += sh
        elif ev.kind == "vsp_issue":
            if ev.vsp_issued:
                holders[vsp_pool_name] = holders.get(vsp_pool_name, 0.0)
                for holder, sh in ev.vsp_issued.items():
                    holders[holder] = holders.get(holder, 0.0) + sh
                    holders[vsp_pool_name] -= sh
        cap_tables.append(dict(holders))
    return events, cap_tables


def compute_valuations(events: List[RoundSummary], cap_tables: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out = []
    prev_total_shares = 0.0
    for idx, ev in enumerate(events):
        totals = {"name": ev.name, "date": ev.date, "kind": ev.kind, "price_per_share": float("nan"), "pre_money": float("nan"), "post_money": float("nan"), "new_money": float("nan"), "new_shares": float("nan")}
        if ev.kind == "investment_round":
            price = ev.price_per_share if ev.price_per_share == ev.price_per_share else float("nan")
            new_money = ev.new_money; new_shares = ev.new_shares
            pre = prev_total_shares * price if price == price else float("nan")
            post = pre + new_money if (pre == pre and new_money == new_money) else float("nan")
            totals.update({"price_per_share": price, "pre_money": pre, "post_money": post, "new_money": new_money, "new_shares": new_shares})
        prev_total_shares = sum(cap_tables[idx].values()) if idx < len(cap_tables) else prev_total_shares
        out.append(totals)
    return out


def cap_table_dataframe(holdings: Dict[str, float]) -> pd.DataFrame:
    total_shares = sum(holdings.values()) if holdings else 0.0
    rows = []
    for holder, sh in sorted(holdings.items(), key=lambda kv: kv[1], reverse=True):
        pct = (sh / total_shares * 100.0) if total_shares else 0.0
        rows.append({"Holder": holder, "Shares": sh, "Ownership %": pct})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Shares"] = df["Shares"].map(lambda x: round(x, 4))
        df["Ownership %"] = df["Ownership %"].map(lambda x: round(x, 4))
    return df


def event_detail_dataframe(ev: RoundSummary) -> pd.DataFrame:
    if ev.kind == "investment_round":
        rows = []
        keys = set(ev.amounts_invested.keys()) | set(ev.shares_received.keys())
        for k in sorted(keys):
            rows.append({"Investor": k, "Invested": ev.amounts_invested.get(k, 0.0), "New Shares": ev.shares_received.get(k, 0.0)})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["Invested"] = df["Invested"].map(lambda x: round(x, 2))
            df["New Shares"] = df["New Shares"].map(lambda x: round(x, 4))
        return df
    elif ev.kind == "vsp_issue":
        rows = [{"Recipient": k, "VSP Granted (Shares)": v} for k, v in sorted(ev.vsp_issued.items())]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["VSP Granted (Shares)"] = df["VSP Granted (Shares)"].map(lambda x: round(x, 4))
        return df
    return pd.DataFrame()
