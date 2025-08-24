import json
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from datetime import datetime, date
import math

import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------
# Data structures & utilities (must stay BEFORE first st.set_page_config for tests)
# ------------------------------

@dataclass
class RoundSummary:
    kind: str
    name: str
    date: str
    new_money: float = 0.0
    new_shares: float = 0.0
    price_per_share: float = float("nan")
    pre_money: float = float("nan")
    post_money: float = float("nan")
    amounts_invested: Dict[str, float] = field(default_factory=dict)
    shares_received: Dict[str, float] = field(default_factory=dict)
    shares_to_vsp: Dict[str, float] = field(default_factory=dict)
    vsp_issued: Dict[str, float] = field(default_factory=dict)


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


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


def _parse_date(d: str) -> date:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return date.today()


def years_between(d0: str, d1: date) -> float:
    if not d0:
        return 0.0
    return max(0.0, (d1 - _parse_date(d0)).days / 365.25)


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
            except Exception:  # pragma: no cover
                pass
    def latest_date_for_class(c: Dict[str, Any]) -> date:
        latest: date | None = None
        for rn in c.get("applies_to_round_names", []) or []:
            rd = round_dates.get(rn)
            if rd and (latest is None or rd > latest):
                latest = rd
        return latest or date(1900,1,1)
    classes_sorted = sorted(classes, key=latest_date_for_class, reverse=True)
    return {"classes": classes_sorted}


def build_investment_tranches(events: List[RoundSummary], raw: Dict[str, Any], liq_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                tranches.append({"investor": investor.strip(), "round_name": ev_raw.get("name"), "date": rdate, "principal": _as_float(amt), "class_name": c.get("name"), "rate": float(c.get("simple_interest_rate")) if c.get("simple_interest_rate") is not None else 0.0, "cap_multiple_total": c.get("cap_multiple_total"), "received": 0.0})
    order_of_class = {c.get("name"): idx for idx, c in enumerate(liq_terms.get("classes", []))}
    tranches.sort(key=lambda t: order_of_class.get(t["class_name"], 9999))
    return tranches


def simulate_exit_proceeds(total_proceeds: float, exit_date: date, cap_table_after: Dict[str, float], events: List[RoundSummary], raw_data: Dict[str, Any], liq_terms: Dict[str, Any]) -> Dict[str, Any]:
    proceeds_left = max(0.0, float(total_proceeds))
    tranches = build_investment_tranches(events, raw_data, liq_terms)
    shares_by_holder = {h: float(s) for h, s in cap_table_after.items()}
    total_shares = sum(shares_by_holder.values()) or 1.0
    payouts_lp: Dict[str, float] = {}; payouts_participation: Dict[str, float] = {}; lp_tranche_records: List[Dict[str, Any]] = []
    for tr in tranches:
        yrs = years_between(tr["date"], exit_date)
        accrued = tr["principal"] * (1.0 + tr["rate"] * yrs)
        cap_total = tr["cap_multiple_total"]
        if cap_total is not None:
            accrued = min(accrued, cap_total * tr["principal"])
        tr["lp_claim"] = max(0.0, accrued)
    for tr in tranches:
        if proceeds_left <= 0: break
        pay = min(tr["lp_claim"], proceeds_left)
        if pay > 0:
            payouts_lp[tr["investor"]] = payouts_lp.get(tr["investor"], 0.0) + pay
            tr["received"] += pay; proceeds_left -= pay
            lp_tranche_records.append({"investor": tr["investor"], "round_name": tr["round_name"], "class_name": tr["class_name"], "lp_claim": tr["lp_claim"], "lp_paid": pay})
    if proceeds_left > 0 and total_shares > 0:
        for h, s in shares_by_holder.items():
            if s <= 0: continue
            amt = proceeds_left * (s / total_shares)
            if amt <= 0: continue
            payouts_participation[h] = payouts_participation.get(h, 0.0) + amt
        proceeds_left = 0.0
    totals: Dict[str, float] = {}
    for h, v in payouts_lp.items(): totals[h] = totals.get(h, 0.0) + v
    for h, v in payouts_participation.items(): totals[h] = totals.get(h, 0.0) + v
    if -1e-6 < proceeds_left < 0: proceeds_left = 0.0
    by_class: Dict[str, float] = {}
    class_of_investor = {tr["investor"]: tr["class_name"] for tr in tranches}
    for h, v in totals.items():
        c = class_of_investor.get(h, "Common/Other")
        by_class[c] = by_class.get(c, 0.0) + v
    return {"payouts_lp": payouts_lp, "payouts_participation": payouts_participation, "totals": totals, "by_class": by_class, "unallocated": max(0.0, proceeds_left), "lp_by_tranche": lp_tranche_records}


def compute_total_invested(raw: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for ev in (raw or {}).get("events", []):
        if ev.get("kind") == "investment_round":
            for investor, amt in (ev.get("amounts_invested") or {}).items():
                k = str(investor).strip(); totals[k] = totals.get(k, 0.0) + _as_float(amt)
    return totals


def money_fmt(x: float, currency: str = "€") -> str:
    if x != x: return "–"
    return f"{currency}{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def shares_fmt(x: float) -> str:
    if x != x: return "–"
    return f"{int(x):,}".replace(",", ".")

st.set_page_config(page_title="Cap Table Toolkit (GmbH)", page_icon="🧮", layout="wide")

# Sidebar navigation
PAGES = ["Cap Table Explorer", "Round Designer", "Exit Simulator"]
page = st.sidebar.radio("Seite wählen", PAGES)

# Shared: file upload (all pages use underlying data). Persist in session_state.
if "uploaded_raw" not in st.session_state:
    st.session_state.uploaded_raw = None
if "data_store" not in st.session_state:
    st.session_state.data_store = None

st.sidebar.markdown("### Datenquelle")
uploaded = st.sidebar.file_uploader("JSON-Datei", type=["json"], key="file_uploader_main")
if uploaded is not None:
    try:
        data = json.load(uploaded)
        st.session_state.uploaded_raw = data
        st.session_state.data_store = data
    except Exception as e:  # pragma: no cover - UI error path
        st.sidebar.error(f"Fehler: {e}")

if st.session_state.data_store is None:
    st.info("Bitte zuerst eine JSON-Datei im Sidebar laden.")
    st.stop()

raw_data: Dict[str, Any] = st.session_state.data_store
raw_events = raw_data.get("events", [])
liq_terms = extract_liquidation_terms(raw_data)

events = [normalize_event(ev) for ev in raw_events]
events_sorted = sorted(events, key=lambda e: e.date or "")
events, cap_tables = compute_cumulative_states(events_sorted)
valuations = compute_valuations(events, cap_tables)

# price history for value charts
price_history: List[float] = []
last_price = float("nan")
for v in valuations:
    p = v.get("price_per_share", float("nan"))
    if p == p:
        last_price = p
    price_history.append(last_price)

def page_cap_table_explorer():
    st.title("📊 Cap Table Explorer")
    st.markdown("Analysiere Bewertungen, Eigentumsverteilung & VSP-Bewegungen pro Event.")

    event_labels = [f"{ev.date or '—'} — {ev.name}" for ev in events]
    idx = st.selectbox("Event auswählen", options=list(range(len(events))), format_func=lambda i: event_labels[i])
    selected_event = events[idx]
    cap_after = cap_tables[idx]
    valuation = valuations[idx]

    st.subheader(f"🧭 Event: {selected_event.name} ({selected_event.date or '—'})")
    col1, col2, col3, col4, col5 = st.columns(5)
    if selected_event.kind == "investment_round":
        col1.metric("Neues Kapital", money_fmt(valuation.get("new_money")))
        col2.metric("Neue Anteile", shares_fmt(valuation.get("new_shares")))
        col3.metric("Preis je Anteil", money_fmt(valuation.get("price_per_share")))
        col4.metric("Pre-Money", money_fmt(valuation.get("pre_money")))
        col5.metric("Post-Money", money_fmt(valuation.get("post_money")))
    else:
        col1.metric("Event-Typ", "VSP-Zuteilung")
        col2.metric("Zuteilungen", shares_fmt(sum(selected_event.vsp_issued.values()) if selected_event.vsp_issued else 0.0))
        col3.metric("Preis je Anteil", "–")
        col4.metric("Pre-Money", "–")
        col5.metric("Post-Money", "–")

    vsp_pool = cap_after.get("VSP Pool", 0.0)
    with st.expander("🎯 VSP-Pool (nach Event)"):
        st.write(f"**VSP-Pool Anteile:** {shares_fmt(vsp_pool)}")
        if selected_event.kind == "investment_round" and selected_event.shares_to_vsp:
            st.caption("Gründer-Übertrag in den VSP-Pool in dieser Runde:")
            st.dataframe(pd.DataFrame([{ "Holder": k, "Shares → VSP": v} for k, v in selected_event.shares_to_vsp.items()]))
        if selected_event.kind == "vsp_issue" and selected_event.vsp_issued:
            st.caption("VSP-Zuteilungen in diesem Event:")
            st.dataframe(pd.DataFrame([{ "Recipient": k, "VSP Granted (Shares)": v} for k, v in selected_event.vsp_issued.items()]))

    st.subheader("📈 Cap Table nach Event")
    df_cap = cap_table_dataframe(cap_after)
    st.dataframe(df_cap, use_container_width=True)
    csv_buf = io.StringIO(); df_cap.to_csv(csv_buf, index=False)
    st.download_button("CSV herunterladen", csv_buf.getvalue(), file_name=f"cap_table_after_{idx:02d}_{selected_event.name.replace(' ', '_')}.csv", mime="text/csv")

    st.subheader("🔎 Event-Details")
    df_detail = event_detail_dataframe(selected_event)
    if df_detail.empty:
        st.caption("Keine spezifischen Detaildaten für dieses Event.")
    else:
        st.dataframe(df_detail, use_container_width=True)

    with st.expander("⏱️ Eigentümerentwicklung über Zeit"):
        MAX_LABEL_CHARS = 28
        def _short(label: str) -> str:
            return label if len(label) <= MAX_LABEL_CHARS else label[: MAX_LABEL_CHARS - 1] + "…"
        owners = sorted({h for table in cap_tables for h in table.keys()})
        view_mode = st.radio("Einheit", ["%", "€"], horizontal=True)
        long_rows = []; short_labels_order: List[str] = []
        for i, table in enumerate(cap_tables):
            full_label = f"{events[i].date or '—'} — {events[i].name}"; short_label = _short(full_label); short_labels_order.append(short_label)
            total = sum(table.values()) or 1.0; price_i = price_history[i]
            for holder in owners:
                sh = table.get(holder, 0.0); pct = (sh / total) * 100.0; value = sh * price_i if price_i == price_i else float('nan')
                long_rows.append({"EventIndex": i + 1, "EventFull": full_label, "EventShort": short_label, "Holder": holder, "Ownership %": round(pct,4), "Wert (€)": value})
        df_long = pd.DataFrame(long_rows)
        chosen = st.multiselect("Akteure auswählen", owners, default=[o for o in owners if "VSP" not in o][:5])
        if chosen:
            plot_df = df_long[df_long["Holder"].isin(chosen)].copy()
            y_field, y_title = ("Ownership %", "Ownership %") if view_mode == "%" else ("Wert (€)", "Wert (€)")
            if view_mode == "€" and all((p != p) for p in price_history):
                st.info("Noch keine Bewertung verfügbar für die ausgewählten Events.")
            chart = alt.Chart(plot_df).mark_line(point=True).encode(
                x=alt.X("EventShort:N", sort=short_labels_order, title="Event", axis=alt.Axis(labelAngle=-25)),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color("Holder:N"),
                tooltip=["EventFull", "Holder", alt.Tooltip(f"{y_field}:Q", format=".2f")],
            ).properties(height=380)
            st.altair_chart(chart, use_container_width=True)
            if st.checkbox("Tabellarische Daten anzeigen"):
                show_cols = ["EventIndex", "EventFull", "EventShort", "Holder", y_field]; table_df = plot_df[show_cols].copy()
                if y_field == "Wert (€)":
                    table_df[y_field] = table_df[y_field].map(lambda x: money_fmt(x) if x == x else "–")
                st.dataframe(table_df, use_container_width=True)

def page_round_designer():  # condensed integration of round_designer_app
    st.title("🧩 Round Designer")
    st.markdown("Neue Runde hinzufügen ODER letzte Runde bearbeiten (inkl. optionalem VSP-Target & über-Pro‑Rata Rabatt).")
    # Determine last investment round index (if any)
    last_round_index = None
    for i in range(len(events) - 1, -1, -1):
        if events[i].kind == "investment_round":
            last_round_index = i
            break
    can_edit_last = last_round_index is not None
    edit_mode = False
    if can_edit_last:
        edit_mode = st.checkbox("Letzte Investment-Runde bearbeiten statt neue hinzufügen", value=False, help="Bearbeitet die zuletzt existierende Investment-Runde direkt (Name, Datum, Beträge, Discount, VSP-Target).")
    else:
        st.info("Noch keine Investment-Runde vorhanden – nur Hinzufügen möglich.")

    # replicate minimal logic from old file (shortened)
    vsp_pool_name = "VSP Pool"
    # classify holders for pro-rata & vsp-only (uses original raw events)
    cash_investors_hist, round_recipients_hist, vsp_grantees_hist = set(), set(), set()
    for ev_raw in raw_events:
        if ev_raw.get("kind") == "investment_round":
            cash_investors_hist.update([str(k).strip() for k in (ev_raw.get("amounts_invested") or {}).keys()])
            round_recipients_hist.update([str(k).strip() for k in (ev_raw.get("shares_received") or {}).keys()])
        elif ev_raw.get("kind") == "vsp_issue":
            vsp_grantees_hist.update([str(k).strip() for k in (ev_raw.get("vsp_received") or {}).keys()])
    vsp_only = set(h for h in vsp_grantees_hist if h not in cash_investors_hist and h not in round_recipients_hist)

    # Baseline (cap table before the designed/edited round)
    if edit_mode and can_edit_last:
        # shares before = cap table previous to last round
        if last_round_index > 0:
            baseline_cap_table = cap_tables[last_round_index - 1]
        else:
            baseline_cap_table = {}
        # event currently being edited
        editing_event = events[last_round_index]
    else:
        baseline_cap_table = cap_tables[-1] if cap_tables else {}
        editing_event = None

    total_shares_before = sum(baseline_cap_table.values()) if baseline_cap_table else 0.0
    current_vsp_shares_before = baseline_cap_table.get(vsp_pool_name, 0.0) if baseline_cap_table else 0.0

    def collect_investors(_raw: Dict[str, Any]) -> List[str]:
        names: set[str] = set()
        for ev in (_raw or {}).get("events", []):
            if ev.get("kind") == "investment_round":
                for inv in (ev.get("amounts_invested") or {}).keys():
                    names.add(str(inv).strip())
        return sorted(names)
    prev_investors = collect_investors(raw_data)
    existing_classes: List[Dict[str, Any]] = (raw_data.get("liquidation_terms", {}) or {}).get("classes", []) or []
    existing_class_names = [c.get("name") for c in existing_classes if c.get("name")]
    lp_options = existing_class_names + ["Neue Klasse definieren …"]

    # Prefill defaults (new round)
    default_name = "Series X"
    default_date = date.today()
    default_pre_money = 0.0
    default_vsp_target_pct = 0.0
    default_min_round = 0.0
    default_discount_pct = 0.0
    default_investor_rows = pd.DataFrame({"Investor": prev_investors if prev_investors else [""], "Investiert (€)": [0.0]*(len(prev_investors) if prev_investors else 1)})
    default_lp_selection = lp_options[0] if lp_options else "Neue Klasse definieren …"
    editing_original_round_name = None

    if edit_mode and editing_event is not None:
        # Need raw event dict for discount & shares_to_vsp
        # Find raw event (last investment_round) in raw_events list
        raw_last_round_idx = None
        for j in range(len(raw_events)-1, -1, -1):
            rj = raw_events[j]
            if rj.get("kind") == "investment_round":
                raw_last_round_idx = j
                break
        raw_last_round = raw_events[raw_last_round_idx] if raw_last_round_idx is not None else {}
        editing_original_round_name = raw_last_round.get("name")
        default_name = editing_event.name
        try:
            default_date = datetime.strptime(editing_event.date, "%Y-%m-%d").date() if editing_event.date else date.today()
        except Exception:
            default_date = date.today()
        # infer weighted price & pre-money
        total_invest = sum(editing_event.amounts_invested.values())
        total_new_shares = sum(editing_event.shares_received.values()) or 1.0
        weighted_price = total_invest / total_new_shares
        default_pre_money = weighted_price * total_shares_before if total_shares_before>0 else 0.0
        # reconstruct discount if stored
        disc = raw_last_round.get("discount") or {}
        default_min_round = float(disc.get("min_round_eur", 0.0))
        default_discount_pct = float(disc.get("discount_percent", 0.0)) * 100.0
        # attempt to deduce target vsp percent from cap table AFTER the round
        after_table = cap_tables[last_round_index]
        pool_after = after_table.get(vsp_pool_name, 0.0); total_after = sum(after_table.values()) or 0.0
        default_vsp_target_pct = (pool_after/total_after*100.0) if total_after>0 else 0.0
        # build invest rows from existing investors
        default_investor_rows = pd.DataFrame({"Investor": list(editing_event.amounts_invested.keys()), "Investiert (€)": list(editing_event.amounts_invested.values())})
        # figure out class selection (first class referencing the round)
        chosen_cls = None
        for c in existing_classes:
            if editing_event.name in (c.get("applies_to_round_names") or []):
                chosen_cls = c.get("name")
                break
        if chosen_cls:
            default_lp_selection = chosen_cls
        st.info(f"Bearbeite letzte Runde: {editing_event.name} (ursprünglicher Name wird in LP-Klassen aktualisiert falls geändert).")

    with st.form("new_round_form"):
        col_a, col_b, col_c = st.columns([2,1,1])
        with col_a:
            new_name = st.text_input("Rundenname", value=default_name)
        with col_b:
            new_date = st.date_input("Datum", value=default_date)
        with col_c:
            st.caption(f"Ausstehende Anteile vor Runde: **{shares_fmt(total_shares_before)}**")
            vsp_pct_before = (current_vsp_shares_before / total_shares_before * 100.0) if total_shares_before>0 else float('nan')
            st.caption("VSP Pool vor Runde: **" + shares_fmt(current_vsp_shares_before) + f"** ({vsp_pct_before:.2f}% )" if vsp_pct_before==vsp_pct_before else "–")
            vsp_target_percent = st.number_input("Ziel-VSP-Pool (% nach Runde)", min_value=0.0, max_value=50.0, value=default_vsp_target_pct, step=0.5, format="%f")
        pre_money_val = st.number_input("Pre-Money Bewertung (EUR)", min_value=0.0, value=default_pre_money, step=100000.0, format="%f")
        pps_preview = (pre_money_val / total_shares_before) if total_shares_before>0 else float('nan')
        st.write("Preis je Anteil (Vorschau):", money_fmt(pps_preview) if pps_preview==pps_preview else "–")
        st.markdown("**Investoren & Beträge**")
        invest_df = st.data_editor(default_investor_rows, num_rows="dynamic", key="round_designer_invest_editor")
        min_round_size_eur = st.number_input("Minimaler Rundengrößen-Schwellenwert (EUR)", min_value=0.0, value=default_min_round, step=100000.0, format="%f")
        discount_percent = st.number_input("Rabatt auf über‑Pro‑Rata‑Anteile (%)", min_value=0.0, value=default_discount_pct, step=0.5, format="%f")
        chosen_lp = st.selectbox("LP-Klasse wählen", lp_options, index=lp_options.index(default_lp_selection) if default_lp_selection in lp_options else 0)
        new_lp_def = None
        if chosen_lp.endswith("definieren …"):
            with st.expander("Neue LP-Klasse definieren", expanded=True):
                lp_name = st.text_input("Klassenname", value=new_name)
                lp_rate = st.number_input("Einfacher Zinssatz p.a.", min_value=0.0, value=0.06, step=0.005, format="%f")
                lp_cap = st.number_input("Cap Multiple gesamt (0 = uncapped)", min_value=0.0, value=0.0, step=0.1, format="%f")
                new_lp_def = {"name": lp_name.strip() or new_name, "simple_interest_rate": float(lp_rate), "cap_multiple_total": None if lp_cap==0 else float(lp_cap)}
        btn_label = "Runde aktualisieren" if edit_mode else "Runde berechnen & integrieren"
        submitted = st.form_submit_button(btn_label, type="primary")
    if not submitted:
        return
    # sanitize inputs
    amounts_invested: Dict[str, float] = {}
    if invest_df is not None and not invest_df.empty:
        for _, row in invest_df.iterrows():
            inv = str(row.get("Investor", "")).strip(); amt = _as_float(row.get("Investiert (€)") or 0.0)
            if inv and amt>0: amounts_invested[inv] = amt
    if total_shares_before<=0 or pre_money_val<=0 or sum(amounts_invested.values())<=0:
        st.error("Bitte gültige Eingaben (Pre-Money, Investitionen) machen."); return
    # eligible holders (exclude pool & vsp-only)
    eligible = {h: sh for h, sh in cap_tables[-1].items() if h != vsp_pool_name and h not in vsp_only and sh>0}
    eligible_total = sum(eligible.values())
    min_round_size = float(min_round_size_eur); discount_pct = float(discount_percent)/100.0
    pro_rata_eur = {}
    if min_round_size>0 and eligible_total>0:
        for h, sh in eligible.items(): pro_rata_eur[h] = min_round_size * (sh/eligible_total)
    invest_normal, invest_over = {}, {}
    for inv, amt in amounts_invested.items():
        ent = pro_rata_eur.get(inv, 0.0); normal = min(amt, ent); over = max(0.0, amt-normal); invest_normal[inv]=normal; invest_over[inv]=over
    pps = pre_money_val / total_shares_before; pps_disc = pps*(1-discount_pct) if discount_pct>0 else pps
    shares_received: Dict[str,float] = {}; eff_price: Dict[str,float] = {}
    for inv, amt in amounts_invested.items():
        n = invest_normal.get(inv,0.0); o = invest_over.get(inv,0.0)
        shares = (n/pps) + (o/pps_disc if discount_pct>0 else o/pps)
        shares_received[inv] = float(math.ceil(shares))
        eff_price[inv] = (n+o)/shares_received[inv] if shares_received[inv]>0 else float('nan')
    # VSP pool target
    t = float(vsp_target_percent)/100.0; shares_to_vsp: Dict[str,float]={}
    if 0 < t < 1:
        temp = dict(cap_tables[-1]);
        for h, sh in shares_received.items(): temp[h] = temp.get(h,0.0)+sh
        pool_after_invest = temp.get(vsp_pool_name,0.0); total_after_invest = sum(temp.values())
        target_pool = t*total_after_invest; needed = max(0.0, target_pool - pool_after_invest)
        eligible_donor_total = sum(sh for h, sh in temp.items() if h!=vsp_pool_name and h not in vsp_only)
        if needed>0 and eligible_donor_total>0:
            for h, sh in temp.items():
                if h==vsp_pool_name or h in vsp_only: continue
                take = needed * (sh/eligible_donor_total)
                if take>0: shares_to_vsp[h]=take
    new_event = {"kind":"investment_round","name":new_name.strip() or "Unnamed","date":new_date.strftime("%Y-%m-%d"),"amounts_invested":amounts_invested,"shares_received":shares_received,"shares_to_vsp":shares_to_vsp if t>0 else {},"discount":{"min_round_eur":min_round_size,"discount_percent":discount_percent,"investor_over_pro_rata_eur":invest_over}}
    # integrate (copy original json)
    new_json = json.loads(json.dumps(raw_data, ensure_ascii=False))
    new_json.setdefault("events", [])
    new_json.setdefault("liquidation_terms", {}).setdefault("classes", [])
    if edit_mode and can_edit_last:
        # replace last investment_round in events list
        replace_idx = None
        for j in range(len(new_json["events"]) - 1, -1, -1):
            if new_json["events"][j].get("kind") == "investment_round":
                replace_idx = j
                break
        if replace_idx is None:
            st.error("Konnte letzte Runde nicht finden (unerwartet). Abbruch."); return
        old_name_for_class = editing_original_round_name or new_json["events"][replace_idx].get("name")
        new_json["events"][replace_idx] = new_event
    else:
        new_json["events"].append(new_event)
        old_name_for_class = None

    # Update LP classes assignment
    if chosen_lp in existing_class_names and not chosen_lp.endswith("definieren …"):
        for c in new_json["liquidation_terms"]["classes"]:
            arr = c.setdefault("applies_to_round_names", [])
            if edit_mode and old_name_for_class and old_name_for_class in arr and c.get("name") != chosen_lp:
                # remove old name from other classes
                arr[:] = [n for n in arr if n != old_name_for_class]
            if c.get("name") == chosen_lp:
                if edit_mode and old_name_for_class and old_name_for_class in arr and old_name_for_class != new_event["name"]:
                    # rename in-place
                    arr[:] = [new_event["name"] if n == old_name_for_class else n for n in arr]
                if new_event["name"] not in arr:
                    arr.append(new_event["name"])
    else:
        if new_lp_def is None:
            st.error("Neue LP-Klasse unvollständig."); return
        # Remove old assignment if editing & old name existed
        if edit_mode and old_name_for_class:
            for c in new_json["liquidation_terms"]["classes"]:
                arr = c.get("applies_to_round_names") or []
                if old_name_for_class in arr:
                    c["applies_to_round_names"] = [n for n in arr if n != old_name_for_class]
        new_json["liquidation_terms"]["classes"].append({"name": new_lp_def["name"],"applies_to_round_names":[new_event["name"]],"simple_interest_rate": new_lp_def["simple_interest_rate"],"cap_multiple_total": new_lp_def["cap_multiple_total"]})
    # preview
    prev = pd.DataFrame([{ "Investor":k, "Investiert (€)":v, "Davon über Pro‑Rata (€)":invest_over.get(k,0.0), "Neue Anteile":shares_received[k], "Effektiver Preis/Anteil":eff_price.get(k,float('nan'))} for k,v in amounts_invested.items()])
    if not prev.empty:
        prev["Investiert (€)"] = prev["Investiert (€)"].map(money_fmt)
        prev["Davon über Pro‑Rata (€)"] = prev["Davon über Pro‑Rata (€)"].map(money_fmt)
        prev["Neue Anteile"] = prev["Neue Anteile"].map(shares_fmt)
        prev["Effektiver Preis/Anteil"] = prev["Effektiver Preis/Anteil"].map(lambda x: money_fmt(x) if x==x else "–")
    st.subheader("Vorschau neue Runde")
    st.dataframe(prev, use_container_width=True)
    if shares_to_vsp:
        transfer_df = pd.DataFrame([{ "Investor":h, "Transfer an VSP": shares_to_vsp[h]} for h in sorted(shares_to_vsp.keys())])
        transfer_df["Transfer an VSP"] = transfer_df["Transfer an VSP"].map(shares_fmt)
        st.dataframe(transfer_df, use_container_width=True)
    out_buf = io.StringIO(); json.dump(new_json, out_buf, ensure_ascii=False, indent=2)
    fname_prefix = "cap_table_updated_" if edit_mode else "cap_table_with_"
    st.download_button("Erweiterte JSON herunterladen", out_buf.getvalue(), file_name=f"{fname_prefix}{new_event['name'].replace(' ','_')}.json", mime="application/json")
    if edit_mode:
        st.success("Runde aktualisiert (nur Download, originale Session-Daten unverändert).")
    else:
        st.success("Neue Runde hinzugefügt (nur Download, originale Session-Daten unverändert).")

def page_exit_simulator():
    st.title("💸 Exit Simulator")
    st.markdown("Liquidation Preference & Participating Preferred Waterfall Simulation.")
    colx1, colx2 = st.columns([2,1])
    with colx1:
        exit_amount = st.number_input("Exit-Erlös (EUR)", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%f")
    with colx2:
        exit_date = st.date_input("Exit-Datum", value=date.today())
    if st.button("Simulation starten", type="primary"):
        final_cap = cap_tables[-1] if cap_tables else {}
        result = simulate_exit_proceeds(exit_amount, exit_date, final_cap, events, raw_data, liq_terms)
        lp_total = sum(result.get("payouts_lp", {}).values()); part_total = sum(result.get("payouts_participation", {}).values()); total_paid = lp_total + part_total
        if total_paid>0:
            df_ratio = pd.DataFrame([{"Komponente":"Liquidation Preference (LP)","Betrag":lp_total},{"Komponente":"Pro-rata Teilnahme","Betrag":part_total}])
            df_ratio["Anteil %"] = df_ratio["Betrag"].map(lambda x: round(100*x/total_paid,2)); df_ratio["Betrag (fmt)"] = df_ratio["Betrag"].map(money_fmt)
            st.markdown("**Verhältnis LP vs. Pro-rata Anteil**")
            pie = alt.Chart(df_ratio).mark_arc(outerRadius=120).encode(theta=alt.Theta("Betrag:Q"), color=alt.Color("Komponente:N"), tooltip=["Komponente","Betrag (fmt)","Anteil %"]).properties(width=320,height=320)
            _c1,_c2,_c3 = st.columns([1,2,1]);
            with _c2: st.altair_chart(pie, use_container_width=True)
        st.markdown("**LP je Investor & Runde**")
        if result.get("lp_by_tranche"):
            df_tr = pd.DataFrame(result["lp_by_tranche"]); pivot = df_tr.pivot_table(index="investor", columns="round_name", values="lp_paid", aggfunc="sum", fill_value=0.0).reset_index().rename(columns={"investor":"Holder"});
            round_cols = [c for c in pivot.columns if c!="Holder"]; pivot["Final"] = pivot[round_cols].sum(axis=1)
            display = pivot.copy();
            for c in round_cols+["Final"]: display[c] = display[c].map(money_fmt)
            st.dataframe(display, use_container_width=True)
        else:
            st.caption("Keine LP-Zuordnungen vorhanden.")
        st.markdown("**Teilnahme pro-rata**")
        if result["payouts_participation"]:
            df_part = pd.DataFrame([{ "Holder":k, "Teilnahme (€)":v} for k,v in sorted(result["payouts_participation"].items(), key=lambda kv: kv[1], reverse=True)])
            df_part["Teilnahme (€)"] = df_part["Teilnahme (€)"].map(money_fmt); st.dataframe(df_part, use_container_width=True)
        st.markdown("**Gesamt je Holder**")
        if result["totals"]:
            invested_by = compute_total_invested(raw_data); rows=[]
            for holder,total_recv in sorted(result["totals"].items(), key=lambda kv: kv[1], reverse=True):
                invested = invested_by.get(holder,0.0); multiple = (total_recv/invested) if invested>0 else float('nan')
                rows.append({"Holder":holder,"_Invested":invested,"_Total":total_recv,"_Multiple":multiple})
            df_tot = pd.DataFrame(rows); df_tot["Investiert (€)"] = df_tot["_Invested"].map(money_fmt); df_tot["Gesamt (€)"] = df_tot["_Total"].map(money_fmt); df_tot["Multiple (x)"] = df_tot["_Multiple"].map(lambda x: f"{x:.2f}x" if x==x else "–");
            st.dataframe(df_tot[["Holder","Investiert (€)","Gesamt (€)","Multiple (x)"]], use_container_width=True)
        if result["unallocated"]>1e-6: st.warning(f"Nicht zugeordnet (Rest): {money_fmt(result['unallocated'])}")
        else: st.success("Gesamterlös vollständig verteilt.")

if page == "Cap Table Explorer":
    page_cap_table_explorer()
elif page == "Round Designer":
    page_round_designer()
elif page == "Exit Simulator":
    page_exit_simulator()

st.caption("Hinweis: Bewertungen = neues Kapital / neue Anteile jeder Runde; Pre-Money = vorherige ausstehende Anteile * Anteilspreis.")
