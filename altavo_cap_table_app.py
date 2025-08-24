import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import io

import pandas as pd
import streamlit as st
import altair as alt

from datetime import datetime, date
import math


# ------------------------------
# Data structures & utilities
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
    date = ev.get("date", "")
    rs = RoundSummary(kind=kind, name=name, date=date)
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
    """
    For each event, compute the cumulative share ownership AFTER applying the event.
    Also track a synthetic "VSP Pool" holder that receives founder transfers and pays out on vsp_issue.
    """
    cap_tables: List[Dict[str, float]] = []
    holders: Dict[str, float] = {}
    vsp_pool_name = "VSP Pool"

    for ev in events:
        if ev.kind == "investment_round":
            # Issue new shares to investors
            for holder, sh in ev.shares_received.items():
                holders[holder] = holders.get(holder, 0.0) + sh

            # If founders transfer shares into the VSP pool, move them (not new shares).
            if ev.shares_to_vsp:
                # Ensure pool exists
                holders[vsp_pool_name] = holders.get(vsp_pool_name, 0.0)
                for holder, sh in ev.shares_to_vsp.items():
                    holders[holder] = holders.get(holder, 0.0) - sh
                    holders[vsp_pool_name] += sh

        elif ev.kind == "vsp_issue":
            # Allocate from pool to individuals (no new shares created)
            if ev.vsp_issued:
                holders[vsp_pool_name] = holders.get(vsp_pool_name, 0.0)
                for holder, sh in ev.vsp_issued.items():
                    holders[holder] = holders.get(holder, 0.0) + sh
                    holders[vsp_pool_name] -= sh

        # snapshot after applying the event
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
        # Build investor-level details for this round
        rows = []
        # Combine keys from amounts and shares
        keys = set(ev.amounts_invested.keys()) | set(ev.shares_received.keys())
        for k in sorted(keys):
            rows.append({
                "Investor": k,
                "Invested": ev.amounts_invested.get(k, 0.0),
                "New Shares": ev.shares_received.get(k, 0.0)
            })
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
    """
    For each investment_round, derive price/share, pre-money, post-money.
    Pre-money = (total outstanding shares BEFORE the round) * (round price/share).
    Post-money = pre-money + new_money = (total shares AFTER the round) * price/share.
    For non-investment events, values remain NaN.
    """
    out = []
    prev_total_shares = 0.0
    for idx, ev in enumerate(events):
        totals = {
            "name": ev.name,
            "date": ev.date,
            "kind": ev.kind,
            "price_per_share": float("nan"),
            "pre_money": float("nan"),
            "post_money": float("nan"),
            "new_money": float("nan"),
            "new_shares": float("nan"),
        }
        if ev.kind == "investment_round":
            price = ev.price_per_share if ev.price_per_share == ev.price_per_share else float("nan")
            new_money = ev.new_money
            new_shares = ev.new_shares
            pre = prev_total_shares * price if price == price else float("nan")
            post = pre + new_money if (pre == pre and new_money == new_money) else float("nan")
            totals.update({
                "price_per_share": price,
                "pre_money": pre,
                "post_money": post,
                "new_money": new_money,
                "new_shares": new_shares,
            })
        # Update prev_total_shares to the AFTER state of this event
        prev_total_shares = sum(cap_tables[idx].values()) if idx < len(cap_tables) else prev_total_shares
        out.append(totals)
    return out


# ------------------------------
# Liquidation Preference & Exit Simulation
# ------------------------------

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
    """Return liquidation preference classes ordered by latest round date (newest first).

    Previous logic used an explicit ``processing_order`` integer (ascending). We now infer
    ordering automatically from the event chronology: classes whose rounds occur later
    should be processed earlier in the waterfall ("latest money first").

    Heuristics:
    - For each class, find all events whose names are listed in ``applies_to_round_names``.
    - Take the *latest* date among those events; if a round name is missing or has no date,
      it's ignored.
    - Classes are then sorted DESC by that latest date.
    - Classes with no resolvable date fall to the end (treated as very old).
    - ``processing_order`` is ignored (still accepted in input but not used for sorting).
    """
    terms = (raw or {}).get("liquidation_terms", {})
    classes = list(terms.get("classes", []))
    events = (raw or {}).get("events", [])

    # Build a map round_name -> parsed date
    round_dates: Dict[str, date] = {}
    for ev in events:
        rn = ev.get("name")
        d = ev.get("date")
        if rn and d:
            try:
                round_dates[rn] = _parse_date(d)
            except Exception:  # pragma: no cover (defensive)
                pass

    def latest_date_for_class(c: Dict[str, Any]) -> date:
        latest: date | None = None
        for rn in c.get("applies_to_round_names", []) or []:
            rd = round_dates.get(rn)
            if rd and (latest is None or rd > latest):
                latest = rd
        # If no date found, return a very old sentinel so it sorts last.
        return latest or date(1900, 1, 1)

    classes_sorted = sorted(classes, key=latest_date_for_class, reverse=True)
    return {"classes": classes_sorted}


def build_investment_tranches(events: List[RoundSummary], raw: Dict[str, Any], liq_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build per-investor investment tranches with round date, principal and class metadata.
    Only rounds whose names are mapped in liquidation_terms are considered preferred with LP. Others are treated as common (no LP).
    """
    name_to_class = {}
    for c in liq_terms.get("classes", []):
        for rn in c.get("applies_to_round_names", []):
            name_to_class[rn] = c

    tranches: List[Dict[str, Any]] = []
    # use original raw events (pre-normalization) to access amounts_invested per round
    raw_events = (raw or {}).get("events", [])
    for ev_raw in raw_events:
        if (ev_raw.get("kind") == "investment_round") and ev_raw.get("name") in name_to_class:
            c = name_to_class[ev_raw.get("name")]
            rdate = ev_raw.get("date")
            for investor, amt in (ev_raw.get("amounts_invested") or {}).items():
                tranches.append({
                    "investor": investor.strip(),
                    "round_name": ev_raw.get("name"),
                    "date": rdate,
                    "principal": _as_float(amt),
                    "class_name": c.get("name"),
                    "rate": float(c.get("simple_interest_rate")) if c.get("simple_interest_rate") is not None else 0.0,
                    "cap_multiple_total": c.get("cap_multiple_total"),  # may be None
                    "received": 0.0
                })
    # sort tranches by class processing order
    order_of_class = {c.get("name"): idx for idx, c in enumerate(liq_terms.get("classes", []))}
    tranches.sort(key=lambda t: order_of_class.get(t["class_name"], 9999))
    return tranches


def simulate_exit_proceeds(total_proceeds: float, exit_date: date, cap_table_after: Dict[str, float],
                           events: List[RoundSummary], raw_data: Dict[str, Any], liq_terms: Dict[str, Any]) -> Dict[str, Any]:
    """
     Participating preferred waterfall (updated logic):
     1. Compute each preferred tranche's liquidation preference (LP) as principal plus simple interest.
         If a cap_multiple_total is defined, it ONLY limits the LP amount itself (principal + interest), i.e. LP <= cap_multiple_total * principal.
     2. Pay all LPs in class processing order until proceeds exhausted.
     3. Distribute ALL remaining proceeds strictly pro-rata across all outstanding shares (no further caps; capped investors
         participate without limitation beyond their LP cap).
     This reflects a structure where a multiple limits only the preferential return, not the sum of LP + participation.
    """
    proceeds_left = max(0.0, float(total_proceeds))

    # Build LP tranches
    tranches = build_investment_tranches(events, raw_data, liq_terms)

    # Map shares per holder for participation phase
    shares_by_holder = {h: float(s) for h, s in cap_table_after.items()}
    total_shares = sum(shares_by_holder.values()) or 1.0

    payouts_lp: Dict[str, float] = {}
    payouts_participation: Dict[str, float] = {}
    lp_tranche_records: List[Dict[str, Any]] = []

    # Pre-compute accrued LP per tranche with simple non-cumulative interest
    for tr in tranches:
        yrs = years_between(tr["date"], exit_date)
        accrued = tr["principal"] * (1.0 + tr["rate"] * yrs)
        cap_total = tr["cap_multiple_total"]
        if cap_total is not None:
            accrued = min(accrued, cap_total * tr["principal"])  # LP portion bounded by total cap
        tr["lp_claim"] = max(0.0, accrued)

    # Phase 1: pay LPs by class order
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
                "lp_paid": pay
            })

    # Phase 2: all remaining proceeds distributed pro-rata by shares (no caps in participation)
    if proceeds_left > 0:
        if total_shares > 0:
            for h, s in shares_by_holder.items():
                if s <= 0:
                    continue
                amt = proceeds_left * (s / total_shares)
                if amt <= 0:
                    continue
                payouts_participation[h] = payouts_participation.get(h, 0.0) + amt
            proceeds_left = 0.0

    # Combine totals per holder
    totals: Dict[str, float] = {}
    for h, v in payouts_lp.items():
        totals[h] = totals.get(h, 0.0) + v
    for h, v in payouts_participation.items():
        totals[h] = totals.get(h, 0.0) + v

    # Numerical guardrails
    if abs((sum(payouts_lp.values()) + sum(payouts_participation.values()) + proceeds_left) - float(total_proceeds)) < 1e-6:
        pass  # OK
    else:
        # Clamp any tiny negative leftovers to zero
        if -1e-6 < proceeds_left < 0:
            proceeds_left = 0.0

    # Build summaries by class (kept for internal use)
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
        "lp_by_tranche": lp_tranche_records
    }


# ------------------------------
# Helper: Compute total invested per holder
# ------------------------------

def compute_total_invested(raw: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for ev in (raw or {}).get("events", []):
        if ev.get("kind") == "investment_round":
            for investor, amt in (ev.get("amounts_invested") or {}).items():
                k = investor.strip()
                totals[k] = totals.get(k, 0.0) + _as_float(amt)
    return totals


def money_fmt(x: float, currency: str = "€") -> str:
    if x != x:  # NaN
        return "–"
    return f"{currency}{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")  # DE-style formatting


def shares_fmt(x: float) -> str:
    if x != x:
        return "–"
    
    # Shares are integers
    return f"{int(x):,}".replace(",", ".")  # DE-style formatting


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Cap Table Explorer (GmbH)", page_icon="📊", layout="wide")
st.title("📊 Cap Table Explorer (GmbH) — Valuation, VSP & Ownership")

st.markdown(
    """
Lade eine JSON-Datei mit Cap-Table-Events hoch. Danach kannst du ein Event auswählen und:
- **Bewertung** (Preis/Aktie, Pre- & Post-Money),
- **VSP**-Bewegungen (Pool & Grants),
- **Eigentumsanteile (%)** je Person nach dem Event,
- **Investoren-Details** pro Finanzierungsrunde
analysieren.
"""
)

uploaded = st.file_uploader("JSON-Datei wählen", type=["json"])

example_note = st.expander("📎 Beispiel-Datenformat anzeigen")
with example_note:
    st.markdown(
        """
Die JSON sollte eine Liste von `events` enthalten. Unterstützte `kind`-Werte:
- `investment_round`: Felder `amounts_invested`, `shares_received`, optional `shares_to_vsp` (Übertrag in VSP-Pool).
- `vsp_issue`: Feld `vsp_received` (Zuteilung aus VSP-Pool an Personen).

Numerische Werte werden als **Anzahl Anteile** bzw. **Geldbeträge in EUR** interpretiert.
"""
    )

if uploaded is None:
    st.info("Bitte eine JSON-Datei hochladen, um fortzufahren.")
    st.stop()

# Parse file
try:
    data = json.load(uploaded)
    raw_data = data
    raw_events = data.get("events", [])
    liq_terms = extract_liquidation_terms(data)
except Exception as e:
    st.error(f"Fehler beim Lesen der Datei: {e}")
    st.stop()

# Normalize & compute states
events = [normalize_event(ev) for ev in raw_events]
events_sorted = sorted(events, key=lambda e: e.date or "")
events, cap_tables = compute_cumulative_states(events_sorted)
valuations = compute_valuations(events, cap_tables)

# Build a price/share history carrying forward the last known price for value calculations.
price_history: List[float] = []
last_price = float('nan')
for v in valuations:
    p = v.get("price_per_share", float('nan'))
    if p == p:  # not NaN
        last_price = p
    price_history.append(last_price)

# Haupt-Event Auswahl (aus Sidebar in Main Area verschoben)
event_labels = [f"{ev.date or '—'} — {ev.name}" for ev in events]
idx = st.selectbox(
    "Event auswählen",
    options=list(range(len(events))),
    format_func=lambda i: event_labels[i],
    key="event_select_main",
)

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

# VSP pool status AFTER event
vsp_pool = cap_after.get("VSP Pool", 0.0)
with st.expander("🎯 VSP-Pool (nach Event)"):
    st.write(f"**VSP-Pool Anteile:** {shares_fmt(vsp_pool)}")
    if selected_event.kind == "investment_round" and selected_event.shares_to_vsp:
        st.caption("Gründer-Übertrag in den VSP-Pool in dieser Runde:")
        st.dataframe(pd.DataFrame(
            [{"Holder": k, "Shares → VSP": v} for k, v in selected_event.shares_to_vsp.items()]
        ))
    if selected_event.kind == "vsp_issue" and selected_event.vsp_issued:
        st.caption("VSP-Zuteilungen in diesem Event:")
        st.dataframe(pd.DataFrame(
            [{"Recipient": k, "VSP Granted (Shares)": v} for k, v in selected_event.vsp_issued.items()]
        ))

# Cap table AFTER event
st.subheader("📈 Cap Table nach Event")
df_cap = cap_table_dataframe(cap_after)
st.dataframe(df_cap, use_container_width=True)

# Download CSV of cap table
csv_buf = io.StringIO()
df_cap.to_csv(csv_buf, index=False)
st.download_button("CSV herunterladen", csv_buf.getvalue(), file_name=f"cap_table_after_{idx:02d}_{selected_event.name.replace(' ', '_')}.csv", mime="text/csv")

# Event detail (per-round investors or VSP recipients)
st.subheader("🔎 Event-Details")
df_detail = event_detail_dataframe(selected_event)
if df_detail.empty:
    st.caption("Keine spezifischen Detaildaten für dieses Event.")
else:
    st.dataframe(df_detail, use_container_width=True)

# Ownership over time (optional view)
with st.expander("⏱️ Eigentümerentwicklung über Zeit (vereinfacht)"):
    # Konfiguration für Achsen-Beschriftungen
    MAX_LABEL_CHARS = 28  # feste maximale Länge für Eventnamen auf der X-Achse
    def _short(label: str) -> str:
        return label if len(label) <= MAX_LABEL_CHARS else label[: MAX_LABEL_CHARS - 1] + "…"

    owners = sorted({h for table in cap_tables for h in table.keys()})
    view_mode = st.radio("Einheit", ["%", "€"], horizontal=True)

    # Long-format Datensätze aufbauen inkl. gekürzter Event Labels
    long_rows = []
    short_labels_order: List[str] = []
    for i, table in enumerate(cap_tables):
        full_label = f"{events[i].date or '—'} — {events[i].name}"
        short_label = _short(full_label)
        short_labels_order.append(short_label)
        total = sum(table.values()) or 1.0
        price_i = price_history[i]
        for holder in owners:
            sh = table.get(holder, 0.0)
            pct = (sh / total) * 100.0
            value = sh * price_i if price_i == price_i else float('nan')
            long_rows.append({
                "EventIndex": i + 1,
                "EventFull": full_label,
                "EventShort": short_label,
                "Holder": holder,
                "Ownership %": round(pct, 4),
                "Wert (€)": value,
            })
    df_long = pd.DataFrame(long_rows)

    chosen = st.multiselect(
        "Akteure auswählen",
        owners,
        default=[o for o in owners if "VSP" not in o][:5],
    )
    if chosen:
        plot_df = df_long[df_long["Holder"].isin(chosen)].copy()
        if view_mode == "%":
            y_field = "Ownership %"
            y_title = "Ownership %"
        else:
            y_field = "Wert (€)"
            y_title = "Wert (€)"
            if all((p != p) for p in price_history):
                st.info("Noch keine Bewertung verfügbar für die ausgewählten Events.")

        # Altair Liniendiagramm mit benutzerdefinierten X-Ticks (gekürzte Eventnamen)
        domain_order = short_labels_order  # Reihenfolge der Ereignisse beibehalten
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "EventShort:N",
                    sort=domain_order,
                    title="Event",
                    axis=alt.Axis(labelOverlap=True, labelLimit=140, labelAngle=-25),
                ),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color("Holder:N", title="Holder"),
                tooltip=["EventFull", "Holder", alt.Tooltip(f"{y_field}:Q", format=".2f")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

        show_table = st.checkbox("Tabellarische Daten anzeigen")
        if show_table:
            show_cols = ["EventIndex", "EventFull", "EventShort", "Holder", y_field]
            table_df = plot_df[show_cols].copy()
            if y_field == "Wert (€)":
                table_df[y_field] = table_df[y_field].map(lambda x: money_fmt(x) if x == x else "–")
            st.dataframe(table_df, use_container_width=True)

# ------------------------------
# Exit Simulator UI
# ------------------------------
st.subheader("💸 Exit-Simulator")
with st.expander("Proceeds bei Exit simulieren", expanded=False):
    colx1, colx2 = st.columns([2,1])
    with colx1:
        exit_amount = st.number_input("Exit-Erlös (EUR)", min_value=0.0, value=10000000.0, step=100000.0, format="%f")
    with colx2:
        exit_date = st.date_input("Exit-Datum", value=date.today())

    if st.button("Simulation starten", type="primary"):
        final_cap = cap_tables[-1] if cap_tables else {}
        result = simulate_exit_proceeds(exit_amount, exit_date, final_cap, events, raw_data, liq_terms)

        # Pie chart: ratio of LP proceeds vs pro-rata participation
        lp_total = sum(result.get("payouts_lp", {}).values())
        part_total = sum(result.get("payouts_participation", {}).values())
        total_paid = lp_total + part_total
        if total_paid > 0:
            df_ratio = pd.DataFrame([
                {"Komponente": "Liquidation Preference (LP)", "Betrag": lp_total},
                {"Komponente": "Pro-rata Teilnahme", "Betrag": part_total},
            ])
            df_ratio["Anteil %"] = df_ratio["Betrag"].map(lambda x: round(100 * x / total_paid, 2))
            df_ratio["Betrag (fmt)"] = df_ratio["Betrag"].map(lambda x: money_fmt(x))
            st.markdown("**Verhältnis LP vs. Pro-rata Anteil**")
            pie = (
                alt.Chart(df_ratio)
                .mark_arc(outerRadius=120)
                .encode(
                    theta=alt.Theta(field="Betrag", type="quantitative"),
                    color=alt.Color(field="Komponente", type="nominal"),
                    tooltip=["Komponente", "Betrag (fmt)", "Anteil %"]
                )
                .properties(width=320, height=320)
            )
            _c1, _c2, _c3 = st.columns([1,2,1])
            with _c2:
                st.altair_chart(pie, use_container_width=True)
        else:
            st.info("Keine Auszahlungen zur Visualisierung (LP & Pro-rata sind 0).")

        st.markdown("**LP je Investor & Runde**")
        if result.get("lp_by_tranche"):
            # Build a DataFrame of LP paid per investor per round
            df_tr = pd.DataFrame(result["lp_by_tranche"])  # columns: investor, round_name, class_name, lp_claim, lp_paid
            # Ensure every investor-round combination exists (fill missing with 0)
            pivot = df_tr.pivot_table(index="investor", columns="round_name", values="lp_paid", aggfunc="sum", fill_value=0.0)
            pivot = pivot.reset_index().rename(columns={"investor": "Holder"})
            # Compute Final column as sum across round columns
            round_cols = [c for c in pivot.columns if c != "Holder"]
            pivot["Final"] = pivot[round_cols].sum(axis=1)

            # Order columns: rounds (by occurrence in data) then Final
            ordered_rounds = [rn for rn in df_tr["round_name"].unique().tolist() if rn in round_cols]
            pivot = pivot[["Holder"] + ordered_rounds + ["Final"]]

            # Format money columns for display
            display = pivot.copy()
            for c in ordered_rounds + ["Final"]:
                display[c] = display[c].map(lambda x: money_fmt(x))

            st.dataframe(display, use_container_width=True)
        else:
            st.caption("Keine LP-Zuordnungen für Investoren und Runden vorhanden.")

        st.markdown("**Teilnahme pro-rata**")
        if result["payouts_participation"]:
            df_part = pd.DataFrame([
                {"Holder": k, "Teilnahme (€)": v} for k, v in sorted(result["payouts_participation"].items(), key=lambda kv: kv[1], reverse=True)
            ])
            df_part["Teilnahme (€)"] = df_part["Teilnahme (€)"].map(lambda x: money_fmt(x))
            st.dataframe(df_part, use_container_width=True)

        st.markdown("**Gesamt je Holder**")
        if result["totals"]:
            invested_by = compute_total_invested(raw_data)
            rows = []
            for holder, total_recv in sorted(result["totals"].items(), key=lambda kv: kv[1], reverse=True):
                invested = invested_by.get(holder, 0.0)
                multiple = (total_recv / invested) if invested > 0 else float("nan")
                rows.append({
                    "Holder": holder,
                    "_Invested": invested,
                    "_Total": total_recv,
                    "_Multiple": multiple,
                })
            df_tot = pd.DataFrame(rows)
            # Display-friendly columns
            df_tot["Investiert (€)"] = df_tot["_Invested"].map(lambda x: money_fmt(x))
            df_tot["Gesamt (€)"] = df_tot["_Total"].map(lambda x: money_fmt(x))
            df_tot["Multiple (x)"] = df_tot["_Multiple"].map(lambda x: f"{x:.2f}x" if x == x else "–")
            st.dataframe(df_tot[["Holder", "Investiert (€)", "Gesamt (€)", "Multiple (x)"]], use_container_width=True)

        if result["unallocated"] > 1e-6:
            st.warning(f"Nicht zugeordnet (Rest): {money_fmt(result['unallocated'])}")
        else:
            st.success("Gesamterlös vollständig verteilt.")

st.divider()
st.caption("Hinweis: Bewertungen werden aus Rundendaten hergeleitet (impliziter Anteilspreis = neues Kapital / neue Anteile). Pre-Money basiert auf ausstehenden Anteilen * Anteilspreis vor Ausgabe. Bei VSP-Events gibt es keine neue Bewertung.")
