import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import io

import pandas as pd
import streamlit as st


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


def money_fmt(x: float, currency: str = "€") -> str:
    if x != x:  # NaN
        return "–"
    return f"{currency}{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")  # DE-style formatting


def shares_fmt(x: float) -> str:
    if x != x:
        return "–"
    # Show up to 4 decimals
    return f"{x:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
    raw_events = data.get("events", [])
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

# Sidebar selection
event_labels = [f"{ev.date or '—'} — {ev.name}" for ev in events]
idx = st.sidebar.selectbox("Event auswählen", options=list(range(len(events))), format_func=lambda i: event_labels[i])

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
    # Build owner list
    owners = sorted({h for table in cap_tables for h in table.keys()})
    records = []
    view_mode = st.radio("Einheit", ["%", "€"], horizontal=True)
    for i, table in enumerate(cap_tables):
        total = sum(table.values()) or 1.0
        price_i = price_history[i]
        for o in owners:
            shares_o = table.get(o, 0.0)
            pct = (shares_o / total) * 100.0
            value = shares_o * price_i if price_i == price_i else float('nan')
            records.append({
                "Event #": i + 1,
                "Event": f"{events[i].date or '—'} — {events[i].name}",
                "Holder": o,
                "Ownership %": round(pct, 4),
                "Wert (€)": value,
            })
    df_over_time = pd.DataFrame(records)
    chosen = st.multiselect("Akteure auswählen", owners, default=[o for o in owners if "VSP" not in o][:5])
    if chosen:
        subset = df_over_time[df_over_time["Holder"].isin(chosen)]
        if view_mode == "%":
            pivot = subset.pivot(index="Event #", columns="Holder", values="Ownership %").fillna(0.0)
            st.line_chart(pivot)
        else:
            if all((p != p) for p in price_history):
                st.info("Noch keine Bewertung verfügbar für die ausgewählten Events.")
            pivot_val = subset.pivot(index="Event #", columns="Holder", values="Wert (€)")
            st.line_chart(pivot_val)
        show_table = st.checkbox("Tabellarische Daten anzeigen")
        if show_table:
            if view_mode == "%":
                st.dataframe(subset[["Event #", "Event", "Holder", "Ownership %"]])
            else:
                fmt_subset = subset.copy()
                fmt_subset["Wert (€)"] = fmt_subset["Wert (€)"].map(lambda x: money_fmt(x) if x == x else "–")
                st.dataframe(fmt_subset[["Event #", "Event", "Holder", "Wert (€)"]])

st.divider()
st.caption("Hinweis: Bewertungen werden aus Rundendaten hergeleitet (impliziter Anteilspreis = neues Kapital / neue Anteile). Pre-Money basiert auf ausstehenden Anteilen * Anteilspreis vor Ausgabe. Bei VSP-Events gibt es keine neue Bewertung.")
