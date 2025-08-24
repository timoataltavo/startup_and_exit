# round_designer_app.py
import json
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from datetime import datetime, date
import math

import pandas as pd
import streamlit as st

# =========================
# Minimal helpers (copied/adapted)
# =========================

@dataclass
class RoundSummary:
    kind: str
    name: str
    date: str
    new_money: float = 0.0
    new_shares: float = 0.0
    price_per_share: float = float("nan")
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
    d = ev.get("date", "")
    rs = RoundSummary(kind=kind, name=name, date=d)
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
            for h, sh in ev.shares_received.items():
                holders[h] = holders.get(h, 0.0) + sh
            if ev.shares_to_vsp:
                holders[vsp_pool_name] = holders.get(vsp_pool_name, 0.0)
                for h, sh in ev.shares_to_vsp.items():
                    holders[h] = holders.get(h, 0.0) - sh
                    holders[vsp_pool_name] += sh
        elif ev.kind == "vsp_issue":
            if ev.vsp_issued:
                holders[vsp_pool_name] = holders.get(vsp_pool_name, 0.0)
                for h, sh in ev.vsp_issued.items():
                    holders[h] = holders.get(h, 0.0) + sh
                    holders[vsp_pool_name] -= sh
        cap_tables.append(dict(holders))
    return events, cap_tables

def money_fmt(x: float, currency: str = "€") -> str:
    if x != x:  # NaN
        return "–"
    return f"{currency}{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def shares_fmt(x: float) -> str:
    if x != x:
        return "–"
    return f"{int(x):,}".replace(",", ".")

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Round Designer (GmbH)", page_icon="🧩", layout="wide")
st.title("🧩 Investment-Runden-Designer")

st.markdown(
    """
Lade eine **bestehende JSON** mit Cap-Table-Events und Liquidation-Preferences.
Erstelle danach eine **neue Finanzierungsrunde**:
- **Name**, **Datum**, **Pre-Money**  
- **Investitionen** pro bestehendem Investor (neue Investoren beliebig hinzufügen)  
- **LP-Klasse** aus vorhandenen auswählen **oder neue definieren**  

Zum Schluss kannst du die **erweiterte JSON** herunterladen.
"""
)

uploaded = st.file_uploader("JSON-Datei wählen", type=["json"])

if not uploaded:
    st.info("Bitte JSON-Datei hochladen, um fortzufahren.")
    st.stop()

# Parse existing data
try:
    raw_data: Dict[str, Any] = json.load(uploaded)
except Exception as e:
    st.error(f"Fehler beim Lesen der Datei: {e}")
    st.stop()

raw_events = raw_data.get("events", [])
events = [normalize_event(ev) for ev in raw_events]
events_sorted = sorted(events, key=lambda e: e.date or "")

# Helper sets to classify holders
cash_investors_hist = set()
round_recipients_hist = set()
vsp_grantees_hist = set()
for ev_raw in raw_events:
    if ev_raw.get("kind") == "investment_round":
        cash_investors_hist.update([str(k).strip() for k in (ev_raw.get("amounts_invested") or {}).keys()])
        round_recipients_hist.update([str(k).strip() for k in (ev_raw.get("shares_received") or {}).keys()])
    elif ev_raw.get("kind") == "vsp_issue":
        vsp_grantees_hist.update([str(k).strip() for k in (ev_raw.get("vsp_received") or {}).keys()])
# VSP-only holders: received VSP but never invested cash nor received round shares
vsp_only_holders_hist = set(h for h in vsp_grantees_hist if h not in cash_investors_hist and h not in round_recipients_hist)

_, cap_tables = compute_cumulative_states(events_sorted)

# Outstanding shares BEFORE the new round (after last known event)
total_shares_before = sum(cap_tables[-1].values()) if cap_tables else 0.0

# Current VSP Pool BEFORE the round
vsp_pool_name = "VSP Pool"
current_vsp_shares_before = (cap_tables[-1].get(vsp_pool_name, 0.0) if cap_tables else 0.0)
current_vsp_percent_before = (current_vsp_shares_before / total_shares_before) if total_shares_before > 0 else float("nan")

# Collect previous investors (names appearing in amounts_invested across rounds)
def collect_investors(_raw: Dict[str, Any]) -> List[str]:
    names: set[str] = set()
    for ev in (_raw or {}).get("events", []):
        if ev.get("kind") == "investment_round":
            for inv in (ev.get("amounts_invested") or {}).keys():
                names.add(str(inv).strip())
    return sorted(names)

prev_investors = collect_investors(raw_data)

# LP classes
existing_classes: List[Dict[str, Any]] = (raw_data.get("liquidation_terms", {}) or {}).get("classes", []) or []
existing_class_names = [c.get("name") for c in existing_classes if c.get("name")]
lp_options = existing_class_names + ["Neue Klasse definieren …"]

# --------------------------
# Round form
# --------------------------
st.subheader("Runde konfigurieren")
with st.form("new_round_form"):
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        new_name = st.text_input("Rundenname", value="Series B")
    with col_b:
        new_date = st.date_input("Datum", value=date.today())
    with col_c:
        st.caption(f"Ausstehende Anteile vor Runde: **{shares_fmt(total_shares_before)}**")
        st.caption(
            "VSP Pool vor Runde: **" + shares_fmt(current_vsp_shares_before) + "** ("
            + (f"{current_vsp_percent_before*100:.2f}%" if current_vsp_percent_before == current_vsp_percent_before else "–")
            + ")"
        )
        
        # Target VSP pool size after the round (percentage of fully diluted shares post-invest)
        vsp_target_percent = st.number_input(
            "Ziel-VSP-Pool (% nach Runde)", min_value=0.0, max_value=50.0, value=float(0.0), step=0.5, format="%f",
            help="Prozentualer Zielanteil des VSP-Pools nach der Runde. Wird durch pro-rata Transfers von allen Nicht-Pool-Inhabern erreicht."
        )

    pre_money_val = st.number_input(
        "Pre-Money Bewertung (EUR)", min_value=0.0, value=0.0, step=100000.0, format="%f"
    )
    price_per_share_preview = (pre_money_val / total_shares_before) if total_shares_before > 0 else float("nan")
    st.write("Preis je Anteil (Vorschau): ", money_fmt(price_per_share_preview) if price_per_share_preview == price_per_share_preview else "–")

    # --- Optional Over-Pro-Rata Discount Settings ---
    st.markdown("**Optional: Über‑Pro‑Rata Rabatt**")
    min_round_size_eur = st.number_input(
        "Minimaler Rundengrößen-Schwellenwert (EUR)",
        min_value=0.0,
        value=0.0,
        step=100000.0,
        format="%f",
        help=(
            "Dient zur Berechnung der Pro‑Rata‑Ansprüche der bestehenden Anteilseigner. "
            "Investitionen über diese Pro‑Rata‑Anteile hinaus gelten als über‑Pro‑Rata."
        ),
    )
    discount_percent = st.number_input(
        "Rabatt auf über‑Pro‑Rata‑Anteile (%)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        format="%f",
        help=(
            "Prozentsatz, um den der Anteilspreis für über‑Pro‑Rata‑Anteile reduziert wird (z. B. 10%)."
        ),
    )

    st.markdown("**Investoren & Beträge** (Zeilen anpassen / neue Zeilen hinzufügen)")
    init_rows = pd.DataFrame({
        "Investor": prev_investors if prev_investors else [""],
        "Investiert (€)": [0.0] * (len(prev_investors) if prev_investors else 1),
    })
    invest_df = st.data_editor(
        init_rows,
        num_rows="dynamic",
        use_container_width=True,
        key="invest_table_editor_round_designer",
        column_config={
            "Investor": st.column_config.TextColumn("Investor"),
            "Investiert (€)": st.column_config.NumberColumn("Investiert (€)", min_value=0.0, step=1000.0, format="%f"),
        },
    )

    st.markdown("**Liquidation Preference (LP)**")
    chosen_lp = st.selectbox("LP-Klasse wählen", lp_options)

    new_lp_def = None
    if chosen_lp.endswith("definieren …"):
        with st.expander("Neue LP-Klasse definieren", expanded=True):
            lp_name = st.text_input("Klassenname", value="Series B")
            lp_rate = st.number_input("Einfacher Zinssatz p.a.", min_value=0.0, value=0.06, step=0.005, format="%f")
            lp_cap = st.number_input("Cap Multiple gesamt (inkl. LP & Participation) — 0 = uncapped", min_value=0.0, value=0.0, step=0.1, format="%f")
            lp_cap_use = None if lp_cap == 0.0 else float(lp_cap)
            new_lp_def = {
                "name": lp_name.strip() or "Series B",
                "simple_interest_rate": float(lp_rate),
                "cap_multiple_total": lp_cap_use,
            }

    submitted = st.form_submit_button("Runde berechnen & JSON erzeugen", type="primary")

if not submitted:
    st.stop()

# --------------------------
# Build new round + augmented JSON
# --------------------------
# Sanitize investor inputs
amounts_invested: Dict[str, float] = {}
if invest_df is not None and not invest_df.empty:
    for _, row in invest_df.iterrows():
        inv = str(row.get("Investor", "")).strip()
        amt = _as_float(row.get("Investiert (€)") or 0.0)
        if inv and amt > 0:
            amounts_invested[inv] = amt

new_money = sum(amounts_invested.values())

if total_shares_before <= 0:
    st.error("Keine ausstehenden Anteile gefunden. Bitte prüfen, ob die hochgeladene JSON gültige Events enthält.")
    st.stop()
if pre_money_val <= 0:
    st.error("Bitte eine positive Pre-Money Bewertung angeben.")
    st.stop()
if new_money <= 0:
    st.error("Bitte mindestens einen positiven Investitionsbetrag eingeben.")
    st.stop()

# Eligible holders for pro‑rata (alle Real‑Shareholder, exkl. VSP‑Pool und reine VSP‑Empfänger)
eligible_shares_before: Dict[str, float] = {}
for holder, sh in (cap_tables[-1].items() if cap_tables else []):
    pass
if cap_tables:
    for holder, sh in cap_tables[-1].items():
        if holder == vsp_pool_name:
            continue
        if holder in vsp_only_holders_hist:
            continue
        if sh > 0:
            eligible_shares_before[holder] = float(sh)
eligible_total_shares_before = sum(eligible_shares_before.values())

# --- Pro‑Rata & Über‑Pro‑Rata Berechnung ---
min_round_size = float(min_round_size_eur) if 'min_round_size_eur' in locals() else 0.0
discount_pct = max(0.0, min(float(discount_percent if 'discount_percent' in locals() else 0.0), 50.0)) / 100.0

pro_rata_eur: Dict[str, float] = {}
if min_round_size > 0.0 and eligible_total_shares_before > 0.0:
    for holder, sh in eligible_shares_before.items():
        pro_rata_eur[holder] = min_round_size * (sh / eligible_total_shares_before)

invest_normal_eur: Dict[str, float] = {}
invest_over_eur: Dict[str, float] = {}
for inv, amt in amounts_invested.items():
    entitlement = pro_rata_eur.get(inv, 0.0)
    normal_part = min(amt, entitlement)
    over_part = max(0.0, amt - normal_part)
    invest_normal_eur[inv] = normal_part
    invest_over_eur[inv] = over_part

pps = pre_money_val / total_shares_before
pps_discounted = pps * (1.0 - discount_pct) if discount_pct > 0.0 else pps

shares_received: Dict[str, float] = {}
effective_price_per_investor: Dict[str, float] = {}
for inv, amt in amounts_invested.items():
    normal_eur = invest_normal_eur.get(inv, 0.0)
    over_eur = invest_over_eur.get(inv, 0.0)
    if discount_pct <= 0.0 or min_round_size <= 0.0:
        shares = (normal_eur + over_eur) / pps
    else:
        shares = (normal_eur / pps) + (over_eur / pps_discounted if pps_discounted > 0 else 0.0)
    shares_received[inv] = float(math.ceil(shares))
    paid = normal_eur + over_eur
    effective_price_per_investor[inv] = (paid / shares_received[inv]) if shares_received[inv] > 0 else float('nan')

# Optional: pro-rata TRANSFER to VSP Pool to reach target percent (post-round)
t = float(vsp_target_percent) / 100.0 if 'vsp_target_percent' in locals() else 0.0
shares_to_vsp: Dict[str, float] = {}
vsp_pool_name = "VSP Pool"
if t > 0.0 and t < 1.0:
    # Build temporary post-round (pre-transfer) holdings
    temp_holdings: Dict[str, float] = dict(cap_tables[-1]) if cap_tables else {}
    for h, sh in shares_received.items():
        temp_holdings[h] = temp_holdings.get(h, 0.0) + sh
    current_pool_post_invest = float(temp_holdings.get(vsp_pool_name, 0.0))
    total_shares_after_invest = float(sum(temp_holdings.values()))
    target_pool_shares = t * total_shares_after_invest
    pool_transfer = max(0.0, target_pool_shares - current_pool_post_invest)
    # Pro-rata take from all non-pool holders EXCEPT VSP-only grantees
    eligible_donor_total = 0.0
    for holder, sh in temp_holdings.items():
        if holder == vsp_pool_name:
            continue
        if holder in vsp_only_holders_hist:
            continue
        eligible_donor_total += sh
    if pool_transfer > 0 and eligible_donor_total > 0:
        for holder, sh in temp_holdings.items():
            if holder == vsp_pool_name or holder in vsp_only_holders_hist or sh <= 0:
                continue
            take = pool_transfer * (sh / eligible_donor_total)
            if take > 0:
                shares_to_vsp[holder] = take

new_event = {
    "kind": "investment_round",
    "name": new_name.strip() or "Unnamed",
    "date": new_date.strftime("%Y-%m-%d"),
    "amounts_invested": amounts_invested,
    "shares_received": shares_received,
    "shares_to_vsp": shares_to_vsp if t > 0 else {},
    "discount": {
        "min_round_eur": float(min_round_size),
        "discount_percent": float(discount_pct * 100.0),
        "investor_over_pro_rata_eur": invest_over_eur,
    },
}

# Deep copy and append
new_json = json.loads(json.dumps(raw_data, ensure_ascii=False))
new_json.setdefault("events", []).append(new_event)

# LP mapping update
new_json.setdefault("liquidation_terms", {}).setdefault("classes", [])
if chosen_lp in existing_class_names:
    # append round name to existing class
    for c in new_json["liquidation_terms"]["classes"]:
        if c.get("name") == chosen_lp:
            arr = c.setdefault("applies_to_round_names", [])
            if new_event["name"] not in arr:
                arr.append(new_event["name"])
            break
else:
    # add new class definition
    if new_lp_def is None:
        st.error("Bitte die neue LP-Klasse vollständig ausfüllen.")
        st.stop()
    cdef = {
        "name": new_lp_def["name"],
        "applies_to_round_names": [new_event["name"]],
        "simple_interest_rate": new_lp_def["simple_interest_rate"],
        "cap_multiple_total": new_lp_def["cap_multiple_total"],
    }
    new_json["liquidation_terms"]["classes"].append(cdef)

# --------------------------
# Preview + Download
# --------------------------
st.subheader("Vorschau: neue Anteile pro Investor")
prev = pd.DataFrame([
    {
        "Investor": k,
        "Investiert (€)": v,
        "Davon über Pro‑Rata (€)": invest_over_eur.get(k, 0.0),
        "Neue Anteile": shares_received[k],
        "Effektiver Preis/Anteil": effective_price_per_investor.get(k, float('nan')),
    }
    for k, v in amounts_invested.items()
])
if not prev.empty:
    prev["Investiert (€)"] = prev["Investiert (€)"].map(lambda x: money_fmt(x))
    prev["Davon über Pro‑Rata (€)"] = prev["Davon über Pro‑Rata (€)"].map(lambda x: money_fmt(x))
    prev["Neue Anteile"] = prev["Neue Anteile"].map(lambda x: shares_fmt(x))
    prev["Effektiver Preis/Anteil"] = prev["Effektiver Preis/Anteil"].map(lambda x: money_fmt(x) if x == x else "–")

# Preview pro-rata transfers to VSP Pool
if shares_to_vsp:
    transfer_rows = [
        {"Investor": h, "Transfer an VSP": shares_to_vsp[h]}
        for h in sorted(shares_to_vsp.keys())
    ]
    transfer_df = pd.DataFrame(transfer_rows)
    # compute expected post-round pool % (transfers don't change total shares)
    total_after = total_shares_before + sum(shares_received.values())
    vsp_after = 0.0
    current_vsp_shares_before = cap_tables[-1].get(vsp_pool_name, 0.0) if cap_tables else 0.0
    vsp_after = current_vsp_shares_before + sum(shares_to_vsp.values())
    vsp_after_pct = (vsp_after / total_after) if total_after > 0 else float('nan')
    st.info(
        "VSP Pool pro-rata Transfer (nur reale Anteilseigner): "
        + f"**{shares_fmt(int(round(sum(shares_to_vsp.values()))))}** Anteile → Ziel ca. "
        + (f"{vsp_after_pct*100:.2f}%" if vsp_after_pct == vsp_after_pct else "–")
    )
    st.dataframe(
        transfer_df.assign(**{"Transfer an VSP": transfer_df["Transfer an VSP"].map(lambda x: shares_fmt(x))}),
        use_container_width=True,
    )

st.dataframe(prev, use_container_width=True)

out_buf = io.StringIO()
json.dump(new_json, out_buf, ensure_ascii=False, indent=2)
st.download_button(
    label="Erweiterte JSON herunterladen",
    data=out_buf.getvalue(),
    file_name=f"cap_table_with_{(new_event['name']).replace(' ', '_')}.json",
    mime="application/json",
)

st.success("Neue Runde wurde in die JSON integriert. Du kannst sie jetzt herunterladen.")
st.caption(
    "Hinweis: Preis je Anteil = Pre-Money / ausstehende Anteile vor Runde. Neue Anteile = Investition / Preis je Anteil. "
    "Falls eine Zielgröße für den VSP-Pool angegeben ist, werden Anteile **pro-rata** von allen Nicht-Pool-Inhabern an den VSP-Pool übertragen, "
    "so dass der Ziel-Prozentsatz *nach* der Runde erreicht wird (keine neuen Anteile, rein umverteilend)."
    " Zusätzliche Funktion: Für Investitionen über die pro‑rata Zuteilung (bezogen auf die minimale Rundengröße) wird ein konfigurierbarer Rabatt auf den Anteilspreis angewandt."
)