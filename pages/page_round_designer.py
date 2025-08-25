from __future__ import annotations
from typing import List, Dict, Any
from datetime import date, datetime
import math
import json
import pandas as pd
import streamlit as st
from cap_table import RoundSummary, shares_fmt, money_fmt, _as_float


def render(events: List[RoundSummary], cap_tables: List[Dict[str, float]], raw_events: List[Dict[str, Any]], raw_data: Dict[str, Any]):
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
        if last_round_index > 0:
            baseline_cap_table = cap_tables[last_round_index - 1]
        else:
            baseline_cap_table = {}
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
    lp_options = existing_class_names + ["Neue Klasse definieren …"] if existing_class_names else ["Neue Klasse definieren …"]

    # Prefill defaults
    default_name = "Series X"
    default_date = date.today()
    default_pre_money = 0.0
    default_vsp_target_pct = 0.0
    default_min_round = 0.0
    default_discount_pct = 0.0
    default_investor_rows = pd.DataFrame({"Investor": prev_investors if prev_investors else [""], "Investiert (€)": [0.0]*(len(prev_investors) if prev_investors else 1)})
    default_lp_selection = lp_options[0]
    editing_original_round_name = None

    if edit_mode and editing_event is not None:
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
        total_invest = sum(editing_event.amounts_invested.values())
        total_new_shares = sum(editing_event.shares_received.values()) or 1.0
        weighted_price = total_invest / total_new_shares
        default_pre_money = weighted_price * total_shares_before if total_shares_before>0 else 0.0
        disc = raw_last_round.get("discount") or {}
        default_min_round = float(disc.get("min_round_eur", 0.0))
        default_discount_pct = float(disc.get("discount_percent", 0.0)) * 100.0
        after_table = cap_tables[last_round_index]
        pool_after = after_table.get(vsp_pool_name, 0.0); total_after = sum(after_table.values()) or 0.0
        default_vsp_target_pct = (pool_after/total_after*100.0) if total_after>0 else 0.0
        default_investor_rows = pd.DataFrame({"Investor": list(editing_event.amounts_invested.keys()), "Investiert (€)": list(editing_event.amounts_invested.values())})
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

    amounts_invested: Dict[str, float] = {}
    if invest_df is not None and not invest_df.empty:
        for _, row in invest_df.iterrows():
            inv = str(row.get("Investor", "")).strip(); amt = _as_float(row.get("Investiert (€)") or 0.0)
            if inv and amt>0: amounts_invested[inv] = amt
    if total_shares_before<=0 or pre_money_val<=0 or sum(amounts_invested.values())<=0:
        st.error("Bitte gültige Eingaben (Pre-Money, Investitionen) machen."); return

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

    new_json = json.loads(json.dumps(raw_data, ensure_ascii=False))
    new_json.setdefault("events", [])
    new_json.setdefault("liquidation_terms", {}).setdefault("classes", [])
    if edit_mode and can_edit_last:
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

    existing_class_names = [c.get("name") for c in existing_classes if c.get("name")]
    if chosen_lp in existing_class_names and not chosen_lp.endswith("definieren …"):
        for c in new_json["liquidation_terms"]["classes"]:
            arr = c.setdefault("applies_to_round_names", [])
            if edit_mode and old_name_for_class and old_name_for_class in arr and c.get("name") != chosen_lp:
                arr[:] = [n for n in arr if n != old_name_for_class]
            if c.get("name") == chosen_lp:
                if edit_mode and old_name_for_class and old_name_for_class in arr and old_name_for_class != new_event["name"]:
                    arr[:] = [new_event["name"] if n == old_name_for_class else n for n in arr]
                if new_event["name"] not in arr:
                    arr.append(new_event["name"])
    else:
        if new_lp_def is None:
            st.error("Neue LP-Klasse unvollständig."); return
        if edit_mode and old_name_for_class:
            for c in new_json["liquidation_terms"]["classes"]:
                arr = c.get("applies_to_round_names") or []
                if old_name_for_class in arr:
                    c["applies_to_round_names"] = [n for n in arr if n != old_name_for_class]
        new_json["liquidation_terms"]["classes"].append({"name": new_lp_def["name"],"applies_to_round_names":[new_event["name"]],"simple_interest_rate": new_lp_def["simple_interest_rate"],"cap_multiple_total": new_lp_def["cap_multiple_total"]})

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
    out_buf = json.dumps(new_json, ensure_ascii=False, indent=2)
    fname_prefix = "cap_table_updated_" if edit_mode else "cap_table_with_"
    st.download_button("Erweiterte JSON herunterladen", out_buf, file_name=f"{fname_prefix}{new_event['name'].replace(' ','_')}.json", mime="application/json")
    if edit_mode:
        st.success("Runde aktualisiert (nur Download, originale Session-Daten unverändert).")
    else:
        st.success("Neue Runde hinzugefügt (nur Download, originale Session-Daten unverändert).")
