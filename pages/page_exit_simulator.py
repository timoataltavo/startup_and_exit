from __future__ import annotations
from typing import List, Dict, Any
from datetime import date
import pandas as pd
import altair as alt
import streamlit as st
from cap_table import RoundSummary, simulate_exit_proceeds, money_fmt, compute_total_invested


def render(events: List[RoundSummary], cap_tables: List[Dict[str, float]], raw_data: Dict[str, Any], liq_terms: Dict[str, Any]):
    st.title("💸 Exit Simulator")
    st.markdown("Liquidation Preference & Participating Preferred Waterfall Simulation.")
    colx1, colx2 = st.columns([2,1])
    with colx1:
        exit_amount = st.number_input("Exit-Erlös (EUR)", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%f")
    with colx2:
        exit_date = st.date_input("Exit-Datum", value=date.today())
    if st.button("Simulation starten", type="primary"):
        final_cap = cap_tables[-1] if cap_tables else {}
        result = simulate_exit_proceeds(exit_amount, exit_date, final_cap, raw_data, liq_terms)
        payouts_lp = result.get("payouts_lp", {}) or {}
        payouts_part = result.get("payouts_participation", {}) or {}
        # Aggregate LP & participation totals
        lp_total = sum(sum(r.values()) for r in payouts_lp.values())
        part_total = sum(payouts_part.values())
        total_paid = lp_total + part_total

        if total_paid > 0:
            df_ratio = pd.DataFrame([
                {"Komponente": "Liquidation Preference (LP)", "Betrag": lp_total},
                {"Komponente": "Pro-rata Teilnahme", "Betrag": part_total},
            ])
            df_ratio["Anteil %"] = df_ratio["Betrag"].map(lambda x: round(100 * x / total_paid, 2))
            df_ratio["Betrag (fmt)"] = df_ratio["Betrag"].map(money_fmt)
            st.markdown("**Verhältnis LP vs. Pro-rata Anteil**")
            pie = (
                alt.Chart(df_ratio)
                .mark_arc(outerRadius=120)
                .encode(
                    theta=alt.Theta("Betrag:Q"),
                    color=alt.Color("Komponente:N"),
                    tooltip=["Komponente", "Betrag (fmt)", "Anteil %"],
                )
                .properties(width=320, height=320)
            )
            _c1, _c2, _c3 = st.columns([1, 2, 1])
            with _c2:
                st.altair_chart(pie, use_container_width=True)

        # LP table (by investor & round)
        st.markdown("**LP je Investor & Runde**")
        if payouts_lp:
            rows = []
            for investor, per_round in payouts_lp.items():
                for round_name, amount in per_round.items():
                    rows.append({"Holder": investor, "Round": round_name, "LP (€)": amount})
            df_lp = pd.DataFrame(rows)
            pivot = (
                df_lp.pivot_table(
                    index="Holder", columns="Round", values="LP (€)", aggfunc="sum", fill_value=0.0
                )
                .reset_index()
            )
            round_cols = [c for c in pivot.columns if c != "Holder"]
            pivot["Final"] = pivot[round_cols].sum(axis=1)
            display = pivot.copy()
            for c in round_cols + ["Final"]:
                display[c] = display[c].map(money_fmt)
            st.dataframe(display, use_container_width=True)
        else:
            st.caption("Keine LP-Zuordnungen vorhanden.")

        # Participation table
        st.markdown("**Teilnahme pro-rata**")
        if payouts_part:
            df_part = pd.DataFrame(
                [
                    {"Holder": k, "Teilnahme (€)": v}
                    for k, v in sorted(payouts_part.items(), key=lambda kv: kv[1], reverse=True)
                ]
            )
            df_part["Teilnahme (€)"] = df_part["Teilnahme (€)"].map(money_fmt)
            st.dataframe(df_part, use_container_width=True)

        # Totals table
        st.markdown("**Gesamt je Holder**")
        if result.get("totals"):
            invested_by = compute_total_invested(raw_data)
            rows = []
            for holder, total_recv in sorted(result["totals"].items(), key=lambda kv: kv[1], reverse=True):
                invested = invested_by.get(holder, 0.0)
                multiple = (total_recv / invested) if invested > 0 else float("nan")
                rows.append(
                    {"Holder": holder, "_Invested": invested, "_Total": total_recv, "_Multiple": multiple}
                )
            df_tot = pd.DataFrame(rows)
            df_tot["Investiert (€)"] = df_tot["_Invested"].map(money_fmt)
            df_tot["Gesamt (€)"] = df_tot["_Total"].map(money_fmt)
            df_tot["Multiple (x)"] = df_tot["_Multiple"].map(lambda x: f"{x:.2f}x" if x == x else "–")
            st.dataframe(
                df_tot[["Holder", "Investiert (€)", "Gesamt (€)", "Multiple (x)"]],
                use_container_width=True,
            )

        if result.get("unallocated", 0.0) > 1e-6:
            st.warning(f"Nicht zugeordnet (Rest): {money_fmt(result['unallocated'])}")
        else:
            st.success("Gesamterlös vollständig verteilt.")
