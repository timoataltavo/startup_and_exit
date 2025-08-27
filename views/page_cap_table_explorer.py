from __future__ import annotations
from typing import List, Dict
import io
import pandas as pd
import altair as alt
import streamlit as st
from cap_table import RoundSummary, money_fmt, shares_fmt
from cap_table.events import cap_table_dataframe, event_detail_dataframe


def render(events: List[RoundSummary], cap_tables: List[Dict[str, float]], valuations: List[Dict[str, float]], price_history: List[float]):
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
                    from cap_table import money_fmt as _mf
                    table_df[y_field] = table_df[y_field].map(lambda x: _mf(x) if x == x else "–")
                st.dataframe(table_df, use_container_width=True)
