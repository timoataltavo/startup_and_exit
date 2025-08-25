import json
from typing import Dict, List, Any

import streamlit as st

# Import modularized logic (must appear before first st.set_page_config for tests)
from cap_table import (
    normalize_event,
    compute_cumulative_states,
    compute_valuations,
    extract_liquidation_terms,
    simulate_exit_proceeds,  # re-exported for tests
    years_between,           # re-exported for tests
    _as_float,               # re-exported for tests
)

# Explicit re-export list for unit tests that exec this file pre-UI
__all__ = [
    'normalize_event',
    'compute_cumulative_states',
    'compute_valuations',
    'extract_liquidation_terms',
    'simulate_exit_proceeds',
    'years_between',
    '_as_float',
]

# Page modules (aliased to avoid name collision with wrapper functions)
from pages import (
    page_cap_table_explorer as page_cap_table_explorer_page,
    page_round_designer as page_round_designer_page,
    page_exit_simulator as page_exit_simulator_page,
    page_json_editor as page_json_editor_page,
)

st.set_page_config(page_title="Cap Table Toolkit (GmbH)", page_icon="🧮", layout="wide")

# Sidebar navigation
PAGES = ["Cap Table Explorer", "Round Designer", "Exit Simulator", "JSON Editor"]
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
    except (ValueError, TypeError) as e:  # pragma: no cover - UI error path
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
    page_cap_table_explorer_page.render(events, cap_tables, valuations, price_history)

def page_round_designer():
    page_round_designer_page.render(events, cap_tables, raw_events, raw_data)

def page_exit_simulator():
    page_exit_simulator_page.render(events, cap_tables, raw_data, liq_terms)
def page_json_editor():
    page_json_editor_page.render(raw_data)

if page == "Cap Table Explorer":
    page_cap_table_explorer()
elif page == "Round Designer":
    page_round_designer()
elif page == "Exit Simulator":
    page_exit_simulator()
elif page == "JSON Editor":
    page_json_editor()

st.caption("Hinweis: Bewertungen = neues Kapital / neue Anteile jeder Runde; Pre-Money = vorherige ausstehende Anteile * Anteilspreis.")
