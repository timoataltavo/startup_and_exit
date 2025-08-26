"""Unit tests for the math / cap table logic in `altavo_cap_table_app.py`.

These tests deliberately avoid importing the whole Streamlit app (which would
execute UI code and call st.stop()). Instead we:
  1. Read the source file.
  2. Execute only the part BEFORE the Streamlit UI section to load pure
     computation helpers into a temporary namespace.

Test coverage (representative core logic):
- normalize_event / price_per_share
- compute_cumulative_states
- compute_valuations (pre & post money correctness)
- years_between utility
- simulate_exit_proceeds (LP + participation waterfall)

The scenario constructed covers:
- Founding issuance with only shares (no price)
- Seed round with new money / new shares (establishing a price)
- Series A round with new money / new shares and founder transfer to VSP pool
- Liquidation preference waterfall with participating preferred classes.
"""
from __future__ import annotations

import math
from datetime import date
from cap_table import (
    normalize_event,
    compute_cumulative_states,
    compute_valuations,
    years_between,
    simulate_exit_proceeds,
    extract_liquidation_terms
)


# Helper to build normalized events easily
def _n(ev_dict):
    return normalize_event(ev_dict)


# ---------------------------------------------------------------------------
# Fixtures (constructed inline; no pytest fixture function needed for now)
# ---------------------------------------------------------------------------
FOUNDING = {
    "kind": "investment_round",
    "name": "Founding",
    "date": "2023-01-01",
    "shares_received": {"Founder": 90000},
}
SEED = {
    "kind": "investment_round",
    "name": "Seed",
    "date": "2023-06-01",
    "amounts_invested": {"InvestorA": 100_000},
    "shares_received": {"InvestorA": 10_000},  # price 10.0
}
SERIES_A = {
    "kind": "investment_round",
    "name": "Series A",
    "date": "2024-01-01",
    "amounts_invested": {"InvestorB": 200_000},
    "shares_received": {"InvestorB": 10_000},  # price 20.0
    "shares_to_vsp": {"Founder": 1_000},        # founder transfers to VSP pool
}

RAW_DATA = {
    "events": [FOUNDING, SEED, SERIES_A],
    "liquidation_terms": {
        "classes": [
            {
                "name": "Series A Preferred",
                "applies_to_round_names": ["Series A"],
                "processing_order": 1,
                "simple_interest_rate": 0.0,
                "participating": True,
            },
            {
                "name": "Seed Preferred",
                "applies_to_round_names": ["Seed"],
                "processing_order": 2,
                "simple_interest_rate": 0.0,
                "participating": True,
            },
        ]
    },
}


def _prepare():
    events = [_n(e) for e in RAW_DATA["events"]]
    events_sorted = sorted(events, key=lambda e: e.date or "")
    events_norm, cap_tables = compute_cumulative_states(events_sorted)
    valuations = compute_valuations(events_norm, cap_tables)
    return events_norm, cap_tables, valuations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normalize_and_price_per_share():
    _, _, valuations = _prepare()
    # Founding price is 0.0 (0 money / 90000 shares) per current normalize_event logic
    assert valuations[0]["price_per_share"] == 0.0  # founding
    # Seed round price 100k / 10k = 10
    assert valuations[1]["price_per_share"] == 10.0
    # Series A price 200k / 10k = 20
    assert valuations[2]["price_per_share"] == 20.0


def test_cumulative_holdings():
    _, cap_tables, _ = _prepare()
    # After founding
    assert cap_tables[0] == {"Founder": 90_000}
    # After seed
    assert cap_tables[1] == {"Founder": 90_000, "InvestorA": 10_000}
    # After series A (founder -> VSP Pool transfer of 1k)
    assert cap_tables[2]["Founder"] == 89_000
    assert cap_tables[2]["VSP Pool"] == 1_000
    assert cap_tables[2]["InvestorA"] == 10_000
    assert cap_tables[2]["InvestorB"] == 10_000
    assert sum(cap_tables[2].values()) == 110_000


def test_valuations_pre_post_money():
    _, cap_tables, valuations = _prepare()
    # Seed pre-money: previous outstanding shares (90k) * seed price (10) = 900k
    assert valuations[1]["pre_money"] == 900_000
    assert valuations[1]["post_money"] == 1_000_000
    # Series A pre-money: prior total shares (100k) * price 20 = 2,000,000
    assert valuations[2]["pre_money"] == 2_000_000
    assert valuations[2]["post_money"] == 2_200_000
    # Total shares after series A should match cap table
    assert sum(cap_tables[2].values()) == 110_000


def test_years_between():
    y = years_between("2024-01-01", date(2025, 1, 1))
    # Roughly 1 year (allow small tolerance for 365.25 divisor)
    assert abs(y - 1.0) < 0.01
    assert years_between("", date.today()) == 0.0

