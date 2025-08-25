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


def test_exit_waterfall_participating():
    events, cap_tables, _ = _prepare()
    liq_terms = extract_liquidation_terms(RAW_DATA)
    final_cap = cap_tables[-1]
    total_exit = 3_300_000.0
    result = simulate_exit_proceeds(total_exit, date(2025, 1, 1), final_cap, events, RAW_DATA, liq_terms)

    payouts = result["totals"]
    # Basic integrity: total distributed (plus any negligible remainder) ~ total_exit
    distributed = sum(payouts.values()) + result.get("unallocated", 0.0)
    assert abs(distributed - total_exit) < 1e-4

    # Expected LP amounts (1x each, no interest): Series A 200k, Seed 100k
    assert math.isclose(result["payouts_lp"].get("InvestorB", 0.0), 200_000, rel_tol=1e-6)
    assert math.isclose(result["payouts_lp"].get("InvestorA", 0.0), 100_000, rel_tol=1e-6)

    # Remaining proceeds 3,000,000 distributed pro-rata over 110,000 shares -> ~27.272727 per share
    per_share = 3_000_000 / 110_000
    assert math.isclose(per_share, 27.27272727, rel_tol=1e-6)

    # Founder share-based payout (no LP): 89,000 * per_share
    founder_participation = 89_000 * per_share
    assert math.isclose(result["payouts_participation"].get("Founder", 0.0), founder_participation, rel_tol=1e-6)

    # InvestorA total = LP + participation (10,000 * per_share)
    inv_a_total_expected = 100_000 + 10_000 * per_share
    assert math.isclose(payouts.get("InvestorA", 0.0), inv_a_total_expected, rel_tol=1e-6)

    # InvestorB total = LP + participation
    inv_b_total_expected = 200_000 + 10_000 * per_share
    assert math.isclose(payouts.get("InvestorB", 0.0), inv_b_total_expected, rel_tol=1e-6)

    # VSP Pool participation only (1,000 shares)
    vsp_expected = 1_000 * per_share
    assert math.isclose(payouts.get("VSP Pool", 0.0), vsp_expected, rel_tol=1e-6)

    # Class aggregation: ensure classes appear
    by_class = result["by_class"]
    assert "Series A Preferred" in by_class
    assert "Seed Preferred" in by_class
    # Founder should map to Common/Other bucket
    assert "Common/Other" in by_class


def test_exit_waterfall_capped_preference_voids_lp_when_threshold_exceeded():
    """New capped LP semantics: LP void if pro-rata share >= principal * cap_multiple.

    Scenario:
    - Founder: 90,000 common
    - Cap Round: InvestorCap 100k for 10,000 shares (10%). cap multiple 2x => threshold 200k.
    - Exit: 5,000,000 => pro-rata for investor = 500k (> 200k) so LP is void (gets only participation).
    """
    raw = {
        "events": [
            {"kind": "investment_round", "name": "Founding2", "date": "2023-01-01", "shares_received": {"Founder": 90_000}},
            {"kind": "investment_round", "name": "Cap Round", "date": "2023-06-01", "amounts_invested": {"InvestorCap": 100_000}, "shares_received": {"InvestorCap": 10_000}},
        ],
        "liquidation_terms": {"classes": [{"name": "Cap Preferred", "applies_to_round_names": ["Cap Round"], "simple_interest_rate": 0.0,"cap_multiple_total": 2.0}]},
    }
    events = [normalize_event(e) for e in raw["events"]]
    events_sorted = sorted(events, key=lambda e: e.date or "")
    events_norm, cap_tables = compute_cumulative_states(events_sorted)
    liq_terms = extract_liquidation_terms(raw)
    final_cap = cap_tables[-1]
    total_exit = 5_000_000.0
    result = simulate_exit_proceeds(total_exit, date(2025, 1, 1), final_cap, events_norm, raw, liq_terms)

    payouts = result["totals"]
    distributed = sum(payouts.values()) + result.get("unallocated", 0.0)
    assert abs(distributed - total_exit) < 1e-4
    # LP voided
    assert result["payouts_lp"].get("InvestorCap", 0.0) == 0.0
    # All proceeds distributed pro-rata: per share = 5,000,000 / 100,000 = 50
    per_share = 5_000_000 / 100_000
    assert math.isclose(per_share, 50.0, rel_tol=1e-6)
    assert math.isclose(result["payouts_participation"].get("InvestorCap", 0.0), 10_000 * 50.0, rel_tol=1e-6)
    assert math.isclose(payouts.get("Founder", 0.0), 90_000 * 50.0, rel_tol=1e-6)


def test_exit_waterfall_capped_preference_lp_applies_below_threshold():
    """When pro-rata share < threshold, LP applies then participation on remainder."""
    raw = {
        "events": [
            {"kind": "investment_round", "name": "Founding2", "date": "2023-01-01", "shares_received": {"Founder": 90_000}},
            {"kind": "investment_round", "name": "Cap Round", "date": "2024-01-01", "amounts_invested": {"InvestorCap": 100_000}, "shares_received": {"InvestorCap": 10_000}},
        ],
        "liquidation_terms": {"classes": [{"name": "Cap Preferred", "applies_to_round_names": ["Cap Round"], "simple_interest_rate": 1.0, "cap_multiple_total": 2.0}]},
    }
    events = [normalize_event(e) for e in raw["events"]]
    events_sorted = sorted(events, key=lambda e: e.date or "")
    events_norm, cap_tables = compute_cumulative_states(events_sorted)
    liq_terms = extract_liquidation_terms(raw)
    final_cap = cap_tables[-1]
    total_exit = 250_000.0  # pro-rata (10%) = 25k < threshold 200k => LP effective
    result = simulate_exit_proceeds(total_exit, date(2025, 1, 1), final_cap, events_norm, raw, liq_terms)
    # LP pays 100k first, remainder 150k shared => per share 1.5
    assert math.isclose(result["payouts_lp"].get("InvestorCap", 0.0), years_between("2024-01-01", date(2025, 1, 1)) * 100_000 + 100_000, rel_tol=1e-6)

    per_share = (total_exit - result["payouts_lp"].get("InvestorCap", 0.0)) / 100_000
    assert math.isclose(result["payouts_participation"].get("InvestorCap", 0.0), 10_000 * per_share, rel_tol=1e-6)
    assert math.isclose(result["totals"].get("Founder", 0.0), 90_000 * per_share, rel_tol=1e-6)
    assert math.isclose(result["totals"].get("InvestorCap", 0.0), result["payouts_lp"].get("InvestorCap", 0.0) + result["payouts_participation"].get("InvestorCap", 0.0), rel_tol=1e-6)


def test_exit_waterfall_capped_and_not_capped_lp():
    """When both capped and non-capped LPs are present, the correct LP is applied."""
    raw = {
        "events": [
            {"kind": "investment_round", "name": "Founding2", "date": "2023-01-01", "shares_received": {"Founder": 90_000}},
            {"kind": "investment_round", "name": "Non-Capped Round", "date": "2024-01-01", "amounts_invested": {"InvestorCap": 100_000}, "shares_received": {"InvestorCap": 10_000}},
            {"kind": "investment_round", "name": "Cap Round", "date": "2025-01-01", "amounts_invested": {"InvestorCap": 100_000}, "shares_received": {"InvestorCap": 10_000}},
        ],
        "liquidation_terms": {"classes": [{"name": "Cap Preferred", "applies_to_round_names": ["Cap Round"], "simple_interest_rate": .05, "cap_multiple_total": 2.0}, {"name": "Non-Cap", "applies_to_round_names": ["Non-Capped Round"], "simple_interest_rate": .005, "cap_multiple_total": 1.0}]}
    }
    
    events = [normalize_event(e) for e in raw["events"]]
    events_sorted = sorted(events, key=lambda e: e.date or "")
    events_norm, cap_tables = compute_cumulative_states(events_sorted)
    liq_terms = extract_liquidation_terms(raw)
    final_cap = cap_tables[-1]
    total_exit = 250_000.0  # pro-rata (10%) = 25k < threshold 200k => LP effective
    result = simulate_exit_proceeds(total_exit, date(2025, 1, 1), final_cap, events_norm, raw, liq_terms)
    


if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly for a quick sanity check
    import pytest  # type: ignore

    raise SystemExit(pytest.main([__file__, "-q"]))
