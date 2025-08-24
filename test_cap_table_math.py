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
from pathlib import Path

# ---------------------------------------------------------------------------
# Load only the pure logic section of the app (stop before Streamlit UI)
# ---------------------------------------------------------------------------
SRC_PATH = Path(__file__).parent / "altavo_cap_table_app.py"
source_lines = []
for line in SRC_PATH.read_text(encoding="utf-8").splitlines(True):
    # Break right before the first Streamlit config line
    if line.strip().startswith("st.set_page_config"):
        break
    source_lines.append(line)
logic_code = "".join(source_lines)
ns: dict[str, object] = {}
exec(logic_code, ns)  # noqa: S102 (intentional dynamic exec for test isolation)

# Extract needed symbols
normalize_event = ns["normalize_event"]
compute_cumulative_states = ns["compute_cumulative_states"]
compute_valuations = ns["compute_valuations"]
years_between = ns["years_between"]
simulate_exit_proceeds = ns["simulate_exit_proceeds"]
_as_float = ns["_as_float"]  # just in case


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
    liq_terms = ns["extract_liquidation_terms"](RAW_DATA)
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


def test_exit_waterfall_capped_preference():
    """Updated logic: cap limits ONLY LP (principal+interest), not total payout.

    Scenario:
    - Founder: 90,000 common shares (no LP).
    - Cap Round: InvestorCap invests 100k for 10,000 shares (price 10).
      Cap multiple 2.0 -> LP (principal + interest) capped at 200k (interest 0 so LP = 100k < 200k).
    - Exit: 5,000,000.

    Distribution:
      LP: 100,000 to InvestorCap.
      Remaining 4,900,000 shared over 100,000 shares => 49.0 per share.
      Founder: 90,000 * 49 = 4,410,000
      InvestorCap participation: 10,000 * 49 = 490,000
      Total InvestorCap = 590,000 (exceeds 2x principal because cap doesn't limit participation phase).
    """
    raw = {
        "events": [
            {"kind": "investment_round", "name": "Founding2", "date": "2023-01-01", "shares_received": {"Founder": 90_000}},
            {"kind": "investment_round", "name": "Cap Round", "date": "2023-06-01", "amounts_invested": {"InvestorCap": 100_000}, "shares_received": {"InvestorCap": 10_000}},
        ],
        "liquidation_terms": {
            "classes": [
                {
                    "name": "Cap Preferred",
                    "applies_to_round_names": ["Cap Round"],
                    "processing_order": 1,
                    "simple_interest_rate": 0.0,
                    "participating": True,
                    "cap_multiple_total": 2.0,
                }
            ]
        },
    }
    events = [normalize_event(e) for e in raw["events"]]
    events_sorted = sorted(events, key=lambda e: e.date or "")
    events_norm, cap_tables = compute_cumulative_states(events_sorted)
    liq_terms = ns["extract_liquidation_terms"](raw)
    final_cap = cap_tables[-1]
    total_exit = 5_000_000.0
    result = simulate_exit_proceeds(total_exit, date(2025, 1, 1), final_cap, events_norm, raw, liq_terms)

    payouts = result["totals"]
    # Integrity check
    distributed = sum(payouts.values()) + result.get("unallocated", 0.0)
    assert abs(distributed - total_exit) < 1e-4

    # LP remains 100k (within cap)
    lp_paid = result["payouts_lp"].get("InvestorCap", 0.0)
    assert math.isclose(lp_paid, 100_000.0, rel_tol=1e-6)
    # Participation per-share = 4.9M / 100k = 49
    per_share = 4_900_000.0 / 100_000.0
    assert math.isclose(per_share, 49.0, rel_tol=1e-6)
    part_paid = result["payouts_participation"].get("InvestorCap", 0.0)
    assert math.isclose(part_paid, 10_000 * 49.0, rel_tol=1e-6)
    # Total investor amount (uncapped participation)
    assert math.isclose(payouts.get("InvestorCap", 0.0), 100_000 + 490_000, rel_tol=1e-6)
    # Founder participation
    assert math.isclose(payouts.get("Founder", 0.0), 90_000 * 49.0, rel_tol=1e-6)


if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly for a quick sanity check
    import pytest  # type: ignore

    raise SystemExit(pytest.main([__file__, "-q"]))
