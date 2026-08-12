import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.risk_metrics import (  # noqa: E402
    RiskLimits,
    concentration_by,
    dv01,
    eve_sensitivity,
    expected_loss,
    find_concentration_breaches,
    find_single_name_flags,
    portfolio_duration,
    summarize_portfolio,
)


@pytest.fixture
def toy_portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": ["A", "B", "C", "D"],
        "sector": ["Energy", "Energy", "Tech", "Healthcare"],
        "region": ["APAC", "APAC", "EMEA", "LATAM"],
        "exposure_at_default": [1_000_000, 1_000_000, 500_000, 500_000],
        "probability_default": [0.10, 0.02, 0.01, 0.30],
        "loss_given_default": [0.5, 0.5, 0.4, 0.6],
        "duration_years": [5.0, 5.0, 2.0, 1.0],
        "rate_type": ["fixed", "floating", "fixed", "fixed"],
        "risk_rating": ["B", "BBB", "A", "CCC"],
    })


def test_expected_loss(toy_portfolio):
    el = expected_loss(toy_portfolio)
    assert el.iloc[0] == pytest.approx(1_000_000 * 0.10 * 0.5)
    assert el.sum() > 0


def test_portfolio_duration_is_exposure_weighted(toy_portfolio):
    dur = portfolio_duration(toy_portfolio)
    # weighted avg should sit between the min and max individual durations
    assert 1.0 <= dur <= 5.0


def test_dv01_positive_for_positive_exposures(toy_portfolio):
    assert dv01(toy_portfolio) > 0


def test_eve_sensitivity_negative_under_rate_rise(toy_portfolio):
    # a rate *increase* should reduce EVE (negative sensitivity) for a long book
    assert eve_sensitivity(toy_portfolio, shock_bps=100) < 0


def test_concentration_by_sector_sums_to_one(toy_portfolio):
    conc = concentration_by(toy_portfolio, "sector")
    assert conc.sum() == pytest.approx(1.0)
    assert conc["Energy"] == pytest.approx(2_000_000 / 3_000_000)


def test_concentration_breach_detected(toy_portfolio):
    limits = RiskLimits(max_sector_concentration=0.15)
    breaches = find_concentration_breaches(toy_portfolio, limits)
    sectors_breached = {b.name for b in breaches if b.dimension == "sector"}
    assert "Energy" in sectors_breached  # Energy is 66.7% of book, well above 15%


def test_single_name_flags_catch_high_pd(toy_portfolio):
    limits = RiskLimits(max_single_name_pd=0.15, max_single_name_exposure=10_000_000)
    flags = find_single_name_flags(toy_portfolio, limits)
    assert "D" in flags["loan_id"].values  # PD=0.30 breaches the 0.15 limit
    assert "C" not in flags["loan_id"].values  # low PD, low exposure -> no flag


def test_summarize_portfolio_end_to_end(toy_portfolio):
    summary = summarize_portfolio(toy_portfolio)
    assert summary.total_exposure == pytest.approx(3_000_000)
    assert summary.expected_loss > 0
    assert 0 < summary.weighted_avg_pd < 1
    assert len(summary.concentration_breaches) >= 1
    assert not summary.single_name_flags.empty
