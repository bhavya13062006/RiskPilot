"""
Core risk-metric calculations shared by the agent graph.

Kept dependency-free (pandas/numpy only) and framework-agnostic so these
functions can be unit-tested in isolation from LangGraph/LLM concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConcentrationBreach:
    dimension: str          # "sector" or "region"
    name: str                # e.g. "Energy"
    exposure: float
    share_of_portfolio: float
    limit: float


@dataclass
class RiskLimits:
    """Illustrative internal risk-appetite limits (basis for anomaly detection)."""
    max_sector_concentration: float = 0.15   # any single sector <= 15% of EAD
    max_region_concentration: float = 0.40   # any single region  <= 40% of EAD
    max_single_name_pd: float = 0.15         # flag obligors with PD > 15%
    max_single_name_exposure: float = 25_000_000  # USD
    dv01_shock_bps: float = 100              # parallel curve shock used for DV01/EVE


@dataclass
class PortfolioRiskSummary:
    total_exposure: float
    expected_loss: float
    weighted_avg_pd: float
    weighted_avg_lgd: float
    portfolio_duration: float
    dv01: float                    # dollar value of a 1bp parallel rate move
    eve_sensitivity: float         # economic value of equity change under shock
    sector_concentration: pd.Series
    region_concentration: pd.Series
    concentration_breaches: list[ConcentrationBreach] = field(default_factory=list)
    single_name_flags: pd.DataFrame = field(default_factory=pd.DataFrame)


def expected_loss(df: pd.DataFrame) -> pd.Series:
    """EL = PD x LGD x EAD, per loan."""
    return df["probability_default"] * df["loss_given_default"] * df["exposure_at_default"]


def portfolio_duration(df: pd.DataFrame) -> float:
    """Exposure-weighted average duration, in years."""
    weights = df["exposure_at_default"] / df["exposure_at_default"].sum()
    return float((weights * df["duration_years"]).sum())


def dv01(df: pd.DataFrame) -> float:
    """
    Approximate portfolio DV01 (dollar value of a 1 basis-point rate move):
        DV01 ≈ Exposure * Duration * 0.0001
    Summed across all loans. Sign convention: positive DV01 = asset value falls
    as rates rise (long-duration asset book).
    """
    return float((df["exposure_at_default"] * df["duration_years"] * 0.0001).sum())


def eve_sensitivity(df: pd.DataFrame, shock_bps: float) -> float:
    """
    First-order estimate of the change in Economic Value of Equity (EVE) for a
    parallel rate shock of `shock_bps` basis points:
        delta_EVE ≈ -Duration * Exposure * delta_y
    Reported as a negative number = EVE loss under a rising-rate shock.
    """
    delta_y = shock_bps / 10_000
    return float(-(df["duration_years"] * df["exposure_at_default"] * delta_y).sum())


def concentration_by(df: pd.DataFrame, dimension: str) -> pd.Series:
    total = df["exposure_at_default"].sum()
    return (df.groupby(dimension)["exposure_at_default"].sum() / total).sort_values(ascending=False)


def find_concentration_breaches(df: pd.DataFrame, limits: RiskLimits) -> list[ConcentrationBreach]:
    breaches: list[ConcentrationBreach] = []
    total = df["exposure_at_default"].sum()

    sector_conc = concentration_by(df, "sector")
    for name, share in sector_conc.items():
        if share > limits.max_sector_concentration:
            breaches.append(ConcentrationBreach(
                dimension="sector", name=name,
                exposure=float(share * total), share_of_portfolio=float(share),
                limit=limits.max_sector_concentration,
            ))

    region_conc = concentration_by(df, "region")
    for name, share in region_conc.items():
        if share > limits.max_region_concentration:
            breaches.append(ConcentrationBreach(
                dimension="region", name=name,
                exposure=float(share * total), share_of_portfolio=float(share),
                limit=limits.max_region_concentration,
            ))

    return breaches


def find_single_name_flags(df: pd.DataFrame, limits: RiskLimits) -> pd.DataFrame:
    mask = (
        (df["probability_default"] > limits.max_single_name_pd)
        | (df["exposure_at_default"] > limits.max_single_name_exposure)
    )
    cols = ["loan_id", "sector", "region", "exposure_at_default",
            "probability_default", "risk_rating"]
    return df.loc[mask, cols].sort_values("probability_default", ascending=False)


def summarize_portfolio(df: pd.DataFrame, limits: RiskLimits | None = None) -> PortfolioRiskSummary:
    limits = limits or RiskLimits()

    el = expected_loss(df)
    total_exposure = float(df["exposure_at_default"].sum())
    weights = df["exposure_at_default"] / total_exposure

    return PortfolioRiskSummary(
        total_exposure=total_exposure,
        expected_loss=float(el.sum()),
        weighted_avg_pd=float((weights * df["probability_default"]).sum()),
        weighted_avg_lgd=float((weights * df["loss_given_default"]).sum()),
        portfolio_duration=portfolio_duration(df),
        dv01=dv01(df),
        eve_sensitivity=eve_sensitivity(df, limits.dv01_shock_bps),
        sector_concentration=concentration_by(df, "sector"),
        region_concentration=concentration_by(df, "region"),
        concentration_breaches=find_concentration_breaches(df, limits),
        single_name_flags=find_single_name_flags(df, limits),
    )
