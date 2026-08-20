"""
Generates a synthetic bank loan/credit portfolio for the AI Risk Analyst Agent demo.

This is intentionally deterministic (fixed seed) so the same "risk events" are
reproducible for demos and screenshots. Swap this out for a real data source
(a data warehouse query, a CSV export, a Kaggle credit-risk dataset, etc.)
by implementing the same output schema in src/data_loader.py.

Schema
------
loan_id            : str   - unique identifier
sector             : str   - borrower industry sector
region             : str   - booking region
exposure_at_default: float - EAD, USD
probability_default: float - PD, 0-1
loss_given_default : float - LGD, 0-1
duration_years      : float - interest-rate duration of the exposure
rate_type           : str   - "fixed" or "floating"
risk_rating          : str   - internal rating bucket, AAA..CCC
"""

import numpy as np
import pandas as pd

SECTORS = [
    "Commercial Real Estate", "Energy", "Manufacturing", "Retail & Consumer",
    "Technology", "Healthcare", "Financial Institutions", "Transportation",
]
REGIONS = ["APAC", "North America", "EMEA", "LATAM"]
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

# Rating -> baseline PD (annualized), roughly aligned with typical rating-agency bands
RATING_PD = {
    "AAA": 0.0002, "AA": 0.0006, "A": 0.0015, "BBB": 0.004,
    "BB": 0.014, "B": 0.045, "CCC": 0.12,
}


def generate_portfolio(n_loans: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ratings = rng.choice(RATINGS, size=n_loans, p=[0.03, 0.07, 0.15, 0.30, 0.25, 0.15, 0.05])
    base_pd = np.array([RATING_PD[r] for r in ratings])
    # add idiosyncratic noise so not every loan of the same rating is identical
    pd_noise = rng.normal(1.0, 0.15, size=n_loans).clip(0.5, 2.0)
    probability_default = (base_pd * pd_noise).clip(0.0001, 0.60)

    exposure_at_default = rng.lognormal(mean=14.5, sigma=1.1, size=n_loans).clip(50_000, 75_000_000)
    loss_given_default = rng.normal(0.45, 0.12, size=n_loans).clip(0.10, 0.95)
    duration_years = rng.gamma(shape=2.2, scale=1.6, size=n_loans).clip(0.1, 15)
    rate_type = rng.choice(["fixed", "floating"], size=n_loans, p=[0.55, 0.45])
    sector = rng.choice(SECTORS, size=n_loans)
    region = rng.choice(REGIONS, size=n_loans, p=[0.35, 0.30, 0.25, 0.10])

    df = pd.DataFrame({
        "loan_id": [f"LN-{100000 + i}" for i in range(n_loans)],
        "sector": sector,
        "region": region,
        "exposure_at_default": exposure_at_default.round(2),
        "probability_default": probability_default.round(5),
        "loss_given_default": loss_given_default.round(4),
        "duration_years": duration_years.round(2),
        "rate_type": rate_type,
        "risk_rating": ratings,
    })

    # Deliberately inject a handful of concentration / tail-risk scenarios so the
    # anomaly-detection agent has something real to catch in the demo.
    df = _inject_risk_events(df, rng)
    return df


def _inject_risk_events(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # 1) Sector concentration spike: push a cluster of large exposures into Energy
    energy_idx = df.sample(12, random_state=1).index
    df.loc[energy_idx, "sector"] = "Energy"
    df.loc[energy_idx, "exposure_at_default"] = rng.uniform(20_000_000, 60_000_000, size=12).round(2)

    # 2) A handful of large floating-rate, long-duration exposures (rate-shock sensitive)
    rate_idx = df.sample(8, random_state=2).index
    df.loc[rate_idx, "rate_type"] = "floating"
    df.loc[rate_idx, "duration_years"] = rng.uniform(9, 15, size=8).round(2)
    df.loc[rate_idx, "exposure_at_default"] = rng.uniform(15_000_000, 40_000_000, size=8).round(2)

    # 3) A few near-default names (high PD, weak rating) worth flagging individually
    tail_idx = df.sample(5, random_state=3).index
    df.loc[tail_idx, "risk_rating"] = "CCC"
    df.loc[tail_idx, "probability_default"] = rng.uniform(0.18, 0.35, size=5).round(5)

    return df


if __name__ == "__main__":
    portfolio = generate_portfolio()
    out_path = "data/loan_portfolio.csv"
    portfolio.to_csv(out_path, index=False)
    print(f"Wrote {len(portfolio)} loans to {out_path}")
