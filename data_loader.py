"""
Loads the loan portfolio the agent will analyze.

Currently backed by the synthetic generator in data/generate_synthetic_data.py.
To point this at a real source, replace `load_portfolio` with a query against
your warehouse/API, as long as it returns a DataFrame with the same schema
documented in that file.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.generate_synthetic_data import generate_portfolio  # noqa: E402

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_portfolio.csv")


def load_portfolio(refresh: bool = False, n_loans: int = 500) -> pd.DataFrame:
    """
    Loads the loan portfolio, generating + caching it on first run so repeated
    demo runs are fast and consistent.
    """
    if refresh or not os.path.exists(_CACHE_PATH):
        df = generate_portfolio(n_loans=n_loans)
        df.to_csv(_CACHE_PATH, index=False)
        return df
    return pd.read_csv(_CACHE_PATH)
