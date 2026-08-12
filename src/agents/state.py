"""
Shared state passed between nodes in the risk-analyst agent graph.

LangGraph threads a single state object through every node; each node reads
what it needs and returns a partial update that gets merged in. Keeping this
as an explicit, typed schema (rather than a free-form dict) makes the graph
easy to reason about and easy to unit-test node-by-node.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class RiskAgentState(TypedDict, total=False):
    # --- inputs ---
    n_loans: int                       # how many synthetic loans to load/generate

    # --- populated by data_retrieval node ---
    portfolio: pd.DataFrame            # raw loan-level data

    # --- populated by risk_calculation node ---
    risk_summary: Any                  # PortfolioRiskSummary (see src/risk_metrics.py)

    # --- populated by anomaly_detection node ---
    flags: list[str]                   # human-readable list of breaches/anomalies
    severity: str                      # "low" | "medium" | "high"

    # --- populated by summarization node ---
    memo: str                          # final natural-language risk memo
    used_llm: bool                     # True if an LLM generated the memo, else rule-based

    # --- trace, for UI transparency ---
    trace: list[str]                   # ordered log of what each node did
