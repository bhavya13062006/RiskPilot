"""
Node implementations for the risk-analyst agent graph.

Each function takes the current RiskAgentState and returns a partial state
update, per the LangGraph convention. Nodes are kept intentionally small and
side-effect-free (aside from the LLM call in `summarize`) so they're easy to
test individually — see tests/test_risk_metrics.py for the pure-calculation
layer these nodes build on.
"""

from __future__ import annotations

from src.data_loader import load_portfolio
from src.llm_config import get_llm
from src.risk_metrics import RiskLimits, summarize_portfolio

from .state import RiskAgentState

LIMITS = RiskLimits()


def retrieve_data(state: RiskAgentState) -> dict:
    """Loads the loan portfolio (real data source or cached synthetic set)."""
    n_loans = state.get("n_loans", 500)
    portfolio = load_portfolio(n_loans=n_loans)
    trace = state.get("trace", [])
    trace.append(f"Data Retrieval: loaded {len(portfolio)} loans "
                 f"across {portfolio['sector'].nunique()} sectors, "
                 f"{portfolio['region'].nunique()} regions.")
    return {"portfolio": portfolio, "trace": trace}


def calculate_risk(state: RiskAgentState) -> dict:
    """Computes portfolio-level credit and interest-rate risk metrics."""
    portfolio = state["portfolio"]
    summary = summarize_portfolio(portfolio, LIMITS)

    trace = state.get("trace", [])
    trace.append(
        "Risk Calculation: "
        f"EAD=${summary.total_exposure:,.0f}, "
        f"Expected Loss=${summary.expected_loss:,.0f}, "
        f"Duration={summary.portfolio_duration:.2f}y, "
        f"DV01=${summary.dv01:,.0f}, "
        f"EVE @ {LIMITS.dv01_shock_bps:.0f}bp shock=${summary.eve_sensitivity:,.0f}."
    )
    return {"risk_summary": summary, "trace": trace}


def detect_anomalies(state: RiskAgentState) -> dict:
    """Flags limit breaches and tail-risk exposures against internal risk appetite."""
    summary = state["risk_summary"]
    flags: list[str] = []

    for breach in summary.concentration_breaches:
        flags.append(
            f"{breach.dimension.title()} concentration breach: {breach.name} is "
            f"{breach.share_of_portfolio:.1%} of portfolio EAD "
            f"(limit {breach.limit:.0%}, exposure ${breach.exposure:,.0f})."
        )

    if not summary.single_name_flags.empty:
        n = len(summary.single_name_flags)
        worst = summary.single_name_flags.iloc[0]
        flags.append(
            f"{n} single-name exposures breach PD/size limits — worst case "
            f"{worst['loan_id']} ({worst['sector']}, {worst['region']}) at "
            f"PD={worst['probability_default']:.1%}, "
            f"EAD=${worst['exposure_at_default']:,.0f}, "
            f"rated {worst['risk_rating']}."
        )

    if abs(summary.eve_sensitivity) > 0.02 * summary.total_exposure:
        flags.append(
            f"Interest-rate sensitivity is elevated: a {LIMITS.dv01_shock_bps:.0f}bp "
            f"parallel shock moves EVE by ${summary.eve_sensitivity:,.0f} "
            f"({abs(summary.eve_sensitivity) / summary.total_exposure:.1%} of EAD)."
        )

    if len(flags) >= 3:
        severity = "high"
    elif len(flags) >= 1:
        severity = "medium"
    else:
        severity = "low"

    trace = state.get("trace", [])
    trace.append(f"Anomaly Detection: {len(flags)} flag(s) raised, severity={severity}.")
    return {"flags": flags, "severity": severity, "trace": trace}


_MEMO_SYSTEM_PROMPT = """\
You are a risk analyst assistant at a bank's Treasury & Corporate (CTC) Risk \
team. Write a concise, professional risk memo for a risk manager based ONLY \
on the structured metrics and flags provided. Do not invent numbers that are \
not given to you.

Structure the memo with these sections:
1. Portfolio Overview (1-2 sentences)
2. Key Risk Metrics (short bullet list)
3. Flags & Concentration Risks (short bullet list; write "None" if empty)
4. Recommended Actions (2-3 concrete, specific bullets)

Keep the tone factual and boardroom-appropriate. Target 150-220 words.
"""


def _build_memo_payload(state: RiskAgentState) -> str:
    s = state["risk_summary"]
    flags = state.get("flags", [])
    lines = [
        f"Total exposure (EAD): ${s.total_exposure:,.0f}",
        f"Expected loss: ${s.expected_loss:,.0f}",
        f"Weighted-avg PD: {s.weighted_avg_pd:.2%}",
        f"Weighted-avg LGD: {s.weighted_avg_lgd:.2%}",
        f"Portfolio duration: {s.portfolio_duration:.2f} years",
        f"DV01: ${s.dv01:,.0f} per 1bp",
        f"EVE sensitivity ({LIMITS.dv01_shock_bps:.0f}bp shock): ${s.eve_sensitivity:,.0f}",
        "Top sector concentrations: " + ", ".join(
            f"{k} {v:.1%}" for k, v in s.sector_concentration.head(3).items()
        ),
        "Flags: " + ("; ".join(flags) if flags else "None"),
    ]
    return "\n".join(lines)


def _rule_based_memo(state: RiskAgentState) -> str:
    """Deterministic fallback memo used when no LLM API key is configured."""
    s = state["risk_summary"]
    flags = state.get("flags", [])
    severity = state.get("severity", "low")

    top_sectors = ", ".join(f"{k} ({v:.1%})" for k, v in s.sector_concentration.head(3).items())

    lines = [
        "PORTFOLIO OVERVIEW",
        f"The portfolio holds ${s.total_exposure:,.0f} in total exposure (EAD) with a "
        f"weighted-average PD of {s.weighted_avg_pd:.2%} and LGD of {s.weighted_avg_lgd:.2%}. "
        f"Overall flag severity is {severity.upper()}.",
        "",
        "KEY RISK METRICS",
        f"- Expected loss: ${s.expected_loss:,.0f}",
        f"- Portfolio duration: {s.portfolio_duration:.2f} years",
        f"- DV01: ${s.dv01:,.0f} per 1bp parallel move",
        f"- EVE sensitivity ({LIMITS.dv01_shock_bps:.0f}bp shock): ${s.eve_sensitivity:,.0f}",
        f"- Top sector concentrations: {top_sectors}",
        "",
        "FLAGS & CONCENTRATION RISKS",
    ]
    if flags:
        lines += [f"- {f}" for f in flags]
    else:
        lines.append("- None: all exposures within configured risk-appetite limits.")

    lines += [
        "",
        "RECOMMENDED ACTIONS",
        "- Review and, where breached, re-hedge or reduce exposures flagged above "
        "against internal concentration limits.",
        "- Escalate single-name exposures with PD or size breaches to credit "
        "committee for a rating/limit review.",
        "- Re-run interest-rate sensitivity under a wider set of yield-curve "
        "scenarios (steepening/flattening, not just parallel) given the "
        "duration/DV01 profile above.",
    ]
    return "\n".join(lines)


def summarize(state: RiskAgentState) -> dict:
    """
    Turns the calculated metrics + flags into a readable risk memo.
    Uses an LLM (Anthropic/OpenAI) when a key is configured; otherwise falls
    back to a deterministic rule-based memo so the pipeline always completes.
    """
    llm = get_llm()
    trace = state.get("trace", [])

    if llm is None:
        memo = _rule_based_memo(state)
        trace.append("Summarization: no LLM key configured — used rule-based memo.")
        return {"memo": memo, "used_llm": False, "trace": trace}

    payload = _build_memo_payload(state)
    messages = [
        ("system", _MEMO_SYSTEM_PROMPT),
        ("human", f"Here are this cycle's portfolio metrics and flags:\n\n{payload}"),
    ]
    response = llm.invoke(messages)
    memo = response.content if hasattr(response, "content") else str(response)

    trace.append(f"Summarization: memo generated via {llm.__class__.__name__}.")
    return {"memo": memo, "used_llm": True, "trace": trace}
