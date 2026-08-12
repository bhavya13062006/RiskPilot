"""
Builds the LangGraph pipeline:

    retrieve_data -> calculate_risk -> detect_anomalies -> summarize -> END

A linear graph is enough to demonstrate real agentic orchestration (explicit
state, composable nodes, inspectable trace) without over-engineering the demo
with branches it doesn't need. See README.md for how this would extend to a
conditional graph (e.g. routing to a "deep dive" sub-agent only when severity
is high).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import nodes
from .state import RiskAgentState


def build_graph():
    graph = StateGraph(RiskAgentState)

    graph.add_node("retrieve_data", nodes.retrieve_data)
    graph.add_node("calculate_risk", nodes.calculate_risk)
    graph.add_node("detect_anomalies", nodes.detect_anomalies)
    graph.add_node("summarize", nodes.summarize)

    graph.set_entry_point("retrieve_data")
    graph.add_edge("retrieve_data", "calculate_risk")
    graph.add_edge("calculate_risk", "detect_anomalies")
    graph.add_edge("detect_anomalies", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def run_pipeline(n_loans: int = 500) -> RiskAgentState:
    app = build_graph()
    result = app.invoke({"n_loans": n_loans, "trace": []})
    return result
