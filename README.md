# AI Risk Analyst Agent

An **agentic risk-analysis pipeline** built with **LangGraph** that mirrors the
workflow of a bank risk analyst: pull portfolio data, compute credit and
interest-rate risk metrics, flag limit breaches, and draft a natural-language
risk memo for a risk manager — with an interactive **Streamlit** dashboard on top.

Built as a practical exploration of using foundational models and agent
orchestration frameworks (LangChain/LangGraph) for risk management use cases —
market/credit/interest-rate risk analytics, automated flagging, and
LLM-assisted reporting.

## Why this project

Most portfolio ML/risk projects stop at a static dashboard. This one goes a
step further: it wires risk calculations into an **agent graph** where each
step (retrieval → calculation → anomaly detection → summarization) is an
explicit, inspectable node with typed state passed between them — the same
pattern used to build production agentic systems on top of LLMs.

## Architecture

```
                 ┌────────────────┐
   n_loans  ───▶ │ retrieve_data  │  loads / generates the loan portfolio
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │ calculate_risk │  PD/LGD/EAD → Expected Loss, duration,
                 └───────┬────────┘  DV01, EVE sensitivity, concentrations
                         ▼
                 ┌──────────────────┐
                 │ detect_anomalies │  checks metrics against risk-appetite
                 └───────┬──────────┘  limits (RiskLimits), raises flags
                         ▼
                 ┌────────────────┐
                 │   summarize    │  drafts a risk memo — LLM-generated when
                 └───────┬────────┘  an API key is set, else rule-based
                         ▼
                       memo + trace
```

Implemented as a `langgraph.graph.StateGraph` over a single typed state object
(`RiskAgentState`), so the whole run — inputs, intermediate metrics, flags,
and the final memo — is inspectable at every step (see the "Agent execution
trace" panel in the UI).

## Risk metrics implemented

| Metric | What it measures |
|---|---|
| **Expected Loss (EL)** | `PD × LGD × EAD`, per loan and portfolio-wide |
| **Portfolio duration** | Exposure-weighted interest-rate duration |
| **DV01** | Dollar value of a 1bp parallel rate move |
| **EVE sensitivity** | First-order change in Economic Value of Equity under a configurable rate shock |
| **Concentration risk** | Exposure share by sector / region vs. configured limits |
| **Single-name flags** | Obligors breaching PD or exposure-size limits |

All calculations live in `src/risk_metrics.py`, are framework-agnostic (pure
pandas/numpy), and are unit-tested independently of the agent/LLM layer.

## LLM-backed summarization, with graceful fallback

The `summarize` node uses an LLM  to turn the computed metrics and flags into
a boardroom-ready memo. If no API key is configured, it falls back to a
deterministic, rule-based memo built from the same data, so **the full
pipeline always runs end-to-end** — with or without a key. This also makes the
project runnable in CI/without secrets while still demonstrating real
LLM-generated output when a key is present.

## Getting started

```bash
git clone <this-repo>
cd ai_risk_analyst_agent
pip install -r requirements.txt

# Optional: enable LLM-generated memos
cp .env.example .env
# then set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env

streamlit run app.py
```

Run the pipeline without the UI:

```bash
python -c "from src.agents.graph import run_pipeline; r = run_pipeline(); print(r['memo'])"
```

Run the tests:

```bash
pytest tests/ -v
```

## Project structure

```
ai_risk_analyst_agent/
├── app.py                       # Streamlit dashboard
├── data/
│   └── generate_synthetic_data.py   # synthetic loan portfolio (schema documented inline)
├── src/
│   ├── data_loader.py            # swap-in point for a real data source
│   ├── risk_metrics.py           # EL, duration, DV01, EVE, concentration, single-name flags
│   ├── llm_config.py             # provider-agnostic LLM setup + fallback
│   └── agents/
│       ├── state.py               # typed LangGraph state schema
│       ├── nodes.py               # the 4 agent nodes
│       └── graph.py               # graph wiring (retrieve → calc → detect → summarize)
└── tests/
    └── test_risk_metrics.py       # unit tests for the calculation layer
```

## Data

The demo runs on a **synthetic** loan portfolio (`data/generate_synthetic_data.py`)
with realistic rating-to-PD bands and a few deliberately injected concentration
/ tail-risk scenarios so the anomaly-detection agent has real breaches to
catch. Swap `src/data_loader.load_portfolio` for a real data source (warehouse
query, CSV export, a public credit-risk dataset) as long as it returns the
documented schema.

## Possible extensions

- **Conditional routing:** branch to a "deep dive" sub-agent only when
  `severity == "high"`, using LangGraph's conditional edges.
- **Tool-calling agent:** let the LLM itself decide which risk metrics to pull
  via LangChain tools, rather than a fixed linear graph.
- **Multi-scenario stress testing:** loop the calculation node over a set of
  yield-curve shocks (steepening, flattening, parallel) instead of one.
- **Persistence:** checkpoint state with LangGraph's built-in checkpointing to
  support human-in-the-loop review before the memo is finalized.

## Tech stack

Python · LangGraph · LangChain (Anthropic / OpenAI) · pandas · NumPy ·
Streamlit · pytest
