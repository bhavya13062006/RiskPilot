"""
AI Risk Analyst — Executive Risk Cockpit
Run with:
    streamlit run app.py

This is a presentation-grade Streamlit front end for the existing
LangGraph AI Risk Analyst pipeline. The backend contract is preserved:
run_pipeline(n_loans=...) should return the same result structure.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.agents.graph import run_pipeline


# ---------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Risk Analyst | Executive Cockpit",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY = {
    "low": ("LOW", "#16835b"),
    "medium": ("MEDIUM", "#b7791f"),
    "high": ("HIGH", "#c53030"),
}


def money(x: Any) -> str:
    try:
        x = float(x)
        if abs(x) >= 1_000_000:
            return f"${x / 1_000_000:.1f}M"
        if abs(x) >= 1_000:
            return f"${x / 1_000:.0f}K"
        return f"${x:,.0f}"
    except Exception:
        return "—"


def pct(x: Any) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "—"


def num(x: Any, suffix: str = "") -> str:
    try:
        return f"{float(x):,.2f}{suffix}"
    except Exception:
        return "—"


def get_attr(obj: Any, *names: str, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def inject_css():
    st.markdown(
        """
        <style>
        /* Overall canvas */
        .stApp {
            background:
                radial-gradient(circle at 85% 5%, rgba(37,99,235,.07), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #ffffff 42%, #f8fafc 100%);
        }

        [data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        /* Hide Streamlit chrome for a cleaner product feel */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .hero {
            padding: 28px 30px 25px 30px;
            border-radius: 22px;
            background: linear-gradient(135deg, #0b1220 0%, #172554 62%, #1e3a8a 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(15,23,42,.14);
            margin-bottom: 22px;
        }
        .hero-kicker {
            font-size: 12px;
            letter-spacing: .16em;
            text-transform: uppercase;
            color: #93c5fd;
            font-weight: 700;
        }
        .hero-title {
            font-size: 35px;
            line-height: 1.08;
            font-weight: 800;
            margin: 8px 0 8px 0;
        }
        .hero-sub {
            font-size: 15px;
            color: #cbd5e1;
            max-width: 820px;
            line-height: 1.55;
        }

        .section-title {
            font-size: 21px;
            font-weight: 800;
            color: #0f172a;
            margin: 18px 0 10px;
        }

        .kpi {
            background: rgba(255,255,255,.96);
            border: 1px solid #e2e8f0;
            border-radius: 17px;
            padding: 17px 18px;
            min-height: 112px;
            box-shadow: 0 7px 25px rgba(15,23,42,.05);
        }
        .kpi-label {
            color: #64748b;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .07em;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 25px;
            font-weight: 800;
            margin-top: 8px;
        }
        .kpi-note {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 3px;
        }

        .status-card {
            border-radius: 16px;
            padding: 16px 18px;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 7px 25px rgba(15,23,42,.04);
        }
        .status-pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            color: white;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .06em;
        }

        .workflow {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 8px 0 18px;
        }
        .step {
            background: #eef2ff;
            color: #3730a3;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 12px;
            font-weight: 700;
        }

        .insight {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            border-radius: 0 12px 12px 0;
            padding: 13px 15px;
            color: #334155;
            margin-bottom: 9px;
        }

        .footnote {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_css()

    # -------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ◈ AI Risk Analyst")
        st.caption("Executive risk intelligence cockpit")
        st.divider()

        st.markdown("### Simulation")
        n_loans = st.slider(
            "Portfolio size",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
        )
        refresh = st.button(
            "↻  Run fresh risk analysis",
            use_container_width=True,
            type="primary",
        )

        st.divider()
        st.markdown("### Analysis layers")
        st.caption("Credit risk  •  IRR  •  Concentration  •  Duration")
        st.caption("LLM memo  •  Limit monitoring  •  Agent trace")

        st.divider()
        st.markdown(
            "<div style='font-size:11px;color:#94a3b8'>"
            "Prototype environment · Synthetic portfolio · "
            "Decision-support only"
            "</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------
    if "result" not in st.session_state or refresh:
        from src.data_loader import load_portfolio

        load_portfolio(refresh=True, n_loans=n_loans)
        with st.spinner(
            "Running AI risk workflow · retrieve → calculate → detect → summarize"
        ):
            st.session_state["result"] = run_pipeline(n_loans=n_loans)

    result = st.session_state["result"]
    summary = result["risk_summary"]
    portfolio: pd.DataFrame = result["portfolio"]
    flags = result.get("flags", [])
    severity = str(result.get("severity", "low")).lower()
    sev_text, sev_color = SEVERITY.get(severity, SEVERITY["low"])

    # -------------------------------------------------------------
    # Hero
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">AI-powered treasury & credit intelligence</div>
          <div class="hero-title">Risk Command Center</div>
          <div class="hero-sub">
            A decision-oriented view of portfolio exposure, expected loss,
            interest-rate sensitivity and concentration risk — powered by an
            agent workflow rather than a static dashboard.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Workflow visual — makes the app feel like an actual product
    st.markdown(
        """
        <div class="workflow">
          <span class="step">01 · Retrieve</span>
          <span class="step">02 · Calculate</span>
          <span class="step">03 · Detect breaches</span>
          <span class="step">04 · Assess severity</span>
          <span class="step">05 · Draft memo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------
    tab_exec, tab_risk, tab_portfolio, tab_agent = st.tabs(
        ["Executive Brief", "Risk Analytics", "Portfolio Explorer", "AI Trace"]
    )

    # =============================================================
    # EXECUTIVE BRIEF
    # =============================================================
    with tab_exec:
        st.markdown('<div class="section-title">Executive snapshot</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi("Total exposure", money(summary.total_exposure), "Exposure at default")
        with c2:
            kpi("Expected loss", money(summary.expected_loss), "Portfolio-level estimate")
        with c3:
            kpi("Weighted PD", pct(summary.weighted_avg_pd), "Exposure-weighted")
        with c4:
            kpi("Portfolio duration", num(summary.portfolio_duration, " yrs"), "Rate sensitivity horizon")
        with c5:
            kpi("DV01", money(summary.dv01), "Value change per bp")

        st.write("")
        left, right = st.columns([1.35, 1])

        with left:
            st.markdown("### Risk posture")
            st.markdown(
                f"""
                <div class="status-card">
                    <span class="status-pill" style="background:{sev_color}">{sev_text} RISK</span>
                    <h3 style="margin:12px 0 5px;color:#0f172a">
                        {len(flags)} risk flag(s) identified
                    </h3>
                    <p style="margin:0;color:#64748b">
                        The agent has reviewed the synthetic portfolio and surfaced
                        concentration / limit observations for analyst review.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Decision signals")
            if flags:
                for flag in flags[:5]:
                    st.markdown(f'<div class="insight">⚠ {flag}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="insight">✓ No limit breaches detected in this run.</div>',
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("### Exposure concentration")
            sector = getattr(summary, "sector_concentration", pd.Series(dtype=float))
            if not sector.empty:
                st.bar_chart(sector)
            else:
                st.info("Sector concentration data is not available.")

        st.markdown("### Analyst memo")
        st.markdown(result["memo"])

    # =============================================================
    # RISK ANALYTICS
    # =============================================================
    with tab_risk:
        st.markdown('<div class="section-title">Risk analytics</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.markdown("#### Sector concentration")
            sector = getattr(summary, "sector_concentration", pd.Series(dtype=float))
            if not sector.empty:
                st.bar_chart(sector)
            else:
                st.info("No sector data available.")

        with b:
            st.markdown("#### Regional concentration")
            region = getattr(summary, "region_concentration", pd.Series(dtype=float))
            if not region.empty:
                st.bar_chart(region)
            else:
                st.info("No regional data available.")

        st.markdown("#### Interest-rate risk")
        ir1, ir2, ir3 = st.columns(3)
        with ir1:
            kpi("Duration", num(summary.portfolio_duration, " yrs"))
        with ir2:
            kpi("DV01", money(summary.dv01))
        with ir3:
            # Keep this compatible with the current backend; if EVE/NII
            # fields are added later, the UI will automatically surface them.
            eve = get_attr(summary, "eve_sensitivity", "eve", default=None)
            nii = get_attr(summary, "nii_sensitivity", "nii", default=None)
            kpi(
                "EVE / NII sensitivity",
                money(eve) if eve is not None else "Ready",
                "Auto-surfaces when backend fields are added"
                if eve is None
                else f"NII: {money(nii)}" if nii is not None else "Economic value sensitivity",
            )

        st.markdown("#### Liquidity risk")
        l1, l2, l3 = st.columns(3)
        with l1:
            nsfr = get_attr(summary, "nsfr", "net_stable_funding_ratio", default=None)
            kpi("NSFR", pct(nsfr) if nsfr is not None else "Ready",
                "Structural liquidity ratio")
        with l2:
            lcr = get_attr(summary, "lcr", "liquidity_coverage_ratio", default=None)
            kpi("LCR", pct(lcr) if lcr is not None else "Ready",
                "Short-term liquidity ratio")
        with l3:
            kpi("Flags", str(len(flags)), "Limit / concentration observations")

        st.caption(
            "The current front end is intentionally backward-compatible. "
            "NSFR, LCR, EVE and NII cards populate automatically once those "
            "metrics are exposed by the existing risk_summary object."
        )

    # =============================================================
    # PORTFOLIO EXPLORER
    # =============================================================
    with tab_portfolio:
        st.markdown('<div class="section-title">Portfolio explorer</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Loans", f"{len(portfolio):,}")
        with p2:
            st.metric("Columns", f"{len(portfolio.columns):,}")
        with p3:
            st.metric("Missing values", f"{int(portfolio.isna().sum().sum()):,}")

        st.dataframe(
            portfolio,
            use_container_width=True,
            height=520,
            hide_index=True,
        )

        st.download_button(
            "Download portfolio CSV",
            data=portfolio.to_csv(index=False).encode("utf-8"),
            file_name="risk_portfolio.csv",
            mime="text/csv",
        )

    # =============================================================
    # AI TRACE
    # =============================================================
    with tab_agent:
        st.markdown('<div class="section-title">Agent execution trace</div>', unsafe_allow_html=True)

        trace = result.get("trace", [])
        if trace:
            for i, step in enumerate(trace, start=1):
                st.markdown(
                    f"""
                    <div class="status-card" style="margin-bottom:8px">
                      <b style="color:#2563eb">STEP {i:02d}</b>
                      <div style="margin-top:5px;color:#334155">{step}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No execution trace returned.")

        engine = (
            "LLM-generated"
            if result.get("used_llm")
            else "Rule-based fallback · no API key configured"
        )
        st.success(f"Memo engine: {engine}")

        st.markdown(
            """
            **Why this matters:** the interface is designed to show not only
            the answer, but the workflow that produced it. That makes the
            project easier to explain in a placement interview, demo, or
            portfolio review.
            """
        )

    st.markdown(
        "<div class='footnote'>AI Risk Analyst · Synthetic data · "
        "For demonstration and analyst-support purposes</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
