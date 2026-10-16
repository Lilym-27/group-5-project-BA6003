"""
Streamlit wrapper for the Innovation Investment COO Report Agent.

Run (from the project root):
    uv run streamlit run agent/streamlit_app.py
"""

import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from dotenv import load_dotenv

load_dotenv(_AGENT_DIR.parent / ".env")

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Innovation Investment — COO Report Agent",
    layout="wide",
)

ROOT_DIR = _AGENT_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


@st.cache_resource
def configure_logger():
    logger.add(
        LOG_DIR / "week11_agent.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )
    return True


configure_logger()


# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    path = ROOT_DIR / "Sample.xlsx"
    if not path.exists():
        st.error(f"Sample.xlsx not found at {path}")
        st.stop()

    df = pd.read_excel(path)

    numeric_cols = [
        "year", "revenue", "rd_expense", "net_profit", "total_assets",
        "market_value", "total_liabilities", "patent",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rd_intensity"] = np.where(
        df["revenue"].notna() & (df["revenue"] != 0),
        df["rd_expense"] / df["revenue"],
        np.nan,
    )
    df["roa"] = np.where(
        df["total_assets"].notna() & (df["total_assets"] != 0),
        df["net_profit"] / df["total_assets"],
        np.nan,
    )
    df["tobinq"] = np.where(
        df["total_assets"].notna() & (df["total_assets"] != 0),
        (df["market_value"] + df["total_liabilities"]) / df["total_assets"],
        np.nan,
    )
    return df


df = load_data()

# Inject data into the tools module
import tools as tools_module   # noqa: E402
tools_module.df_global = df


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("📈 Innovation Investment — COO Report Agent")
st.markdown(
    "Click the button below to run the autonomous agent. "
    "It will call several analytical tools and produce a structured risk report."
)

col1, col2, col3 = st.columns(3)
col1.metric("Observations", f"{len(df):,}")
col2.metric("Companies", f"{df['name'].nunique():,}")
col3.metric("Industries", f"{df['industry'].nunique():,}")

st.divider()

if st.button("🎯 Generate COO Report", type="primary"):
    from coo_agent import run_coo_report_agent   # noqa: E402

    logger.info("COO report agent triggered from Streamlit")

    try:
        with st.spinner("Agent is analysing the portfolio — this may take a minute..."):
            report = run_coo_report_agent()

        st.markdown(report)
        st.caption(
            "AI-generated analysis. Verify important claims against the underlying data."
        )
    except Exception as exc:
        logger.exception("Agent run failed")
        st.error(f"Agent failed: {exc}")

st.divider()
with st.expander("About this agent"):
    st.markdown(
        """
        - The agent uses **6 tools** to query the innovation dataset.
        - It decides on its own which tools to call, in what order, and how many times.
        - Every tool call is logged to `logs/week11_agent.log`.
        - The final output is a structured markdown report with 7 sections.
        """
    )