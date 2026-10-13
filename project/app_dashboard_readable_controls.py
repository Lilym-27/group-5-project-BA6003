import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "ai_usage.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

st.set_page_config(
    page_title="Innovation Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
      --ink: #14253d;
      --muted: #63748b;
      --blue: #315efb;
      --teal: #10a6a0;
      --line: #e4eaf2;
      --panel: #ffffff;
      --canvas: #f5f7fb;
    }
    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
    [data-testid="stSidebar"] {
      background: #10213a;
      border-right: 1px solid #243752;
    }
    /* Sidebar readability: use dark text on the light sidebar surface */
    [data-testid="stSidebar"] {
      background: #ffffff;
      border-right: 1px solid #e4eaf2;
    }
    [data-testid="stSidebar"] * {
      color: #243b60;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
      color: #52647b !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    [data-testid="stSidebar"] [data-testid="stSlider"] p,
    [data-testid="stSidebar"] [data-testid="stSlider"] label {
      color: #14253d !important;
      -webkit-text-fill-color: #14253d !important;
      opacity: 1 !important;
    }

    /* Slider values, number-input values, and +/- controls need explicit contrast */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
      color: #14253d !important;
      -webkit-text-fill-color: #14253d !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button *,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
      background: #203653 !important;
      color: #ffffff !important;
      fill: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:disabled,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:disabled * {
      color: #cbd5e1 !important;
      fill: #cbd5e1 !important;
      -webkit-text-fill-color: #cbd5e1 !important;
      opacity: 1 !important;
    }

    /* Sidebar controls: maintain readable contrast on the dark sidebar */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
      color: #dce7f7 !important;
    }
    /* Force the actual widget surfaces to use dark text on light inputs */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      background: #ffffff !important;
      border: 1px solid #d6dfeb !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
      color: #14253d !important;
      -webkit-text-fill-color: #14253d !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] {
      background: #ffffff !important;
      border: 1px solid #d6dfeb !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input[type="number"],
    [data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] input {
      background: #ffffff !important;
      color: #14253d !important;
      -webkit-text-fill-color: #14253d !important;
      caret-color: #315efb !important;
      opacity: 1 !important;
      font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] input:focus {
      color: #14253d !important;
      -webkit-text-fill-color: #14253d !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input::placeholder {
      color: #63748b !important;
      -webkit-text-fill-color: #63748b !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
      background: #203653 !important;
      border-color: #48617f !important;
      color: #f8fbff !important;
      -webkit-text-fill-color: #f8fbff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
      fill: #63748b !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
      background-color: #315174 !important;
      border-color: #78a7e8 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
    [data-testid="stSidebar"] [data-baseweb="input"] > div:focus-within {
      border-color: #7aa7ff !important;
      box-shadow: 0 0 0 1px #7aa7ff !important;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {
      background-color: #ffffff !important;
      color: #14253d !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] * {
      color: #14253d !important;
    }

    .block-container {
      padding-top: 2rem;
      padding-bottom: 3rem;
      max-width: 1500px;
    }
    h1, h2, h3 {
      font-family: 'Manrope', sans-serif !important;
      color: var(--ink);
      letter-spacing: -0.035em;
    }
    h1 { font-size: 2.15rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.4rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
    [data-testid="stMetric"] {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 20px;
      box-shadow: 0 5px 18px rgba(22, 42, 73, .035);
      min-height: 112px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .84rem; }
    [data-testid="stMetricValue"] {
      color: var(--ink); font-family: 'Manrope', sans-serif;
      font-weight: 800; font-size: 1.75rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      background: #fff; border: 1px solid var(--line);
      border-radius: 16px; padding: 14px 16px;
    }
    .hero {
      padding: 26px 30px; border-radius: 22px; margin-bottom: 18px;
      background: linear-gradient(115deg, #112541 0%, #1b3c67 58%, #315efb 120%);
      color: white; border: 1px solid rgba(255,255,255,.12);
      box-shadow: 0 12px 32px rgba(16,33,58,.14);
    }
    .hero-kicker {
      color: #9bdedb; font-size: .75rem; text-transform: uppercase;
      letter-spacing: .15em; font-weight: 700; margin-bottom: 8px;
    }
    .hero-title {
      color: #fff; font-family: 'Manrope', sans-serif; font-weight: 800;
      font-size: 2rem; line-height: 1.2; letter-spacing: -.04em;
    }
    .hero-copy { color: #d4e0f2; margin-top: 10px; max-width: 780px; line-height: 1.65; }
    .section-kicker {
      color: var(--blue); font-size: .72rem; font-weight: 800;
      text-transform: uppercase; letter-spacing: .13em; margin-bottom: 3px;
    }
    .section-note { color: var(--muted); font-size: .88rem; margin-top: -8px; margin-bottom: 16px; }
    .insight {
      background: #eef4ff; border: 1px solid #dce7ff; border-left: 4px solid #315efb;
      border-radius: 12px; padding: 14px 16px; color: #243b60;
    }
    .ai-panel {
      background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
      border: 1px solid #dce7f5; border-radius: 18px; padding: 20px 22px;
    }
    div.stButton > button[kind="primary"] {
      background: linear-gradient(100deg, #315efb, #237bd7);
      border: 0; border-radius: 10px; font-weight: 700;
      padding: .65rem 1.1rem; box-shadow: 0 6px 16px rgba(49,94,251,.2);
    }
    div.stButton > button { border-radius: 10px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    hr { border-color: var(--line); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data loading ----------
DATA_CANDIDATES = [
    ROOT_DIR / "Sample.xlsx",
    Path(__file__).resolve().parent / "Sample.xlsx",
    Path.cwd() / "Sample.xlsx",
]

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if data_path is None:
    st.error("Dataset not found. Place `Sample.xlsx` in the project root folder.")
    st.caption(f"Expected location: {ROOT_DIR / 'Sample.xlsx'}")
    st.stop()

try:
    df = load_data(str(data_path))
except Exception as exc:
    st.error("Unable to read Sample.xlsx. Check that the workbook is valid and not locked.")
    st.exception(exc)
    st.stop()

required_columns = [
    "name", "year", "revenue", "rd_expense", "net_profit",
    "total_assets", "market_value", "total_liabilities", "patent", "industry"
]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    st.error(f"Required columns are missing: {', '.join(missing)}")
    st.caption("Available columns: " + ", ".join(map(str, df.columns)))
    st.stop()

numeric_columns = [
    "year", "revenue", "rd_expense", "net_profit", "total_assets",
    "market_value", "total_liabilities", "patent"
]
for col in numeric_columns:
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

# Visual/regression sample excludes extreme or invalid R&D intensity.
plot_df = df[
    df["rd_intensity"].between(0, 0.50, inclusive="both")
    & df["rd_intensity"].notna()
].copy()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📈 INNOVATION IQ")
    st.caption("Corporate research workspace")
    st.markdown("---")
    st.markdown("### Analysis controls")
    industries = sorted(df["industry"].dropna().astype(str).unique().tolist())
    selected_industry = st.selectbox("Industry", ["All Industries"] + industries)
    available_years = sorted(
        pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    selected_years = st.slider(
        "Year range",
        min_value=min(available_years) if available_years else 2000,
        max_value=max(available_years) if available_years else 2025,
        value=(
            min(available_years) if available_years else 2000,
            max(available_years) if available_years else 2025,
        ),
        help="Filter the dashboard to a range of financial years.",
    ) if available_years and min(available_years) < max(available_years) else None
    risk_profile = st.selectbox(
        "Investor risk preference",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
        help="Changes the suggested analysis priorities and default KPI weights; it does not change the OLS model or AI logic.",
    )
    user_rd = st.number_input(
        "R&D intensity (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5
    )
    user_patent = st.number_input(
        "Patents granted", min_value=0, value=10, step=1
    )
    st.markdown("---")
    st.markdown("### About this workspace")
    st.caption(
        "Explore historical relationships between innovation investment and company performance. "
        "Patent input is contextual only and is not included in the current OLS model."
    )
    st.caption(f"Dataset: {data_path.name}")

if selected_industry == "All Industries":
    filtered_df = df.copy()
else:
    filtered_df = df[df["industry"].astype(str) == selected_industry].copy()

if selected_years is not None:
    filtered_df = filtered_df[
        pd.to_numeric(filtered_df["year"], errors="coerce").between(selected_years[0], selected_years[1])
    ].copy()

filtered_plot_df = filtered_df[
    filtered_df["rd_intensity"].between(0, 0.50, inclusive="both")
    & filtered_df["rd_intensity"].notna()
].copy()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Investment research · Data intelligence</div>
      <div class="hero-title">Corporate Innovation<br>Investment Dashboard</div>
      <div class="hero-copy">
        Explore how R&D intensity relates to company performance, compare industry benchmarks,
        and turn historical financial data into structured research questions.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="insight"> <b>Research note</b> &nbsp; This is a historical decision-support prototype. '
    'Observed associations do not establish causality and should not be interpreted as forecasts or investment advice.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------- Dataset snapshot ----------
st.markdown('<div class="section-kicker">01 / Coverage</div>', unsafe_allow_html=True)
st.header("Dataset overview")
overview = st.columns(4)
min_year = int(df["year"].min()) if df["year"].notna().any() else None
max_year = int(df["year"].max()) if df["year"].notna().any() else None
overview[0].metric("Observations", f"{len(df):,}")
overview[1].metric("Companies", f"{df['name'].nunique():,}")
overview[2].metric("Coverage period", f"{min_year}–{max_year}" if min_year and max_year else "N/A")
overview[3].metric("Industries", f"{df['industry'].nunique():,}")
st.write("")

# ---------- KPI cards ----------
st.markdown('<div class="section-kicker">02 / Selected sample</div>', unsafe_allow_html=True)
st.header("Key performance indicators")
kpis = st.columns(4)
median_rd = filtered_df["rd_intensity"].median()
median_roa = filtered_df["roa"].median()
median_q = filtered_df["tobinq"].median()
kpis[0].metric("Median R&D intensity", f"{median_rd * 100:.2f}%" if pd.notna(median_rd) else "N/A")
kpis[1].metric("Median ROA", f"{median_roa * 100:.2f}%" if pd.notna(median_roa) else "N/A")
kpis[2].metric("Median Tobin's Q", f"{median_q:.2f}" if pd.notna(median_q) else "N/A")
kpis[3].metric("Your R&D input", f"{user_rd:.1f}%")
year_context = (
    f"{selected_years[0]}–{selected_years[1]}"
    if selected_years is not None else "all available years"
)
st.caption(
    f"Current view: **{selected_industry}** · {year_context} · "
    f"{len(filtered_df):,} observations · {filtered_df['name'].nunique():,} companies"
)
st.write("")

# Shared Plotly styling
PLOT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#52647b", size=12),
    title_font=dict(family="Manrope, sans-serif", size=16, color="#14253d"),
    margin=dict(l=20, r=20, t=55, b=25),
    hoverlabel=dict(bgcolor="#14253d", font_color="white"),
)

# ---------- Main charts ----------
st.markdown('<div class="section-kicker">03 / Innovation signals</div>', unsafe_allow_html=True)
st.header("Innovation investment patterns")
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("R&D intensity distribution")
        rd = filtered_plot_df["rd_intensity"].dropna() * 100
        if not rd.empty:
            bins = [0, 5, 10, 20, 30, 40, 50.000001]
            labels = ["0–5%", "5–10%", "10–20%", "20–30%", "30–40%", "40–50%"]
            groups = pd.cut(rd, bins=bins, labels=labels, right=False, include_lowest=True)
            counts = groups.value_counts(sort=False).reindex(labels, fill_value=0).reset_index()
            counts.columns = ["R&D intensity", "Observations"]
            fig = px.bar(counts, x="R&D intensity", y="Observations", text="Observations")
            fig.update_traces(marker_color="#315efb", marker_line_width=0, textposition="outside")
            fig.update_layout(**PLOT_LAYOUT, showlegend=False, xaxis_title="", yaxis_title="Companies")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid R&D intensity observations in this sample.")
        st.caption("For comparability, the chart excludes negative values and values above 50%.")

with right:
    with st.container(border=True):
        st.subheader("R&D intensity vs. Tobin's Q")
        scatter = filtered_plot_df[["rd_intensity", "tobinq", "name", "year"]].dropna().copy()
        if len(scatter) >= 3:
            scatter["R&D intensity (%)"] = scatter["rd_intensity"] * 100
            fig = px.scatter(
                scatter, x="R&D intensity (%)", y="tobinq",
                hover_data={"name": True, "year": True, "rd_intensity": False},
                trendline="ols", opacity=0.55,
            )
            fig.update_traces(marker=dict(color="#315efb", size=7), selector=dict(mode="markers"))
            fig.update_layout(**PLOT_LAYOUT, xaxis_title="R&D intensity (%)", yaxis_title="Tobin's Q")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least three valid observations are needed for this chart.")
        st.caption("The fitted line summarizes an in-sample association; it does not establish causality.")

st.write("")
left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.subheader("R&D intensity over time")
        trend = filtered_df.dropna(subset=["year", "rd_intensity"]).groupby("year", as_index=False)["rd_intensity"].median()
        if not trend.empty:
            trend["R&D intensity (%)"] = trend["rd_intensity"] * 100
            fig = px.line(trend, x="year", y="R&D intensity (%)", markers=True)
            fig.update_traces(line=dict(color="#10a6a0", width=3), marker=dict(size=7))
            fig.update_layout(**PLOT_LAYOUT, xaxis_title="", yaxis_title="Median R&D intensity (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-series data available.")
        st.caption("Annual median R&D intensity for the selected industry/sample.")

with right:
    with st.container(border=True):
        st.subheader("Industry benchmark")
        benchmark = (
            df.dropna(subset=["industry", "rd_intensity"])
            .groupby("industry", as_index=False)["rd_intensity"].median()
        )
        benchmark["R&D intensity (%)"] = benchmark["rd_intensity"] * 100
        benchmark = benchmark.nlargest(15, "R&D intensity (%)").sort_values("R&D intensity (%)")
        if not benchmark.empty:
            fig = px.bar(
                benchmark, x="R&D intensity (%)", y="industry", orientation="h",
                labels={"industry": "Industry"},
            )
            fig.update_traces(marker_color="#10a6a0")
            fig.update_layout(**PLOT_LAYOUT, xaxis_title="Median R&D intensity (%)", yaxis_title="", height=440)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No industry benchmark data available.")
        st.caption("Top 15 industries by median R&D intensity, using the full dataset.")

# ---------- Input interpretation and scenario estimate ----------
st.markdown('<div class="section-kicker">05 / Scenario context</div>', unsafe_allow_html=True)
st.header("Your innovation inputs")

# Fit an in-sample multivariable OLS model using R&D intensity and patent count.
# R&D is expressed as a proportion in the training data, so convert user percentage accordingly.
prediction_df = filtered_df[["tobinq", "rd_intensity", "patent"]].copy()
prediction_df = prediction_df.replace([np.inf, -np.inf], np.nan).dropna()
prediction_df = prediction_df[
    prediction_df["rd_intensity"].between(0, 0.50, inclusive="both")
    & (prediction_df["patent"] >= 0)
]
prediction_model = None
prediction_q = None
prediction_status = None

if len(prediction_df) >= 20 and prediction_df["rd_intensity"].nunique() > 1 and prediction_df["patent"].nunique() > 1:
    try:
        X_prediction = sm.add_constant(prediction_df[["rd_intensity", "patent"]], has_constant="add")
        prediction_model = sm.OLS(prediction_df["tobinq"], X_prediction).fit()
        input_row = pd.DataFrame({
            "const": [1.0],
            "rd_intensity": [user_rd / 100.0],
            "patent": [float(user_patent)],
        })
        input_row = input_row[["const", "rd_intensity", "patent"]]
        prediction_q = float(prediction_model.predict(input_row).iloc[0])
        if not np.isfinite(prediction_q):
            prediction_q = None
            prediction_status = "The model returned a non-finite estimate for these inputs."
    except Exception as exc:
        prediction_status = f"Could not estimate Tobin's Q: {exc}"
else:
    prediction_status = (
        "Not enough valid historical records or variation in both R&D intensity and patents "
        "to fit a two-input model for this sample. Try selecting All Industries."
    )

input_col, insight_col = st.columns([1, 1.6], gap="large")
with input_col:
    with st.container(border=True):
        st.metric("R&D intensity entered", f"{user_rd:.1f}%")
        st.metric("Patents granted entered", f"{user_patent:,}")
        st.metric(
            "Estimated Tobin's Q",
            f"{prediction_q:.2f}" if prediction_q is not None else "N/A",
            help="In-sample OLS estimate based on the entered R&D intensity and patent count. This is not a forecast or guaranteed outcome.",
        )
with insight_col:
    with st.container(border=True):
        st.markdown("**How the estimate is generated**")
        if prediction_q is not None and prediction_model is not None:
            st.write(
                f"The two-input OLS model uses {int(prediction_model.nobs):,} historical records "
                "from the selected sample. It estimates Tobin's Q from R&D intensity and patent count."
            )
            st.caption(
                f"Model R²: {prediction_model.rsquared:.3f} · "
                f"R&D p-value: {prediction_model.pvalues.get('rd_intensity', np.nan):.4f} · "
                f"Patent p-value: {prediction_model.pvalues.get('patent', np.nan):.4f}"
            )
        else:
            st.warning(prediction_status or "A two-input estimate is unavailable for this sample.")
        if pd.notna(median_rd):
            gap_pp = user_rd - median_rd * 100
            if gap_pp > 0.05:
                message = f"Your R&D intensity input is **{gap_pp:.2f} percentage points above** the selected-sample median ({median_rd * 100:.2f}%)."
            elif gap_pp < -0.05:
                message = f"Your R&D intensity input is **{abs(gap_pp):.2f} percentage points below** the selected-sample median ({median_rd * 100:.2f}%)."
            else:
                message = f"Your R&D intensity input is approximately equal to the selected-sample median ({median_rd * 100:.2f}%)."
            st.markdown(message)
        else:
            st.write("The selected sample median could not be calculated.")
        st.caption("This is a descriptive comparison, not an estimate of expected company performance.")
        st.info("Patent count is captured as context for the AI narrative; it is not a variable in the current OLS regression.")

# ---------- OLS model ----------
st.markdown('<div class="section-kicker">06 / Statistical evidence</div>', unsafe_allow_html=True)
st.header("OLS regression")
reg_df = filtered_df[["tobinq", "rd_intensity"]].dropna()
reg_df = reg_df[reg_df["rd_intensity"].between(0, 0.50, inclusive="both")]
if len(reg_df) >= 10 and reg_df["rd_intensity"].nunique() > 1:
    try:
        model = sm.OLS(reg_df["tobinq"], sm.add_constant(reg_df["rd_intensity"])).fit()
        beta = float(model.params.iloc[1])
        r2 = float(model.rsquared)
        pval = float(model.pvalues.iloc[1])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R&D coefficient", f"{beta:.5f}")
        m2.metric("R-squared", f"{r2:.4f}")
        m3.metric("P-value", f"{pval:.4f}")
        m4.metric("Model observations", f"{int(model.nobs):,}")
        with st.container(border=True):
            st.markdown("**Model specification**")
            st.latex(r"\text{Tobin's Q}_i = \alpha + \beta \cdot \text{R\&D Intensity}_i + \varepsilon_i")
            st.write(
                "Coefficient interpretation: "
                + ("positive" if beta > 0 else "negative" if beta < 0 else "approximately zero")
                + " in this sample. "
                + ("The coefficient is statistically significant at the 5% level." if pval < 0.05 else "The coefficient is not statistically significant at the 5% level.")
            )
            st.caption("Simple OLS with R&D intensity as the sole explanatory variable. No controls or causal identification are included.")
    except Exception as exc:
        st.warning("The regression model could not be estimated.")
        st.exception(exc)
else:
    st.info("At least 10 valid observations and variation in R&D intensity are needed for the OLS model.")

# ---------- Interactive drilldown: chart, current view text and drill_df ----------
st.markdown('<div class="section-kicker">04 / Interactive exploration</div>', unsafe_allow_html=True)
st.header("Interactive Drilldown")
st.caption(
    "Choose a grouping dimension and metric to explore the selected sample. "
    "The chart and table update together when the controls change."
)

drill_controls = st.columns(2, gap="large")
with drill_controls[0]:
    drill_dimension = st.selectbox(
        "Group by dimension",
        ["Industry", "Year", "Company"],
        index=0,
        key="drill_dimension",
    )
with drill_controls[1]:
    drill_metric_label = st.selectbox(
        "Metric to analyse",
        ["R&D intensity (%)", "ROA (%)", "Tobin's Q", "Patents granted"],
        index=0,
        key="drill_metric",
    )

dimension_map = {
    "Industry": "industry",
    "Year": "year",
    "Company": "name",
}
metric_map = {
    "R&D intensity (%)": ("rd_intensity", 100.0, "Median R&D intensity (%)"),
    "ROA (%)": ("roa", 100.0, "Median ROA (%)"),
    "Tobin's Q": ("tobinq", 1.0, "Median Tobin's Q"),
    "Patents granted": ("patent", 1.0, "Median patents granted"),
}
group_column = dimension_map[drill_dimension]
metric_column, metric_multiplier, metric_axis_label = metric_map[drill_metric_label]

drill_source = filtered_df[[group_column, metric_column, "name", "year"]].copy()
drill_source[metric_column] = pd.to_numeric(drill_source[metric_column], errors="coerce")
drill_source = drill_source.replace([np.inf, -np.inf], np.nan).dropna(subset=[group_column, metric_column])

if drill_dimension == "Year":
    drill_source[group_column] = pd.to_numeric(drill_source[group_column], errors="coerce")
    drill_source = drill_source.dropna(subset=[group_column])
    drill_source[group_column] = drill_source[group_column].astype(int).astype(str)

drill_df = (
    drill_source.groupby(group_column, dropna=True)
    .agg(
        metric_value=(metric_column, "median"),
        observations=(metric_column, "count"),
        companies=("name", "nunique"),
    )
    .reset_index()
    .rename(columns={group_column: "Group"})
)
drill_df["metric_value"] = drill_df["metric_value"] * metric_multiplier
drill_df = drill_df.sort_values("metric_value", ascending=False, kind="stable")

if drill_df.empty:
    st.info("No valid observations are available for this drilldown selection.")
else:
    # For long dimensions, show the top 15 groups to keep the chart readable.
    drill_chart_df = drill_df.head(15).copy()
    if drill_dimension == "Year":
        drill_chart_df = drill_df.sort_values("Group").tail(15).copy()

    current_year_text = (
        f"from {selected_years[0]} to {selected_years[1]}"
        if selected_years is not None else "across all available years"
    )
    current_view_text = (
        f"Currently viewing **{metric_axis_label} by {drill_dimension.lower()}** "
        f"for **{selected_industry}**, {current_year_text}. "
        f"The chart shows up to 15 groups; the table below contains the grouped results."
    )
    st.markdown(
        f'<div class="insight"><b>Current view</b><br>{current_view_text}</div>',
        unsafe_allow_html=True,
    )

    chart_df = drill_chart_df.sort_values("metric_value", ascending=True)
    if drill_dimension == "Year":
        fig = px.line(
            chart_df, x="Group", y="metric_value", markers=True,
            labels={"Group": "Year", "metric_value": metric_axis_label},
        )
        fig.update_traces(line=dict(color="#315efb", width=3), marker=dict(size=7))
    else:
        fig = px.bar(
            chart_df, x="metric_value", y="Group", orientation="h",
            labels={"Group": drill_dimension, "metric_value": metric_axis_label},
        )
        fig.update_traces(marker_color="#315efb", texttemplate="%{x:.2f}", textposition="outside")
    fig.update_layout(**PLOT_LAYOUT, xaxis_title=metric_axis_label, yaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top groups table (drill_df)")
    display_drill_df = drill_df.copy()
    display_drill_df["metric_value"] = display_drill_df["metric_value"].round(3)
    display_drill_df = display_drill_df.rename(columns={
        "metric_value": metric_axis_label,
        "observations": "Observations",
        "companies": "Companies",
    })
    st.dataframe(display_drill_df.head(50), use_container_width=True, hide_index=True)
    st.caption(
        "Groups are ranked by the selected metric except for the time-series chart, which is displayed chronologically. "
        "Values are historical sample summaries, not forecasts."
    )

# ---------- Risk preference, KPI weighting and risk signals ----------
st.markdown('<div class="section-kicker">06 / Investor profile</div>', unsafe_allow_html=True)
st.header("Risk Preference & KPI Priorities")
st.caption(
    "This is a configurable, descriptive screening aid based on the available historical data. "
    "It is not a portfolio-risk model, a stock recommendation, or a forecast."
)

profile_settings = {
    "Conservative": {
        "label": "Conservative",
        "description": "Prioritise operating performance and historical stability; inspect weak or incomplete data carefully.",
        "weights": {"ROA / operating performance": 35, "Tobin's Q / valuation context": 20,
                    "R&D intensity": 15, "Patents granted": 10, "Historical stability": 20},
    },
    "Balanced": {
        "label": "Balanced",
        "description": "Balance operating performance, valuation context, and innovation inputs and outputs.",
        "weights": {"ROA / operating performance": 25, "Tobin's Q / valuation context": 20,
                    "R&D intensity": 20, "Patents granted": 15, "Historical stability": 20},
    },
    "Aggressive": {
        "label": "Aggressive",
        "description": "Give more emphasis to innovation inputs and patent output while retaining operating and stability checks.",
        "weights": {"ROA / operating performance": 15, "Tobin's Q / valuation context": 15,
                    "R&D intensity": 30, "Patents granted": 25, "Historical stability": 15},
    },
}
chosen_profile = profile_settings[risk_profile]
st.markdown(f"**Selected profile: {chosen_profile['label']}**")
st.write(chosen_profile["description"])
st.caption(
    "Default weights are illustrative product settings, not empirically validated investment rules. "
    "You can adjust them below; the app normalises the weights to total 100%."
)

# Use editable weights, seeded from the selected profile.
weight_cols = st.columns(5)
weight_values = {}
for idx, (factor, default_weight) in enumerate(chosen_profile["weights"].items()):
    with weight_cols[idx]:
        weight_values[factor] = st.slider(
            factor,
            min_value=0,
            max_value=100,
            value=int(default_weight),
            step=5,
            key=f"risk_weight_{risk_profile}_{idx}",
        )

weight_total = sum(weight_values.values())
if weight_total <= 0:
    st.warning("Set at least one KPI weight above zero to calculate the relative profile score.")
    normalised_weights = {key: 0.0 for key in weight_values}
else:
    normalised_weights = {key: value / weight_total for key, value in weight_values.items()}

weight_display = pd.DataFrame({
    "KPI": list(normalised_weights.keys()),
    "Selected weight": [round(value * 100, 1) for value in normalised_weights.values()],
})
weight_display = weight_display.sort_values("Selected weight", ascending=False)
fig_weights = px.bar(
    weight_display.sort_values("Selected weight"),
    x="Selected weight",
    y="KPI",
    orientation="h",
    text="Selected weight",
)
fig_weights.update_traces(marker_color="#315efb", texttemplate="%{text:.1f}%", textposition="outside")
fig_weights.update_layout(**PLOT_LAYOUT, xaxis_title="Normalised weight (%)", yaxis_title="", showlegend=False)
st.plotly_chart(fig_weights, use_container_width=True)

# Relative percentile signals for the selected sample. These are not predicted returns.
risk_source = filtered_df.copy()
risk_source = risk_source.replace([np.inf, -np.inf], np.nan)

# Historical stability is defined here as the inverse percentile of each company's
# within-company ROA standard deviation, calculated only when at least two annual observations exist.
company_roa_std = (
    risk_source.dropna(subset=["name", "year", "roa"])
    .groupby("name")
    .filter(lambda group: group["year"].nunique() >= 2)
    .groupby("name")["roa"]
    .std()
)
if not company_roa_std.empty:
    risk_source["roa_volatility"] = risk_source["name"].map(company_roa_std)
else:
    risk_source["roa_volatility"] = np.nan

# Rank-based scores are more robust to different units, but remain sample-relative.
score_inputs = {
    "ROA / operating performance": ("roa", True),
    "Tobin's Q / valuation context": ("tobinq", True),
    "R&D intensity": ("rd_intensity", True),
    "Patents granted": ("patent", True),
    "Historical stability": ("roa_volatility", False),
}
factor_scores = {}
for factor, (column, higher_is_better) in score_inputs.items():
    values = pd.to_numeric(risk_source[column], errors="coerce")
    if values.notna().sum() >= 2 and values.nunique(dropna=True) >= 2:
        percentile = values.rank(pct=True)
        if not higher_is_better:
            percentile = 1 - percentile
        factor_scores[factor] = float(percentile.mean() * 100)
    else:
        factor_scores[factor] = np.nan

valid_weighted_factors = [
    factor for factor in normalised_weights
    if normalised_weights[factor] > 0 and pd.notna(factor_scores.get(factor, np.nan))
]
available_weight_total = sum(normalised_weights[factor] for factor in valid_weighted_factors)
if available_weight_total > 0:
    relative_score = sum(
        factor_scores[factor] * normalised_weights[factor] for factor in valid_weighted_factors
    ) / available_weight_total
else:
    relative_score = np.nan

score_col, factor_col = st.columns([1, 1.6], gap="large")
with score_col:
    with st.container(border=True):
        st.subheader("Relative profile score")
        st.metric("Sample-relative score", f"{relative_score:.1f}/100" if pd.notna(relative_score) else "N/A")
        st.caption(
            "A weighted average of within-sample percentile ranks for the selected indicators. "
            "It is not a probability of return or a measure of stock-market risk."
        )
        if weight_total > 0:
            st.caption(f"Entered weights total {weight_total}%; normalised to 100% for scoring.")
with factor_col:
    with st.container(border=True):
        st.subheader("Indicator contribution view")
        factor_display = pd.DataFrame({
            "KPI": list(factor_scores.keys()),
            "Relative percentile score": [
                round(factor_scores[key], 1) if pd.notna(factor_scores[key]) else np.nan
                for key in factor_scores
            ],
            "Normalised weight (%)": [
                round(normalised_weights.get(key, 0) * 100, 1) for key in factor_scores
            ],
        })
        st.dataframe(factor_display, use_container_width=True, hide_index=True)
        st.caption(
            "Higher percentile scores indicate a higher relative rank under the chosen direction. "
            "For historical stability, lower ROA volatility receives a higher score."
        )

st.subheader("Risk signals to review")
risk_notes = []
if len(filtered_df) < 30:
    risk_notes.append("The selected sample has fewer than 30 company-year observations; treat sample-relative rankings cautiously.")
if filtered_df["name"].nunique() < 10:
    risk_notes.append("Fewer than 10 companies are represented in the selected sample; peer comparisons may be unstable.")
if filtered_df["roa"].notna().mean() < 0.8:
    risk_notes.append("ROA is missing for more than 20% of observations; verify data completeness before comparing operating performance.")
if filtered_df["tobinq"].notna().mean() < 0.8:
    risk_notes.append("Tobin's Q is missing for more than 20% of observations; verify valuation and balance-sheet inputs.")
if filtered_df["rd_intensity"].notna().mean() < 0.8:
    risk_notes.append("R&D intensity is missing for more than 20% of observations; comparisons may be affected by incomplete revenue or R&D data.")
if (filtered_df["net_profit"].dropna() < 0).mean() > 0.25 if filtered_df["net_profit"].notna().any() else False:
    risk_notes.append("More than 25% of available observations have negative net profit; examine profitability pressure and persistence.")
if not company_roa_std.empty:
    high_volatility_cutoff = company_roa_std.quantile(0.75)
    if pd.notna(high_volatility_cutoff):
        volatile_companies = int((company_roa_std >= high_volatility_cutoff).sum())
        risk_notes.append(
            f"Historical ROA volatility is available for {len(company_roa_std)} companies; "
            f"{volatile_companies} are at or above the selected-sample 75th percentile of ROA volatility."
        )
else:
    risk_notes.append("There are not enough multi-year ROA observations to assess historical operating stability.")

if risk_profile == "Conservative":
    risk_notes.append("Conservative profile: review operating stability, missing data, and valuation context before drawing conclusions.")
elif risk_profile == "Aggressive":
    risk_notes.append("Aggressive profile: innovation indicators receive more weight, but high R&D or patent counts do not guarantee commercial success.")
else:
    risk_notes.append("Balanced profile: check whether operating performance and innovation indicators tell a consistent story.")

for note in risk_notes:
    st.markdown(f"- {note}")

st.info(
    "Interpretation boundary: the score is relative to the currently selected sample and changes when the industry filter or weights change. "
    "Tobin's Q and innovation indicators are not universally 'better' when higher; use the score as a screening aid, not a standalone investment decision."
)

# ---------- AI analyst ----------
st.markdown('<div class="section-kicker">07 / AI research assistant</div>', unsafe_allow_html=True)
st.header("AI Investment Analyst")
st.markdown(
    '<div class="ai-panel"><b>From metrics to interpretation</b><br>'
    '<span style="color:#63748b">Generate a structured summary of the selected sample, innovation inputs, '
    'and model limitations. AI output is decision support, not a stock recommendation.</span></div>',
    unsafe_allow_html=True,
)
with st.expander("What will the AI analyze?"):
    st.markdown(
        "- Selected industry and historical sample size\n"
        "- Median R&D intensity, ROA and Tobin's Q\n"
        "- Your R&D intensity and patent-count inputs\n"
        "- Risks, missing information, and follow-up research questions\n\n"
        "The AI receives summary statistics rather than the full company-level dataset. "
        "Patent count is a user-provided input and is not included in the current OLS model."
    )

if st.button("✨ Generate AI analysis", type="primary"):
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        st.error("OPENROUTER_API_KEY is not configured. Check the .env file in the project root.")
    else:
        try:
            from openai import OpenAI

            valid_rd = filtered_plot_df["rd_intensity"].dropna()
            valid_roa = filtered_df["roa"].dropna()
            valid_q = filtered_df["tobinq"].dropna()
            summary = {
                "selected_industry": selected_industry,
                "observations": int(len(filtered_df)),
                "companies": int(filtered_df["name"].nunique()),
                "median_rd_intensity_pct": round(float(valid_rd.median() * 100), 2) if not valid_rd.empty else None,
                "median_roa_pct": round(float(valid_roa.median() * 100), 2) if not valid_roa.empty else None,
                "median_tobins_q": round(float(valid_q.median()), 3) if not valid_q.empty else None,
                "user_rd_intensity_pct": float(user_rd),
                "user_patents_granted": int(user_patent),
                "estimated_tobins_q_from_two_input_ols": round(prediction_q, 4) if prediction_q is not None else None,
                "two_input_model_r_squared": round(float(prediction_model.rsquared), 4) if prediction_model is not None else None,
                "period": [int(df["year"].min()), int(df["year"].max())] if df["year"].notna().any() else None,
            }
            system_prompt = (
                "You are an investment research assistant for a university project. "
                "Use only supplied statistics; do not invent facts or market data. "
                "Distinguish descriptive comparisons from causal claims. Do not give buy/sell advice "
                "or promise future returns. Explicitly state that patents are user-provided and not "
                "included in the current OLS model. Respond in clear English with headings: "
                "1) Sample Snapshot, 2) Innovation Investment Interpretation, "
                "3) Potential Risks and Limitations, 4) Questions for Further Analysis."
            )
            model_name = os.getenv("OPENROUTER_MODEL", "openrouter/free")
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={"X-Title": "BA6003 Corporate Innovation Dashboard"},
            )
            with st.spinner("Analyzing the selected sample..."):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Analyze this dashboard context:\n" + str(summary)},
                    ],
                    temperature=0.2,
                )
            answer = response.choices[0].message.content
            if answer:
                st.markdown(answer)
            else:
                st.warning("The AI provider returned an empty response.")
            logging.info("AI analysis generated | industry=%s | observations=%s | model=%s", selected_industry, len(filtered_df), model_name)
            st.caption("AI-generated interpretation. Verify important claims against the underlying data; this is not financial advice.")
        except ImportError:
            st.error("Required package missing. Run `uv add openai python-dotenv` in the project terminal.")
        except Exception as exc:
            logging.exception("AI analysis failed")
            st.error("AI analysis failed. The detailed provider response is shown below.")
            st.code(f"{type(exc).__name__}: {exc}")

# ---------- Data explorer and project notes ----------
st.markdown('<div class="section-kicker">08 / Transparency</div>', unsafe_allow_html=True)
st.header("Data explorer")
with st.expander("Preview selected records"):
    st.dataframe(filtered_df.head(50), use_container_width=True, hide_index=True)

with st.expander("About this prototype and its limitations"):
    st.markdown(
        """
        - **Target users:** Investment analysts and investors conducting preliminary research.
        - **Current model:** Historical OLS regression between R&D intensity and Tobin's Q.
        - **Key limitation:** The current model is descriptive and does not predict next-year performance.
        - **Patent variable:** Captured as an input for contextual discussion, but not included in the OLS model.
        - **Potential next step:** Test lagged R&D intensity, company characteristics, year and industry controls,
          and out-of-sample validation before making predictive claims.
        """
    )

st.caption("Innovation IQ · BA6003 Group 5 · Educational decision-support prototype")
