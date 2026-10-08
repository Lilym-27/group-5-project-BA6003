import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Corporate Innovation Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Corporate Innovation Investment Dashboard")

st.markdown(
    """
This dashboard helps users explore the relationship between
corporate innovation investment and company performance.

The current prototype uses historical company-level data to examine
R&D investment intensity and Tobin's Q.
"""
)

st.info(
    "Note: This is an initial decision-support prototype. "
    "The current model shows historical relationships and is not a guarantee "
    "of future investment returns."
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    file_path = "Sample.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Sample.xlsx was not found in the same folder as app.py."
        )

    df = pd.read_excel(file_path)

    return df


try:

    df = load_data()

except Exception as e:

    st.error("❌ Unable to load Sample.xlsx")

    st.write(
        "Please make sure that Sample.xlsx is in the same folder as app.py."
    )

    st.exception(e)

    st.stop()


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "name",
    "year",
    "revenue",
    "rd_expense",
    "net_profit",
    "total_assets",
    "market_value",
    "total_liabilities",
    "patent",
    "industry"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error("❌ Some required columns are missing.")

    st.write("Missing columns:")

    for col in missing_columns:
        st.write(f"- {col}")

    st.write("Available columns in your Excel file:")

    st.write(list(df.columns))

    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

numeric_columns = [
    "year",
    "revenue",
    "rd_expense",
    "net_profit",
    "total_assets",
    "market_value",
    "total_liabilities",
    "patent"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# CALCULATE VARIABLES
# =========================================================

# R&D Intensity

df["rd_intensity"] = np.where(
    df["revenue"] != 0,
    df["rd_expense"] / df["revenue"],
    np.nan
)


# ROA

df["roa"] = np.where(
    df["total_assets"] != 0,
    df["net_profit"] / df["total_assets"],
    np.nan
)


# Tobin's Q

df["tobinq"] = np.where(
    df["total_assets"] != 0,
    (
        df["market_value"]
        + df["total_liabilities"]
    ) / df["total_assets"],
    np.nan
)


# =========================================================
# REMOVE EXTREME / INVALID VALUES FOR VISUALIZATION
# =========================================================

plot_df = df.copy()

plot_df = plot_df[
    plot_df["rd_intensity"].notna()
]

plot_df = plot_df[
    plot_df["rd_intensity"] >= 0
]

plot_df = plot_df[
    plot_df["rd_intensity"] <= 0.50
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Controls")


# Industry selection

industry_values = (
    df["industry"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

industry_values = sorted(industry_values)


if len(industry_values) > 0:

    selected_industry = st.sidebar.selectbox(
        "Select Industry",
        ["All Industries"] + industry_values
    )

else:

    selected_industry = "All Industries"


# R&D intensity input

user_rd = st.sidebar.number_input(
    "R&D Intensity (%)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5
)


# Patent input

user_patent = st.sidebar.number_input(
    "Patent Applications",
    min_value=0,
    value=10,
    step=1
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Patents are currently displayed as an input variable, "
    "but are not included in the current OLS model."
)


# =========================================================
# FILTER DATA BY INDUSTRY
# =========================================================

if selected_industry == "All Industries":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["industry"].astype(str)
        == selected_industry
    ].copy()


# =========================================================
# HEADER INFORMATION
# =========================================================

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Observations",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Companies",
        f"{df['name'].nunique():,}"
    )

with col3:

    if len(df) > 0:

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

        year_text = f"{min_year} - {max_year}"

    else:

        year_text = "N/A"

    st.metric(
        "Data Period",
        year_text
    )

with col4:

    st.metric(
        "Industries",
        f"{df['industry'].nunique():,}"
    )


# =========================================================
# KPI SECTION
# =========================================================

st.subheader("Key Performance Indicators")


kpi1, kpi2, kpi3, kpi4 = st.columns(4)


# Median R&D

median_rd = filtered_df["rd_intensity"].median()

with kpi1:

    if pd.notna(median_rd):

        st.metric(
            "Median R&D Intensity",
            f"{median_rd * 100:.2f}%"
        )

    else:

        st.metric(
            "Median R&D Intensity",
            "N/A"
        )


# Median ROA

median_roa = filtered_df["roa"].median()

with kpi2:

    if pd.notna(median_roa):

        st.metric(
            "Median ROA",
            f"{median_roa * 100:.2f}%"
        )

    else:

        st.metric(
            "Median ROA",
            "N/A"
        )


# Median Tobin Q

median_tobinq = filtered_df["tobinq"].median()

with kpi3:

    if pd.notna(median_tobinq):

        st.metric(
            "Median Tobin's Q",
            f"{median_tobinq:.2f}"
        )

    else:

        st.metric(
            "Median Tobin's Q",
            "N/A"
        )


# Selected R&D

with kpi4:

    st.metric(
        "User R&D Input",
        f"{user_rd:.1f}%"
    )


# =========================================================
# R&D DISTRIBUTION
# =========================================================

st.markdown("---")

st.subheader("1. R&D Intensity Distribution")

rd_plot = plot_df["rd_intensity"]


if len(rd_plot) > 0:

    rd_groups = pd.DataFrame(
        {
            "R&D Intensity Range": [
                "0% - 5%",
                "5% - 10%",
                "10% - 20%",
                "20% - 30%",
                "30% - 40%",
                "40% - 50%"
            ],

            "Number of Companies": [
                (
                    (rd_plot >= 0.00)
                    & (rd_plot < 0.05)
                ).sum(),

                (
                    (rd_plot >= 0.05)
                    & (rd_plot < 0.10)
                ).sum(),

                (
                    (rd_plot >= 0.10)
                    & (rd_plot < 0.20)
                ).sum(),

                (
                    (rd_plot >= 0.20)
                    & (rd_plot < 0.30)
                ).sum(),

                (
                    (rd_plot >= 0.30)
                    & (rd_plot < 0.40)
                ).sum(),

                (
                    (rd_plot >= 0.40)
                    & (rd_plot <= 0.50)
                ).sum()
            ]
        }
    )

    st.bar_chart(
        rd_groups.set_index(
            "R&D Intensity Range"
        )
    )

else:

    st.warning(
        "No valid R&D intensity data available."
    )


# =========================================================
# R&D VS TOBIN'S Q
# =========================================================

st.markdown("---")

st.subheader(
    "2. R&D Intensity and Tobin's Q"
)

scatter_df = plot_df[
    [
        "rd_intensity",
        "tobinq"
    ]
].dropna()


if len(scatter_df) >= 2:

    # Create a simple regression line

    X = sm.add_constant(
        scatter_df["rd_intensity"]
    )

    y = scatter_df["tobinq"]

    try:

        model = sm.OLS(
            y,
            X
        ).fit()

        intercept = model.params.iloc[0]
        slope = model.params.iloc[1]

        scatter_df = scatter_df.copy()

        scatter_df["Predicted Tobin's Q"] = (
            intercept
            + slope * scatter_df["rd_intensity"]
        )

        scatter_chart_df = scatter_df[
            [
                "rd_intensity",
                "tobinq",
                "Predicted Tobin's Q"
            ]
        ].copy()

        scatter_chart_df = (
            scatter_chart_df
            .sort_values("rd_intensity")
            .reset_index(drop=True)
        )

        scatter_chart_df = scatter_chart_df.rename(
            columns={
                "rd_intensity": "R&D Intensity",
                "tobinq": "Tobin's Q"
            }
        )

        scatter_chart_df["R&D Intensity"] = (
            scatter_chart_df["R&D Intensity"] * 100
        )

        st.line_chart(
            scatter_chart_df.set_index(
                "R&D Intensity"
            )
        )

        st.caption(
            "The chart shows the historical relationship between "
            "R&D intensity and Tobin's Q in the available sample."
        )

    except Exception as e:

        st.warning(
            "Regression chart could not be generated."
        )

        st.exception(e)

else:

    st.warning(
        "Not enough valid observations for the regression chart."
    )


# =========================================================
# R&D TREND
# =========================================================

st.markdown("---")

st.subheader(
    "3. R&D Intensity Over Time"
)


trend_df = (
    df[
        df["rd_intensity"].notna()
    ]
    .groupby("year")["rd_intensity"]
    .median()
    .reset_index()
)


trend_df = trend_df[
    trend_df["year"].notna()
]


if len(trend_df) > 0:

    trend_df["R&D Intensity"] = (
        trend_df["rd_intensity"] * 100
    )

    trend_df = trend_df[
        [
            "year",
            "R&D Intensity"
        ]
    ]

    trend_df = trend_df.sort_values(
        "year"
    )

    st.line_chart(
        trend_df.set_index("year")
    )

else:

    st.warning(
        "No time-series data available."
    )


# =========================================================
# INDUSTRY BENCHMARK
# =========================================================

st.markdown("---")

st.subheader(
    "4. Industry Benchmark"
)


industry_benchmark = (
    df[
        df["rd_intensity"].notna()
    ]
    .groupby("industry")["rd_intensity"]
    .median()
    .reset_index()
)


industry_benchmark["R&D Intensity"] = (
    industry_benchmark["rd_intensity"] * 100
)


industry_benchmark = industry_benchmark[
    [
        "industry",
        "R&D Intensity"
    ]
]


industry_benchmark = (
    industry_benchmark
    .sort_values(
        "R&D Intensity",
        ascending=False
    )
    .head(15)
)


if len(industry_benchmark) > 0:

    st.bar_chart(
        industry_benchmark.set_index(
            "industry"
        )
    )

else:

    st.warning(
        "No industry benchmark data available."
    )


# =========================================================
# USER INPUT INTERPRETATION
# =========================================================

st.markdown("---")

st.subheader(
    "5. User Input Interpretation"
)


overall_median_rd = df[
    "rd_intensity"
].median()


if pd.notna(overall_median_rd):

    if user_rd / 100 > overall_median_rd:

        rd_message = (
            f"The entered R&D intensity of {user_rd:.1f}% "
            f"is above the overall sample median "
            f"of {overall_median_rd * 100:.2f}%."
        )

    elif user_rd / 100 < overall_median_rd:

        rd_message = (
            f"The entered R&D intensity of {user_rd:.1f}% "
            f"is below the overall sample median "
            f"of {overall_median_rd * 100:.2f}%."
        )

    else:

        rd_message = (
            f"The entered R&D intensity of {user_rd:.1f}% "
            f"is approximately equal to the sample median."
        )

    st.write(rd_message)

else:

    st.write(
        "The sample median could not be calculated."
    )


st.write(
    f"Patent applications entered: {user_patent:,}"
)


# =========================================================
# REGRESSION RESULTS
# =========================================================

st.markdown("---")

st.subheader(
    "6. Statistical Model"
)


reg_df = df[
    [
        "tobinq",
        "rd_intensity"
    ]
].dropna()


# Remove extreme R&D intensity

reg_df = reg_df[
    (reg_df["rd_intensity"] >= 0)
    &
    (reg_df["rd_intensity"] <= 0.50)
]


if len(reg_df) >= 10:

    try:

        X = sm.add_constant(
            reg_df["rd_intensity"]
        )

        y = reg_df["tobinq"]

        regression_model = sm.OLS(
            y,
            X
        ).fit()

        beta = regression_model.params.iloc[1]

        r_squared = (
            regression_model.rsquared
        )

        p_value = (
            regression_model.pvalues.iloc[1]
        )


        model_col1, model_col2, model_col3 = (
            st.columns(3)
        )


        with model_col1:

            st.metric(
                "R&D Coefficient",
                f"{beta:.4f}"
            )


        with model_col2:

            st.metric(
                "R-squared",
                f"{r_squared:.4f}"
            )


        with model_col3:

            st.metric(
                "P-value",
                f"{p_value:.4f}"
            )


        st.markdown(
            """
            **Model specification**

            Tobin's Q = α + β × R&D Intensity + ε
            """
        )


        st.caption(
            "This OLS model describes an association in the historical "
            "sample. It should not be interpreted as proof of causality."
        )


    except Exception as e:

        st.warning(
            "The regression model could not be estimated."
        )

        st.exception(e)

else:

    st.warning(
        "There are not enough observations for the regression model."
    )


# =========================================================
# DATA PREVIEW
# =========================================================

st.markdown("---")

st.subheader(
    "7. Data Preview"
)


with st.expander(
    "Show raw data"
):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown("---")

st.subheader(
    "About This Prototype"
)

st.write(
    """
    **Target users:** Investment analysts and investors.

    **Main purpose:** Support users in evaluating the relationship between
    corporate innovation investment and company performance.

    **Current indicators:** R&D intensity, ROA and Tobin's Q.

    **Current model:** Historical OLS regression between R&D intensity
    and Tobin's Q.

    **Data sources:** Company-level financial and innovation information.

    **Next development stage:** The model can be extended to predict
    next-year ROIC and Tobin's Q using lagged innovation variables,
    company characteristics, industry information and other controls.
    """
)


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    "Investment Disclaimer: This dashboard is an analytical and "
    "educational prototype. Historical relationships do not guarantee "
    "future company performance or investment returns."
)