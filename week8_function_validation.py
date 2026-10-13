
# %% ==================================================
# Import packages
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import logging


# %% ==================================================
# Logging setup
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# %% ==================================================
# 1. Load Data
# ==================================================

def load_data(path):

    logging.info("Loading dataset...")

    df = pd.read_excel(path)

    logging.info(
        f"Dataset loaded: {len(df)} observations"
    )

    return df


# %% ==================================================
# 2. Validate Raw Data
# ==================================================

def validate_data(df):

    logging.info("Running basic data validation...")

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

    # Check required columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    assert not missing_columns, (
        f"Missing required columns: {missing_columns}"
    )

    # Check dataset is not empty
    assert len(df) > 0, (
        "Dataset is empty"
    )

    print("Basic data validation passed.")

    logging.info(
        f"Number of observations: {len(df)}"
    )


# %% ==================================================
# 3. Calculate Variables
# ==================================================

def calculate_variables(df):

    logging.info(
        "Calculating derived variables..."
    )

    # R&D intensity
    df["rd_intensity"] = (
        df["rd_expense"] /
        df["revenue"]
    )

    # ROA
    df["roa"] = (
        df["net_profit"] /
        df["total_assets"]
    )

    # Tobin's Q
    df["tobinq"] = (
        df["market_value"] +
        df["total_liabilities"]
    ) / df["total_assets"]

    return df


# %% ==================================================
# 4. Validate Calculated Variables
# ==================================================

def validate_calculations(df):

    logging.info(
        "Validating calculated variables..."
    )

    # --------------------------------------------------
    # R&D intensity
    # --------------------------------------------------

    expected_rd = (
        df["rd_expense"] /
        df["revenue"]
    )

    valid_rd = (
        df["rd_intensity"].notna() &
        expected_rd.notna()
    )

    assert np.allclose(
        df.loc[valid_rd, "rd_intensity"],
        expected_rd[valid_rd],
        equal_nan=True
    ), "R&D intensity calculation failed"

    # --------------------------------------------------
    # ROA
    # --------------------------------------------------

    expected_roa = (
        df["net_profit"] /
        df["total_assets"]
    )

    valid_roa = (
        df["roa"].notna() &
        expected_roa.notna()
    )

    assert np.allclose(
        df.loc[valid_roa, "roa"],
        expected_roa[valid_roa],
        equal_nan=True
    ), "ROA calculation failed"

    # --------------------------------------------------
    # Tobin's Q
    # --------------------------------------------------

    expected_tobinq = (
        df["market_value"] +
        df["total_liabilities"]
    ) / df["total_assets"]

    valid_tobinq = (
        df["tobinq"].notna() &
        expected_tobinq.notna()
    )

    assert np.allclose(
        df.loc[valid_tobinq, "tobinq"],
        expected_tobinq[valid_tobinq],
        equal_nan=True
    ), "Tobin's Q calculation failed"

    print(
        "All calculation validations passed."
    )

    logging.info(
        "R&D intensity, ROA and Tobin's Q calculations validated"
    )


# %% ==================================================
# 5. Validate Data Quality
# ==================================================

def validate_data_quality(df):

    logging.info(
        "Checking data quality..."
    )

    # --------------------------------------------------
    # Extreme R&D intensity
    # --------------------------------------------------

    extreme_rd_50 = (
        df["rd_intensity"] > 0.5
    ).sum()

    extreme_rd_100 = (
        df["rd_intensity"] > 1
    ).sum()

    print(
        f"R&D intensity > 50%: {extreme_rd_50}"
    )

    print(
        f"R&D intensity > 100%: {extreme_rd_100}"
    )

    assert (
        extreme_rd_50 >= extreme_rd_100
    )

    logging.info(
        f"Extreme R&D intensity (>50%): "
        f"{extreme_rd_50}"
    )

    logging.info(
        f"Extreme R&D intensity (>100%): "
        f"{extreme_rd_100}"
    )

    # --------------------------------------------------
    # R&D availability by year
    # --------------------------------------------------

    rd_by_year = (
        df.groupby("year")["rd_expense"]
        .count()
    )

    print(
        "R&D expenditure observations by year:"
    )

    print(rd_by_year)

    logging.info(
        "R&D data availability checked"
    )


# %% ==================================================
# 6. Create Visualizations
# ==================================================

def create_visualizations(df):

    logging.info(
        "Creating visualizations..."
    )

    # --------------------------------------------------
    # Patent Distribution
    # --------------------------------------------------

    df["patent"].hist()

    plt.xlabel(
        "Number of Patents Granted"
    )

    plt.ylabel(
        "Number of Observations"
    )

    plt.title(
        "Distribution of Patents Granted"
    )

    plt.xlim(0, 200)

    plt.xticks(
        range(0, 201, 25)
    )

    plt.savefig(
        "charts/patent_distribution.png"
    )

    plt.show()

    # --------------------------------------------------
    # R&D Intensity Distribution
    # --------------------------------------------------

    rd_plot = df[
        df["rd_intensity"] <= 0.5
    ].copy()

    rd_plot["rd_intensity"].hist()

    plt.xlabel(
        "R&D Intensity"
    )

    plt.ylabel(
        "Number of Observations"
    )

    plt.title(
        "Distribution of R&D Intensity"
    )

    plt.xlim(0, 0.5)

    plt.xticks(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    plt.savefig(
        "charts/rd_intensity_distribution.png"
    )

    plt.show()

    # --------------------------------------------------
    # R&D Intensity vs ROA
    # --------------------------------------------------

    plot_data = df[
        (df["rd_intensity"] <= 0.5) &
        (df["roa"] >= -0.2) &
        (df["roa"] <= 0.3)
    ].copy()

    plt.scatter(
        plot_data["rd_intensity"],
        plot_data["roa"],
        alpha=0.5
    )

    plt.xlabel(
        "R&D Intensity"
    )

    plt.ylabel(
        "ROA"
    )

    plt.title(
        "R&D Intensity vs ROA"
    )

    plt.savefig(
        "charts/rd_intensity_vs_roa.png"
    )

    plt.show()

    # --------------------------------------------------
    # R&D Intensity Over Time
    # --------------------------------------------------

    yearly_rd = (
        df[df["year"] >= 2016]
        .groupby("year")["rd_intensity"]
        .median()
    )

    plt.plot(
        yearly_rd.index,
        yearly_rd.values,
        marker="o"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Median R&D Intensity"
    )

    plt.title(
        "Median R&D Intensity Over Time"
    )

    plt.xticks(
        yearly_rd.index
    )

    plt.savefig(
        "charts/rd_intensity_over_time.png"
    )

    plt.show()

    # --------------------------------------------------
    # R&D Intensity by Industry
    # --------------------------------------------------

    industry_rd = (
        df.groupby("industry")["rd_intensity"]
        .median()
        .sort_values()
    )

    industry_rd.plot(
        kind="barh"
    )

    plt.xlabel(
        "Median R&D Intensity"
    )

    plt.ylabel(
        "Industry"
    )

    plt.title(
        "Median R&D Intensity by Industry"
    )

    plt.savefig(
        "charts/rd_intensity_by_industry.png"
    )

    plt.show()

    # --------------------------------------------------
    # Tobin's Q Distribution
    # --------------------------------------------------

    tobinq_plot = df[
        df["tobinq"] <= 10
    ].copy()

    tobinq_plot["tobinq"].hist()

    plt.xlabel(
        "Tobin's Q"
    )

    plt.ylabel(
        "Number of Observations"
    )

    plt.title(
        "Distribution of Tobin's Q"
    )

    plt.xlim(0, 10)

    plt.xticks(
        range(0, 11, 1)
    )

    plt.savefig(
        "charts/tobinq_distribution.png"
    )

    plt.show()


# %% ==================================================
# 7. Prepare Regression Data
# ==================================================

def prepare_regression_data(df):

    logging.info(
        "Preparing regression dataset..."
    )

    # Exclude ST / *ST companies

    reg_data = df[
        ~df["name"].astype(str).str.contains(
            "ST",
            case=False,
            na=False
        )
    ].copy()

    # Keep required variables
    # and remove missing values

    reg_data = reg_data[
        ["tobinq", "rd_intensity"]
    ].dropna()

    # Exclude extreme R&D intensity

    reg_data = reg_data[
        reg_data["rd_intensity"] <= 0.5
    ]

    # --------------------------------------------------
    # Regression validation
    # --------------------------------------------------

    assert len(reg_data) > 0, (
        "Regression dataset is empty"
    )

    assert (
        reg_data["rd_intensity"].isna().sum() == 0
    ), "Missing R&D intensity remains"

    assert (
        reg_data["tobinq"].isna().sum() == 0
    ), "Missing Tobin's Q remains"

    assert (
        reg_data["rd_intensity"] <= 0.5
    ).all(), (
        "Extreme R&D intensity remains"
    )

    print(
        "Regression dataset validation passed."
    )

    print(
        "Number of observations:",
        len(reg_data)
    )

    logging.info(
        f"Regression sample size: {len(reg_data)}"
    )

    return reg_data


# %% ==================================================
# 8. Run OLS Regression
# ==================================================

def run_regression(reg_data):

    logging.info(
        "Running OLS regression..."
    )

    X = sm.add_constant(
        reg_data["rd_intensity"]
    )

    y = reg_data["tobinq"]

    model = sm.OLS(
        y,
        X
    ).fit()

    print(
        model.summary()
    )

    logging.info(
        "OLS regression completed"
    )

    return model


# %% ==================================================
# 9. Regression Visualization
# ==================================================

def plot_regression(reg_data, model):

    plt.scatter(
        reg_data["rd_intensity"],
        reg_data["tobinq"],
        alpha=0.4
    )

    # Regression line

    x_line = np.linspace(
        reg_data["rd_intensity"].min(),
        reg_data["rd_intensity"].max(),
        100
    )

    y_line = (
        model.params["const"] +
        model.params["rd_intensity"] *
        x_line
    )

    plt.plot(
        x_line,
        y_line
    )

    plt.xlabel(
        "R&D Intensity"
    )

    plt.ylabel(
        "Tobin's Q"
    )

    plt.title(
        "R&D Intensity vs Tobin's Q"
    )

    # Visualization range only

    plt.xlim(0, 0.5)

    plt.ylim(0, 10)

    plt.savefig(
        "charts/rd_intensity_vs_tobinq.png"
    )

    plt.show()


# %% ==================================================
# 10. Main Pipeline
# ==================================================

def main():

    logging.info(
        "Pipeline started"
    )

    # Load data
    df = load_data(
        "Sample.xlsx"
    )

    # Validate raw data
    validate_data(df)

    # Calculate variables
    df = calculate_variables(df)

    # Validate calculations
    validate_calculations(df)

    # Validate data quality
    validate_data_quality(df)

    # Create exploratory visualizations
    create_visualizations(df)

    # Prepare regression data
    reg_data = prepare_regression_data(df)

    # Run regression
    model = run_regression(
        reg_data
    )

    # Plot regression
    plot_regression(
        reg_data,
        model
    )

    logging.info(
        "Pipeline completed successfully"
    )

    return df, reg_data, model


# %% ==================================================
# Run Pipeline
# ==================================================

df, reg_data, model = main()
# %%
