# %% --------------------------------------------------
# Import packages and load data
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_excel("Sample.xlsx")

df.info()

df.isna().sum()

df.describe()


# %% --------------------------------------------------
# 1. Patent Distribution
# --------------------------------------------------

df["patent"].hist()

plt.xlabel("Number of Patents Granted")
plt.ylabel("Number of Observations")
plt.title("Distribution of Patents Granted")

plt.xlim(0, 200)
plt.xticks(range(0, 201, 25))

plt.savefig("charts/patent_distribution.png")
plt.show()


# %% --------------------------------------------------
# 2. R&D Intensity
# --------------------------------------------------

df["rd_intensity"] = df["rd_expense"] / df["revenue"]

df["rd_intensity"].describe()

df["revenue"].describe()

df[
    ["name", "year", "revenue", "rd_expense", "rd_intensity"]
].sort_values(
    "rd_intensity",
    ascending=False
).head(10)


# %% --------------------------------------------------
# Check extreme R&D intensity values
# --------------------------------------------------

(df["rd_intensity"] > 1).sum()

(df["rd_intensity"] > 0.5).sum()


# %% --------------------------------------------------
# 3. R&D Intensity Distribution
# --------------------------------------------------

rd_intensity_plot = df[
    df["rd_intensity"] <= 0.5
]

rd_intensity_plot["rd_intensity"].hist()

plt.xlabel("R&D Intensity")
plt.ylabel("Number of Observations")
plt.title("Distribution of R&D Intensity")

plt.xlim(0, 0.5)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

plt.savefig("charts/rd_intensity_distribution.png")
plt.show()


# %% --------------------------------------------------
# 4. ROA
# --------------------------------------------------

df["roa"] = df["net_profit"] / df["total_assets"]

df["roa"].describe()

df[
    ["name", "year", "total_assets", "net_profit", "roa"]
].sort_values(
    "roa"
).head(10)


# %% --------------------------------------------------
# 5. R&D Intensity vs ROA
# --------------------------------------------------

plot_data = df[
    (df["rd_intensity"] <= 0.5) &
    (df["roa"] >= -0.2) &
    (df["roa"] <= 0.3)
]

plt.scatter(
    plot_data["rd_intensity"],
    plot_data["roa"],
    alpha=0.5
)

plt.xlabel("R&D Intensity")
plt.ylabel("ROA")
plt.title("R&D Intensity vs ROA")

plt.savefig("charts/rd_intensity_vs_roa.png")
plt.show()


# %% --------------------------------------------------
# Correlation: full sample
# --------------------------------------------------

df[["rd_intensity", "roa"]].corr()


# %% --------------------------------------------------
# Correlation: filtered visualization sample
# --------------------------------------------------

plot_data[["rd_intensity", "roa"]].corr()


# %% --------------------------------------------------
# 6. R&D Intensity Over Time
# --------------------------------------------------

yearly_rd = df.groupby("year")["rd_intensity"].median()

yearly_rd


# %% --------------------------------------------------
# Check availability of R&D expenditure by year
# --------------------------------------------------

df.groupby("year")["rd_expense"].count()


# %% --------------------------------------------------
# Median R&D Intensity Over Time
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

plt.xlabel("Year")
plt.ylabel("Median R&D Intensity")
plt.title("Median R&D Intensity Over Time")

plt.xticks(yearly_rd.index)

plt.savefig("charts/rd_intensity_over_time.png")
plt.show()


# %% --------------------------------------------------
# 7. R&D Intensity by Industry
# --------------------------------------------------

df["industry"].value_counts()


# %% --------------------------------------------------
# Median R&D Intensity by Industry
# --------------------------------------------------

industry_rd = df.groupby("industry")["rd_intensity"].median()

industry_rd


# %% --------------------------------------------------
# Plot R&D Intensity by Industry
# --------------------------------------------------

industry_rd = industry_rd.sort_values()

industry_rd.plot(kind="barh")

plt.xlabel("Median R&D Intensity")
plt.ylabel("Industry")
plt.title("Median R&D Intensity by Industry")

plt.savefig("charts/rd_intensity_by_industry.png")
plt.show()


# %% --------------------------------------------------
# 8. Identify ST / *ST Companies
# --------------------------------------------------

df[
    df["name"].astype(str).str.contains(
        "ST",
        case=False,
        na=False
    )
][
    ["code", "name", "year", "industry"]
].head(20)


# %% --------------------------------------------------
# 9. Calculate Tobin's Q
# --------------------------------------------------

df["tobinq"] = (
    df["market_value"] +
    df["total_liabilities"]
) / df["total_assets"]

df["tobinq"].describe()


# %% --------------------------------------------------
# Check highest Tobin's Q observations
# --------------------------------------------------

df.sort_values(
    "tobinq",
    ascending=False
)[
    [
        "code",
        "name",
        "year",
        "market_value",
        "total_liabilities",
        "total_assets",
        "tobinq"
    ]
].head(10)


# %% --------------------------------------------------
# 10. Tobin's Q Distribution
# --------------------------------------------------

tobinq_plot = df[
    df["tobinq"] <= 10
]

tobinq_plot["tobinq"].hist()

plt.xlabel("Tobin's Q")
plt.ylabel("Number of Observations")
plt.title("Distribution of Tobin's Q")

plt.xlim(0, 10)
plt.xticks(range(0, 11, 1))

plt.savefig("charts/tobinq_distribution.png")
plt.show()


# %% --------------------------------------------------
# 11. OLS Regression:
# R&D Intensity and Tobin's Q
# --------------------------------------------------

# Exclude ST / *ST companies
reg_data = df[
    ~df["name"].astype(str).str.contains(
        "ST",
        case=False,
        na=False
    )
].copy()

# Keep regression variables and remove missing values
reg_data = reg_data[
    ["tobinq", "rd_intensity"]
].dropna()

# Exclude extreme R&D intensity values
reg_data = reg_data[
    reg_data["rd_intensity"] <= 0.5
]

print(
    "Number of observations:",
    len(reg_data)
)


# %% --------------------------------------------------
# Run OLS regression
# --------------------------------------------------

X = sm.add_constant(
    reg_data["rd_intensity"]
)

y = reg_data["tobinq"]

model = sm.OLS(
    y,
    X
).fit()

print(model.summary())


# %% --------------------------------------------------
# 12. R&D Intensity vs Tobin's Q
# --------------------------------------------------

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
    model.params["rd_intensity"] * x_line
)

plt.plot(
    x_line,
    y_line
)

plt.xlabel("R&D Intensity")
plt.ylabel("Tobin's Q")
plt.title("R&D Intensity vs Tobin's Q")

# Visualization range only
plt.xlim(0, 0.5)
plt.ylim(0, 10)

plt.savefig(
    "charts/rd_intensity_vs_tobinq.png"
)

plt.show()