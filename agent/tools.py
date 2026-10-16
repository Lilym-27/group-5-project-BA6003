"""
Agent tools for the Innovation Investment COO Report.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

# ── Global data holder (set by streamlit_app.py after loading) ─────────────
df_global: pd.DataFrame = None


def get_portfolio_overview(industry: str | None = None) -> dict:
    if df_global is None:
        return {"error": "Data has not been loaded yet."}
    df = df_global.copy()
    if industry and industry != "All Industries":
        df = df[df["industry"].astype(str) == industry]
        if df.empty:
            return {"error": f"No data for industry '{industry}'."}
    valid_rd = df["rd_intensity"].dropna()
    valid_roa = df["roa"].dropna()
    valid_q = df["tobinq"].dropna()
    result = {
        "industry_filter": industry or "All Industries",
        "observations": int(len(df)),
        "companies": int(df["name"].nunique()),
        "year_range": [
            int(df["year"].min()) if df["year"].notna().any() else None,
            int(df["year"].max()) if df["year"].notna().any() else None,
        ],
        "median_rd_intensity_pct": round(float(valid_rd.median() * 100), 3) if not valid_rd.empty else None,
        "median_roa_pct": round(float(valid_roa.median() * 100), 3) if not valid_roa.empty else None,
        "median_tobins_q": round(float(valid_q.median()), 3) if not valid_q.empty else None,
        "negative_profit_share_pct": round(
            float((df["net_profit"].dropna() < 0).mean() * 100), 2
        ) if df["net_profit"].notna().any() else None,
    }
    logger.info(f"get_portfolio_overview | industry={industry} | obs={result['observations']}")
    return result


def get_trend_over_time(metric: str, industry: str | None = None) -> list:
    if df_global is None:
        return [{"error": "Data has not been loaded yet."}]
    if metric not in {"rd_intensity", "roa", "tobinq"}:
        return [{"error": f"Unknown metric '{metric}'."}]
    df = df_global.copy()
    if industry and industry != "All Industries":
        df = df[df["industry"].astype(str) == industry]
    if df.empty:
        return [{"error": "No data for the requested filter."}]
    trend = (
        df.dropna(subset=["year", metric])
        .groupby("year")[metric]
        .median()
        .reset_index()
    )
    trend[metric] = trend[metric].round(4)
    trend["year"] = trend["year"].astype(int)
    logger.info(f"get_trend_over_time | metric={metric} | years={len(trend)}")
    return trend.to_dict(orient="records")


def get_segment_breakdown(dimension: str) -> list:
    if df_global is None:
        return [{"error": "Data has not been loaded yet."}]
    if dimension not in {"industry", "year", "name"}:
        return [{"error": f"Unknown dimension '{dimension}'."}]
    df = df_global.copy()
    grouped = (
        df.dropna(subset=[dimension])
        .groupby(dimension)
        .agg(
            observations=("rd_intensity", "count"),
            median_rd_intensity=("rd_intensity", "median"),
            median_roa=("roa", "median"),
            median_tobinq=("tobinq", "median"),
            median_patent=("patent", "median"),
        )
        .reset_index()
        .sort_values("median_rd_intensity", ascending=False)
    )
    grouped["median_rd_intensity_pct"] = (grouped["median_rd_intensity"] * 100).round(3)
    grouped["median_roa_pct"] = (grouped["median_roa"] * 100).round(3)
    grouped = grouped.drop(columns=["median_rd_intensity", "median_roa"])
    grouped["median_tobinq"] = grouped["median_tobinq"].round(3)
    grouped["median_patent"] = grouped["median_patent"].round(1)
    logger.info(f"get_segment_breakdown | dimension={dimension} | groups={len(grouped)}")
    return grouped.head(20).to_dict(orient="records")


def run_rd_tobinq_regression(industry: str | None = None) -> dict:
    if df_global is None:
        return {"error": "Data has not been loaded yet."}
    df = df_global.copy()
    if industry and industry != "All Industries":
        df = df[df["industry"].astype(str) == industry]
    reg = df[["tobinq", "rd_intensity"]].dropna()
    reg = reg[reg["rd_intensity"].between(0, 0.5)]
    if len(reg) < 10 or reg["rd_intensity"].nunique() < 2:
        return {"error": "Not enough valid observations for regression."}
    X = sm.add_constant(reg["rd_intensity"])
    y = reg["tobinq"]
    model = sm.OLS(y, X).fit()
    result = {
        "industry_filter": industry or "All Industries",
        "n_observations": int(model.nobs),
        "beta_rd_intensity": round(float(model.params["rd_intensity"]), 5),
        "alpha_const": round(float(model.params["const"]), 5),
        "r_squared": round(float(model.rsquared), 4),
        "p_value": round(float(model.pvalues["rd_intensity"]), 5),
        "significant_at_5pct": bool(model.pvalues["rd_intensity"] < 0.05),
    }
    logger.info(f"run_rd_tobinq_regression | industry={industry} | beta={result['beta_rd_intensity']} | p={result['p_value']}")
    return result


def find_high_risk_companies(
    min_rd_intensity: float = 0.1,
    max_roa: float = 0.0,
    max_tobinq: float = 1.0,
    limit: int = 15,
) -> list:
    if df_global is None:
        return [{"error": "Data has not been loaded yet."}]
    df = df_global.copy()
    flagged = df[
        (df["rd_intensity"] >= min_rd_intensity)
        & (df["roa"] <= max_roa)
        & (df["tobinq"] <= max_tobinq)
    ][["name", "year", "industry", "rd_intensity", "roa", "tobinq", "net_profit"]].copy()
    if flagged.empty:
        return [{"note": "No companies matched the risk criteria."}]
    flagged["rd_intensity"] = (flagged["rd_intensity"] * 100).round(3)
    flagged["roa"] = (flagged["roa"] * 100).round(3)
    flagged["tobinq"] = flagged["tobinq"].round(3)
    flagged = flagged.sort_values("rd_intensity", ascending=False).head(limit)
    logger.info(f"find_high_risk_companies | flagged={len(flagged)}")
    return flagged.to_dict(orient="records")


def get_industry_benchmark(top_n: int = 10) -> list:
    if df_global is None:
        return [{"error": "Data has not been loaded yet."}]
    bench = (
        df_global.dropna(subset=["industry", "rd_intensity"])
        .groupby("industry")
        .agg(
            observations=("rd_intensity", "count"),
            median_rd_intensity=("rd_intensity", "median"),
            median_roa=("roa", "median"),
            median_tobinq=("tobinq", "median"),
        )
        .reset_index()
        .sort_values("median_rd_intensity", ascending=False)
        .head(top_n)
    )
    bench["median_rd_intensity_pct"] = (bench["median_rd_intensity"] * 100).round(3)
    bench["median_roa_pct"] = (bench["median_roa"] * 100).round(3)
    bench["median_tobinq"] = bench["median_tobinq"].round(3)
    bench = bench.drop(columns=["median_rd_intensity", "median_roa"])
    logger.info(f"get_industry_benchmark | top_n={top_n}")
    return bench.to_dict(orient="records")


# ── Tool registry ──────────────────────────────────────────────────────────
tool_functions = {
    "get_portfolio_overview": get_portfolio_overview,
    "get_trend_over_time": get_trend_over_time,
    "get_segment_breakdown": get_segment_breakdown,
    "run_rd_tobinq_regression": run_rd_tobinq_regression,
    "find_high_risk_companies": find_high_risk_companies,
    "get_industry_benchmark": get_industry_benchmark,
}