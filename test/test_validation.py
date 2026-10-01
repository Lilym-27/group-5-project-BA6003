import numpy as np
import pandas as pd
import pytest

from src.week8_demo.main import validate_data, validate_calculations


def make_valid_df():
    return pd.DataFrame({
        "name":              ["A", "B"],
        "year":              [2020, 2021],
        "revenue":           [1000.0, 1000.0],
        "rd_expense":        [100.0, 200.0],
        "net_profit":        [50.0, 60.0],
        "total_assets":      [500.0, 600.0],
        "market_value":      [800.0, 900.0],
        "total_liabilities": [200.0, 300.0],
        "patent":            [10, 20],
        "industry":          ["Tech", "Tech"],
    })


def test_validate_data_passes():
    validate_data(make_valid_df())


def test_validate_data_missing_column():
    df = make_valid_df().drop(columns=["revenue"])
    with pytest.raises(AssertionError):
        validate_data(df)


def test_validate_data_empty():
    df = make_valid_df().iloc[0:0]
    with pytest.raises(AssertionError):
        validate_data(df)


def test_validate_calculations_passes():
    df = make_valid_df()
    df["rd_intensity"] = df["rd_expense"] / df["revenue"]
    df["roa"] = df["net_profit"] / df["total_assets"]
    df["tobinq"] = (
        df["market_value"] + df["total_liabilities"]
    ) / df["total_assets"]
    validate_calculations(df)


def test_validate_calculations_detects_error():
    df = make_valid_df()
    df["rd_intensity"] = df["rd_expense"] / df["revenue"]
    df["roa"] = df["net_profit"] / df["total_assets"]
    df["tobinq"] = (
        df["market_value"] + df["total_liabilities"]
    ) / df["total_assets"]
    df.loc[0, "rd_intensity"] = 999.0
    with pytest.raises(AssertionError):
        validate_calculations(df)