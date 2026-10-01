import numpy as np
import pandas as pd
import pytest

from src.week8_demo.main import calculate_variables


def make_test_df():
    return pd.DataFrame({
        "rd_expense":        [100.0, 200.0, np.nan],
        "revenue":           [1000.0, 1000.0, 1000.0],
        "net_profit":        [50.0, 60.0, 70.0],
        "total_assets":      [500.0, 600.0, 700.0],
        "market_value":      [800.0, 900.0, 1000.0],
        "total_liabilities": [200.0, 300.0, 400.0],
    })


def test_rd_intensity_correct():
    result = calculate_variables(make_test_df())
    assert result.loc[0, "rd_intensity"] == pytest.approx(0.1)
    assert result.loc[1, "rd_intensity"] == pytest.approx(0.2)


def test_roa_correct():
    result = calculate_variables(make_test_df())
    assert result.loc[0, "roa"] == pytest.approx(0.1)
    assert result.loc[1, "roa"] == pytest.approx(0.1)


def test_tobinq_correct():
    result = calculate_variables(make_test_df())
    assert result.loc[0, "tobinq"] == pytest.approx(2.0)
    assert result.loc[1, "tobinq"] == pytest.approx(2.0)


def test_missing_value_propagates():
    result = calculate_variables(make_test_df())
    assert np.isnan(result.loc[2, "rd_intensity"])


def test_zero_denominator_handled():
    df = pd.DataFrame({
        "rd_expense":        [100.0],
        "revenue":           [0.0],
        "net_profit":        [50.0],
        "total_assets":      [0.0],
        "market_value":      [800.0],
        "total_liabilities": [200.0],
    })
    result = calculate_variables(df)
    assert np.isnan(result.loc[0, "rd_intensity"])
    assert np.isnan(result.loc[0, "roa"])
    assert np.isnan(result.loc[0, "tobinq"])


def test_no_inf_values():
    result = calculate_variables(make_test_df())
    assert not np.isinf(result["rd_intensity"]).any()
    assert not np.isinf(result["roa"]).any()
    assert not np.isinf(result["tobinq"]).any()