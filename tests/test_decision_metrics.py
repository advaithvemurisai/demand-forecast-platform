import numpy as np
import pandas as pd
import pytest

from forecasting import decision_metrics as dm


def test_window_sums_drop_incomplete_tail():
    matrix = np.arange(14, dtype=float).reshape(1, 14)
    assert dm.window_sums(matrix, 5).tolist() == [[10.0, 35.0]]


def test_wmape_and_bias_signs():
    wmape, bias = dm.wmape_bias(np.array([10.0, 10.0]), np.array([12.0, 8.0]))
    assert wmape == pytest.approx(0.2) and bias == pytest.approx(0.0)
    _, over = dm.wmape_bias(np.array([10.0]), np.array([15.0]))
    assert over == pytest.approx(0.5)


def test_decision_accuracy_uses_each_products_window():
    keys = pd.DataFrame({"cat_id": ["FOODS", "HOBBIES"]})
    actual = np.ones((2, 16))
    forecast = np.ones((2, 16)) * 2
    result = dm.decision_accuracy(actual, forecast, keys, np.array([3, 8]), np.array(["fast", "slow"]), "holdout")
    assert set(result["window_days"]) == {3, 8}
    assert np.allclose(result["bias"], 1.0)


def test_bias_by_weekday_finds_weekend_under_forecast():
    dates = pd.date_range("2016-01-04", periods=14)  # starts on a Monday
    actual = np.ones((1, 14))
    actual[:, dates.dayofweek >= 5] = 3.0
    forecast = np.ones((1, 14)) * 1.5
    table = dm.bias_by_weekday(actual, forecast, dates, "holdout").set_index("weekday")["bias"]
    assert table["Sat"] < 0 < table["Mon"]


def test_tracking_signal_flags_persistent_bias_only():
    rng = np.random.default_rng(0)
    forecast = np.full((2, 16), 10.0)
    actual = np.vstack([10 + rng.normal(0, 1, 16), np.full(16, 14.0) + rng.normal(0, 1, 16)])
    signal = dm.tracking_signal(actual, forecast)
    assert abs(signal[0]) < 4 < signal[1]
    keys = pd.DataFrame({"series_id": ["a", "b"], "item_id": ["a", "b"], "store_id": ["s", "s"], "dept_id": ["d", "d"]})
    flagged = dm.bias_exceptions(actual, forecast, keys, np.array([2.0, 3.0]))
    assert flagged["series_id"].tolist() == ["b"] and flagged["direction"].iloc[0] == "under-forecast"


def test_paired_ci_excludes_zero_for_consistent_gain():
    rng = np.random.default_rng(1)
    b = rng.normal(100, 5, 12)
    result = dm.paired_ci(b + 6 + rng.normal(0, 1, 12), b)
    assert result["lower"] > 0 and result["wins"] == 1.0 and result["n"] == 12


def test_forecast_stability_zero_for_identical_and_scaled_for_shifts():
    previous = np.array([[10.0, 20.0], [5.0, 5.0]])
    assert dm.forecast_stability(previous, previous) == 0.0
    assert dm.forecast_stability(previous, previous * 1.1) == pytest.approx(0.1)


def test_bias_correction_only_touches_persistently_biased_products():
    rng = np.random.default_rng(3)
    forecast = np.full((3, 16), 10.0)
    actual = np.vstack([10 + rng.normal(0, 1, 16), 14 + rng.normal(0, 1, 16), 10 + rng.normal(0, 1, 16)])
    multiplier, flagged = dm.bias_correction(actual, forecast)
    assert flagged.tolist() == [False, True, False]
    assert multiplier[0] == 1.0 and 1.0 < multiplier[1] <= 1.6


def test_forecast_value_added_positive_when_override_removes_bias():
    actual = np.full((2, 4), 14.0)
    base = np.full((2, 4), 10.0)
    fixed = dm.forecast_value_added(actual, base, base * 1.4)
    worse = dm.forecast_value_added(actual, base, base * 0.5)
    assert fixed["fva"] > 0 > worse["fva"] and fixed["n"] == 2
    assert np.isnan(dm.forecast_value_added(actual, base, base, np.zeros(2, dtype=bool))["fva"])
