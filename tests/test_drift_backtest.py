import numpy as np
import pandas as pd
import pytest

from forecasting.backtest import bias, rolling_origin, score, wmape, wrmsse
from forecasting.drift import detect_drift, psi, retraining_flag


def test_psi_counts_values_outside_reference_range():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 5000))
    partly_shifted = pd.Series(np.concatenate([rng.normal(0, 1, 2500), rng.normal(8, 1, 2500)]))
    assert psi(reference, partly_shifted) > 1.0
    assert not detect_drift(reference, pd.Series(rng.normal(0, 1, 5000)))["drift"]


def test_retraining_flag_reasons():
    stable = retraining_flag(pd.Series({"s1": 0.05}), recent_wmape=0.30, reference_wmape=0.29)
    assert not stable["retrain"]
    flagged = retraining_flag(pd.Series({"s1": 0.5}), recent_wmape=0.40, reference_wmape=0.30)
    assert flagged["retrain"] and "PSI" in flagged["reasons"] and "WMAPE" in flagged["reasons"]


def test_rolling_origin_aligns_multiple_series_by_key():
    dates = pd.date_range("2024-01-01", periods=20)
    frame = pd.DataFrame({"series_id": np.repeat(["a", "b"], 20), "date": np.tile(dates, 2), "sales": np.r_[np.ones(20), np.full(20, 100.0)]})

    def last_value(train, horizon):
        last = train.sort_values("date").groupby("series_id").tail(1)
        future = pd.date_range(train["date"].max() + pd.Timedelta(days=1), periods=horizon)
        # deliberately returned in reverse order: alignment must use keys, not position
        last = last[["series_id", "sales"]].rename(columns={"sales": "forecast"})
        return last.merge(pd.DataFrame({"date": future}), how="cross").iloc[::-1]

    result = rolling_origin(frame, last_value, [dates[9]], horizon=5)
    assert result["wmape"].iloc[0] == pytest.approx(0.0)


def test_score_and_bias():
    frame = pd.DataFrame({"g": ["x", "x", "y"], "actual": [10.0, 10.0, 5.0], "forecast": [12.0, 8.0, 5.0]})
    table = score(frame, ["g"]).set_index("g")
    assert table.loc["x", "wmape"] == pytest.approx(0.2) and table.loc["y", "wmape"] == 0
    assert bias([10, 10], [12, 12]) == pytest.approx(0.2)
    assert wmape([0, 0], [1, 1]) == 0.0


def test_wrmsse_matches_hand_calculation():
    actual = np.array([[2.0, 4.0], [10.0, 10.0]])
    forecast = np.array([[2.0, 2.0], [10.0, 14.0]])
    # node RMSE: sqrt(2), sqrt(8); scales sqrt(2), sqrt(2) -> RMSSE 1 and 2; weights 1:3
    assert wrmsse(actual, forecast, np.array([2.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.25 * 1 + 0.75 * 2)
    # a node with no sales history (zero scale) is ignored
    assert wrmsse(actual, forecast, np.array([2.0, 0.0]), np.array([1.0, 3.0])) == pytest.approx(1.0)
