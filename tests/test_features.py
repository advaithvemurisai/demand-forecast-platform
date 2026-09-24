import numpy as np
import pandas as pd
import pytest

from forecasting.features import add_features, feature_columns


def frame(with_promo=True):
    dates = pd.date_range("2024-01-01", periods=120)
    data = pd.DataFrame({
        "series_id": np.repeat(["a", "b"], len(dates)),
        "date": np.tile(dates, 2),
        "sales": np.concatenate([np.arange(120), 1000 + np.arange(120)]).astype(float),
    })
    if with_promo:
        data["promo"] = 0
    return data


def test_features_without_promo_column():
    result = add_features(frame(with_promo=False))
    assert (result["promo"] == 0).all()


def test_lags_respect_series_boundaries_and_min_lag():
    result = add_features(frame()).set_index(["series_id", "date"])
    b = result.loc["b"]
    assert b["lag_28"].iloc[:28].isna().all()
    assert b["lag_28"].iloc[28] == 1000
    # rolling mean over 28 days ending 28 days ago never mixes in series "a"
    first_valid = b["rolling_mean_28"].first_valid_index()
    assert b.loc[first_valid, "rolling_mean_28"] == pytest.approx(np.mean(1000 + np.arange(28)))


def test_short_lags_are_rejected():
    with pytest.raises(ValueError):
        add_features(frame(), lags=(1, 28))


def test_feature_columns_exclude_target():
    columns = feature_columns(add_features(frame()))
    assert "sales" not in columns and "lag_28" in columns
