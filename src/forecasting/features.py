"""Feature engineering for global and local demand models.

Every lag and rolling statistic is shifted by at least ``min_lag`` days. With
``min_lag`` equal to the forecast horizon, one direct model can score every step
of the horizon without recursive feeding or leaking future actuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_FEATURES = ["item_id", "dept_id", "cat_id", "store_id"]


def add_features(
    frame: pd.DataFrame,
    lags=(28, 35, 42, 56),
    windows=(7, 28, 56),
    min_lag: int = 28,
) -> pd.DataFrame:
    """Add lag, rolling, calendar, and price features.

    Expects one contiguous daily row per ``series_id`` (as in M5). Rows whose
    target is unknown (the forecast horizon) may carry ``sales = NaN``.
    """
    if any(lag < min_lag for lag in lags):
        raise ValueError(f"All lags must be >= min_lag ({min_lag}) to avoid leakage")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)
    sales = result["sales"].astype("float32")
    grouped = sales.groupby(result["series_id"], sort=False, observed=True)
    position = grouped.cumcount().to_numpy()

    for lag in lags:
        result[f"lag_{lag}"] = grouped.shift(lag).astype("float32")

    # Rolling stats on the whole column, then blank rows whose window crosses a
    # series boundary. Much faster than per-group lambdas on millions of rows.
    shifted = grouped.shift(min_lag)
    for window in windows:
        valid = position >= min_lag + window - 1
        mean = shifted.rolling(window).mean().to_numpy(dtype="float32")
        result[f"rolling_mean_{window}"] = np.where(valid, mean, np.nan).astype("float32")
    std = shifted.rolling(28).std().to_numpy(dtype="float32")
    result["rolling_std_28"] = np.where(position >= min_lag + 27, std, np.nan).astype("float32")

    result["day_of_week"] = result["date"].dt.dayofweek.astype("int8")
    result["day_of_month"] = result["date"].dt.day.astype("int8")
    result["month"] = result["date"].dt.month.astype("int8")
    result["week_of_year"] = result["date"].dt.isocalendar().week.astype("int8")
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype("int8")

    promo = result["promo"] if "promo" in result else pd.Series(0, index=result.index)
    result["promo"] = promo.fillna(0).astype("int8")
    if "event_name_1" in result:
        result["is_event"] = result["event_name_1"].notna().astype("int8")
        result["event_type"] = result["event_type_1"].astype("category").cat.codes.astype("int8")
    else:
        result["is_event"] = np.int8(0)
        result["event_type"] = np.int8(-1)
    if "state_id" in result and any(f"snap_{state}" in result for state in ("CA", "TX", "WI")):
        snap = np.zeros(len(result), dtype="int8")
        for state in ("CA", "TX", "WI"):
            if f"snap_{state}" in result:
                snap = np.where(result["state_id"].astype(str) == state, result[f"snap_{state}"], snap)
        result["snap"] = snap.astype("int8")
    else:
        result["snap"] = np.int8(0)

    if "sell_price" in result:
        price = result["sell_price"].astype("float32")
        price_groups = price.groupby(result["series_id"], sort=False, observed=True)
        result["price_ratio"] = (price / price_groups.transform("median")).astype("float32")
        result["price_change_7"] = (price / price_groups.shift(7) - 1).astype("float32")
    else:
        result["price_ratio"] = np.float32(np.nan)
        result["price_change_7"] = np.float32(np.nan)
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs present in ``frame`` (numeric features plus categorical IDs)."""
    numeric = [
        column for column in frame.columns
        if column.startswith(("lag_", "rolling_"))
    ] + [
        "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
        "promo", "is_event", "event_type", "snap", "price_ratio", "price_change_7",
    ]
    if "sell_price" in frame:
        numeric.append("sell_price")
    return [column for column in numeric + ID_FEATURES if column in frame.columns]
