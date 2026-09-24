"""Mandatory naive floors for every time series, plus a SARIMA benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(values: pd.Series, horizon: int) -> pd.Series:
    if values.empty:
        raise ValueError("Cannot forecast an empty series")
    return pd.Series([values.iloc[-1]] * horizon, name="forecast")


def seasonal_naive_forecast(values: pd.Series, horizon: int, season_length: int = 7) -> pd.Series:
    if len(values) < season_length:
        return naive_forecast(values, horizon)
    pattern = values.iloc[-season_length:].to_numpy()
    return pd.Series([pattern[index % season_length] for index in range(horizon)], name="forecast")


def arima_forecast(values: pd.Series, horizon: int, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)) -> tuple[pd.Series, float]:
    """Fit a per-series SARIMA model; returns the forecast and in-sample residual variance."""
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(values.astype(float).to_numpy(), order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False)
    burn_in = sum(order) + seasonal_order[1] * seasonal_order[3] + seasonal_order[3]
    variance = float(np.var(fitted.resid[burn_in:]))
    return pd.Series(np.clip(fitted.forecast(horizon), 0, None), name="forecast"), variance


def forecast_all(frame: pd.DataFrame, horizon: int, season_length: int = 7) -> pd.DataFrame:
    """Naive and seasonal-naive forecasts for every ``series_id`` (vectorised).

    Returns one row per series, horizon step, and model with the forecast ``date``.
    """
    ordered = frame.sort_values(["series_id", "date"])
    last_date = ordered["date"].max()
    tail = ordered.groupby("series_id", observed=True).tail(season_length)
    wide = tail.pivot(index="series_id", columns="date", values="sales")
    if wide.shape[1] < season_length:
        raise ValueError(f"Need at least {season_length} days of history")
    pattern = wide.to_numpy(dtype=float)
    steps = np.arange(horizon)
    seasonal = pattern[:, steps % season_length]
    naive = np.repeat(pattern[:, -1:], horizon, axis=1)
    dates = last_date + pd.to_timedelta(steps + 1, unit="D")
    frames = []
    for model, values in (("naive", naive), ("seasonal_naive", seasonal)):
        frames.append(pd.DataFrame({
            "series_id": np.repeat(wide.index.to_numpy(), horizon),
            "horizon": np.tile(steps + 1, len(wide)),
            "date": np.tile(dates, len(wide)),
            "model": model,
            "forecast": values.ravel(),
        }))
    return pd.concat(frames, ignore_index=True)
