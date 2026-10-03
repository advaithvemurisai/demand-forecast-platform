"""Accuracy where planners decide: weekly windows, bias by segment, tracking signals, paired CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_sums(matrix: np.ndarray, days: int) -> np.ndarray:
    """Sums over consecutive non-overlapping ``days``-day windows: (series, windows)."""
    windows = matrix.shape[1] // days
    return matrix[:, : windows * days].reshape(len(matrix), windows, days).sum(axis=2)


def wmape_bias(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """Volume-weighted error and signed bias (positive = over-forecast)."""
    total = np.abs(actual).sum()
    if total == 0:
        return float("nan"), float("nan")
    return float(np.abs(actual - forecast).sum() / total), float((forecast - actual).sum() / total)


def decision_accuracy(actual: np.ndarray, forecast: np.ndarray, keys: pd.DataFrame, windows: np.ndarray, velocity: np.ndarray, fold: str) -> pd.DataFrame:
    """WMAPE and bias of item x store forecasts summed over each product's replenishment window.

    ``windows`` gives each series' protection days (lead time + review period), so a food item
    ordered every 2 days is judged on 3-day sums and a weekly-ordered item on 8-day sums.
    """
    rows = []
    for window in np.unique(windows):
        mask = windows == window
        a, f = window_sums(actual[mask], int(window)), window_sums(forecast[mask], int(window))
        for cat, vel in pd.MultiIndex.from_arrays([keys["cat_id"].astype(str).to_numpy()[mask], velocity[mask]]).unique():
            sub = (keys["cat_id"].astype(str).to_numpy()[mask] == cat) & (velocity[mask] == vel)
            wmape, bias = wmape_bias(a[sub], f[sub])
            rows.append({"fold": fold, "window_days": int(window), "cat_id": cat, "velocity": vel, "wmape": wmape, "bias": bias, "units": float(a[sub].sum())})
    return pd.DataFrame(rows)


def bias_by_weekday(actual: np.ndarray, forecast: np.ndarray, dates: pd.DatetimeIndex, fold: str) -> pd.DataFrame:
    """Bias of total forecast vs actual for each day of the week (weekend peaks show up here)."""
    rows = []
    for weekday in range(7):
        cols = np.flatnonzero(dates.dayofweek == weekday)
        if cols.size:
            _, bias = wmape_bias(actual[:, cols], forecast[:, cols])
            rows.append({"fold": fold, "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday], "bias": bias})
    return pd.DataFrame(rows)


def tracking_signal(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Standardised bias per series: mean error / its standard error (a t-statistic).

    Unlike cumulative error / MAD, whose spread under *no* bias grows with the number of periods,
    this stays near zero for an unbiased forecast, so |signal| > 4 reliably means the forecast
    keeps missing in one direction. Positive = actual above forecast (under-forecast).
    """
    error = actual - forecast
    n = error.shape[1]
    std = error.std(axis=1, ddof=1) if n > 1 else np.zeros(len(error))
    return np.divide(error.mean(axis=1) * np.sqrt(n), std, out=np.zeros(len(error)), where=std > 0)


def bias_exceptions(actual_weeks: np.ndarray, forecast_weeks: np.ndarray, keys: pd.DataFrame, unit_price: np.ndarray, threshold: float = 4.0, top: int = 200) -> pd.DataFrame:
    """Products the forecast keeps missing in the same direction, ranked by dollar impact."""
    signal = tracking_signal(actual_weeks, forecast_weeks)
    per_week = (actual_weeks - forecast_weeks).mean(axis=1)
    frame = keys[["series_id", "item_id", "store_id", "dept_id"]].astype(str).copy()
    frame["tracking_signal"] = signal
    frame["bias_units_per_week"] = -per_week  # positive = over-forecast
    frame["direction"] = np.where(signal > 0, "under-forecast", "over-forecast")
    frame["weekly_impact"] = np.abs(per_week) * unit_price
    flagged = frame[np.abs(frame["tracking_signal"]) > threshold]
    return flagged.sort_values("weekly_impact", ascending=False).head(top).reset_index(drop=True)


def paired_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Mean paired difference a - b with a t-interval and the share of pairs where a > b."""
    from scipy import stats

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    mean = float(diff.mean())
    if n < 2:
        return {"mean": mean, "lower": float("nan"), "upper": float("nan"), "wins": float((diff > 0).mean()), "n": n}
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * diff.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "wins": float((diff > 0).mean()), "n": n}


def forecast_stability(previous: np.ndarray, new: np.ndarray) -> float:
    """How much the forecast for the *same* target days moved between two consecutive runs.

    ``sum |new - previous| / sum previous`` over the overlapping days; 0 means identical.
    Planners distrust forecasts that jump around between weekly runs even when they are accurate.
    """
    previous, new = np.asarray(previous, dtype=float), np.asarray(new, dtype=float)
    total = np.abs(previous).sum()
    return float(np.abs(new - previous).sum() / total) if total > 0 else float("nan")


def bias_correction(prior_actual: np.ndarray, prior_forecast: np.ndarray, threshold: float = 4.0, strength: float = 0.5, limits: tuple[float, float] = (0.6, 1.6)) -> tuple[np.ndarray, np.ndarray]:
    """A planner-style override: nudge forecasts for products with a persistent bias.

    Flags products whose tracking signal exceeds ``threshold`` and scales their forecast by
    ``1 + strength x (mean actual / mean forecast - 1)`` within ``limits``. Returns (multipliers, flagged).
    """
    signal = tracking_signal(prior_actual, prior_forecast)
    flagged = np.abs(signal) > threshold
    mean_forecast = prior_forecast.mean(axis=1)
    ratio = np.divide(prior_actual.mean(axis=1), mean_forecast, out=np.ones(len(mean_forecast)), where=mean_forecast > 0)
    multiplier = np.where(flagged, np.clip(1 + strength * (ratio - 1), *limits), 1.0)
    return multiplier, flagged


def forecast_value_added(actual: np.ndarray, base: np.ndarray, adjusted: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """WMAPE of the base forecast minus WMAPE of the adjusted one (positive = the override helped)."""
    mask = np.ones(len(actual), dtype=bool) if mask is None else mask
    if not mask.any():
        return {"base_wmape": float("nan"), "adjusted_wmape": float("nan"), "fva": float("nan"), "n": 0}
    base_wmape, _ = wmape_bias(actual[mask], base[mask])
    adjusted_wmape, _ = wmape_bias(actual[mask], adjusted[mask])
    return {"base_wmape": base_wmape, "adjusted_wmape": adjusted_wmape, "fva": base_wmape - adjusted_wmape, "n": int(mask.sum())}
