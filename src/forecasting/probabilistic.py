"""Prediction intervals, stockout risk, and service-level safety stock.

Uses split conformal prediction with signed, scale-normalised scores
``(actual - forecast) / scale``. Signed scores keep the asymmetry of demand
errors (long right tail), and the per-series ``scale`` lets one calibration set
cover series whose volumes differ by orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_scores(actual, forecast, scale=None) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    scale = np.ones_like(actual) if scale is None else np.asarray(scale, dtype=float)
    return (actual - forecast) / scale


def conformal_quantile(scores, quantile: float) -> float:
    """Finite-sample corrected empirical quantile of calibration scores."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("No finite calibration scores")
    n = scores.size
    adjusted = np.clip(np.ceil((n + 1) * quantile) / n, 0, 1) if quantile >= 0.5 else np.clip(np.floor((n + 1) * quantile) / n, 0, 1)
    return float(np.quantile(scores, adjusted))


def conformal_interval(point_forecast, scores, level: float = 0.95, scale=None) -> pd.DataFrame:
    """Two-sided interval with nominal ``level`` coverage, lower bound clipped at zero."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    point = np.asarray(point_forecast, dtype=float)
    scale = np.ones_like(point) if scale is None else np.asarray(scale, dtype=float)
    alpha = 1 - level
    low, high = conformal_quantile(scores, alpha / 2), conformal_quantile(scores, 1 - alpha / 2)
    return pd.DataFrame({
        "forecast": point,
        "lower": np.clip(point + low * scale, 0, None),
        "upper": np.clip(point + high * scale, 0, None),
    })


def service_level_safety_stock(point_forecast, scores, service_level: float = 0.95, scale=None) -> pd.DataFrame:
    """Stock needed to meet demand with probability ``service_level`` (cycle service level)."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1")
    point = np.asarray(point_forecast, dtype=float)
    scale = np.ones_like(point) if scale is None else np.asarray(scale, dtype=float)
    safety = np.clip(conformal_quantile(scores, service_level) * scale, 0, None)
    result = pd.DataFrame({"forecast": point, "safety_stock": safety, "order_up_to": point + safety})
    result["stockout_risk"] = stockout_risk(point, result["order_up_to"], scores, scale)
    return result


def stockout_risk(point_forecast, stock, scores, scale=None) -> np.ndarray:
    """Empirical P(demand > stock) implied by the calibration score distribution."""
    point = np.asarray(point_forecast, dtype=float)
    stock = np.asarray(stock, dtype=float)
    scale = np.ones_like(point) if scale is None else np.asarray(scale, dtype=float)
    scores = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    thresholds = (stock - point) / scale
    return 1 - np.searchsorted(scores, thresholds, side="right") / scores.size


def demand_scenarios(point_forecast, scores, scale=None, n_scenarios: int = 25) -> np.ndarray:
    """Equally likely demand scenarios (rows x scenarios) from score quantiles."""
    point = np.asarray(point_forecast, dtype=float)
    scale = np.ones_like(point) if scale is None else np.asarray(scale, dtype=float)
    grid = (np.arange(n_scenarios) + 0.5) / n_scenarios
    quantiles = np.quantile(np.asarray(scores, dtype=float)[np.isfinite(scores)], grid)
    return np.clip(point[:, None] + scale[:, None] * quantiles[None, :], 0, None)


def empirical_coverage(actual, lower, upper) -> float:
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= np.asarray(lower)) & (actual <= np.asarray(upper))))
