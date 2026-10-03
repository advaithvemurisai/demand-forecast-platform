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


def mondrian_interval(point_forecast, scores, score_groups, point_groups, level: float = 0.95, scale=None, min_count: int = 500) -> pd.DataFrame:
    """Conformal interval calibrated separately inside each segment (Mondrian conformal).

    Pooled scores let easy segments (fast sellers) subsidise hard ones (sporadic sellers), so
    coverage looks right overall but is wrong for each segment. A segment with fewer than
    ``min_count`` calibration scores falls back to the pooled scores.
    """
    point = np.asarray(point_forecast, dtype=float)
    scale = np.ones_like(point) if scale is None else np.asarray(scale, dtype=float)
    scores, score_groups = np.asarray(scores, dtype=float), np.asarray(score_groups)
    point_groups = np.asarray(point_groups)
    result = pd.DataFrame({"forecast": point, "lower": np.zeros_like(point), "upper": np.zeros_like(point)})
    for group in np.unique(point_groups):
        rows = point_groups == group
        calibration = scores[score_groups == group]
        if np.isfinite(calibration).sum() < min_count:
            calibration = scores
        interval = conformal_interval(point[rows], calibration, level, scale[rows])
        result.loc[rows, "lower"] = interval["lower"].to_numpy()
        result.loc[rows, "upper"] = interval["upper"].to_numpy()
    return result


def velocity_class(mean_daily_sales) -> np.ndarray:
    """Demand-speed segment from average daily units: fast / medium / slow / sporadic."""
    mean = np.asarray(mean_daily_sales, dtype=float)
    return np.select([mean >= 2.0, mean >= 0.5, mean >= 0.1], ["fast", "medium", "slow"], default="sporadic")


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


# Cycle-service targets by value (ABC) x variability (XYZ): protect staples, accept more risk on the long tail.
SERVICE_TARGETS = {
    ("A", "X"): 0.98, ("A", "Y"): 0.97, ("A", "Z"): 0.95,
    ("B", "X"): 0.96, ("B", "Y"): 0.95, ("B", "Z"): 0.92,
    ("C", "X"): 0.93, ("C", "Y"): 0.90, ("C", "Z"): 0.85,
}


def abc_class(revenue, a_share: float = 0.80, b_share: float = 0.95) -> np.ndarray:
    """A = products making the first 80% of revenue, B = next 15%, C = the rest."""
    revenue = np.asarray(revenue, dtype=float)
    order = np.argsort(-revenue, kind="stable")
    total = revenue.sum()
    cumulative = np.cumsum(revenue[order]) / total if total > 0 else np.ones(len(revenue))
    labels = np.empty(len(revenue), dtype="<U1")
    previous = np.concatenate([[0.0], cumulative[:-1]])  # a product that crosses the line stays in the better class
    labels[order] = np.where(previous < a_share, "A", np.where(previous < b_share, "B", "C"))
    return labels


def xyz_class(weekly_sales, x_cv: float = 0.5, y_cv: float = 1.0) -> np.ndarray:
    """X = steady, Y = variable, Z = erratic, by the coefficient of variation of weekly sales."""
    weekly_sales = np.asarray(weekly_sales, dtype=float)
    mean = weekly_sales.mean(axis=1)
    cv = np.divide(weekly_sales.std(axis=1), mean, out=np.full(len(mean), np.inf), where=mean > 0)
    return np.select([cv < x_cv, cv < y_cv], ["X", "Y"], default="Z")


def service_targets(abc, xyz) -> np.ndarray:
    return np.array([SERVICE_TARGETS[(a, x)] for a, x in zip(abc, xyz)])


def round_to_case(quantity, case_pack: int = 1) -> np.ndarray:
    """Round up to whole case packs: never ship less than the policy asks for."""
    quantity = np.asarray(quantity, dtype=float)
    return np.ceil(np.maximum(quantity, 0) / case_pack - 1e-9) * case_pack


def expected_fill_rate(point_forecast, stock, scores, scale=None, n_points: int = 50) -> np.ndarray:
    """Share of demand units served from ``stock``: E[min(D, stock)] / E[D] under the score distribution."""
    point = np.asarray(point_forecast, dtype=float)
    stock = np.asarray(stock, dtype=float)
    scale = np.ones_like(point) if scale is None else np.asarray(scale, dtype=float)
    scores = np.asarray(scores, dtype=float)
    grid = (np.arange(n_points) + 0.5) / n_points
    demand = np.clip(point[:, None] + scale[:, None] * np.quantile(scores[np.isfinite(scores)], grid)[None, :], 0, None)
    served = np.minimum(demand, stock[:, None]).mean(axis=1)
    total = demand.mean(axis=1)
    return np.divide(served, total, out=np.ones(len(point)), where=total > 0)
