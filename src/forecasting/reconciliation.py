"""Forecast reconciliation methods: bottom-up, top-down, and MinT."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from forecasting.data import LEVEL_COLUMNS, aggregate_bottom_up, make_node_ids

AGGREGATE_LEVELS = [level for level in LEVEL_COLUMNS if level != "item"]


def summing_matrix(bottom_keys: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Aggregation matrix ``A`` (aggregate nodes x bottom series) for the M5 hierarchy.

    ``bottom_keys`` holds one row per bottom series, in the column order used for
    forecasts, with the ID columns of ``LEVEL_COLUMNS``. The full summing matrix is
    ``S = [A; I]``. Returns ``A`` and a table of the aggregate nodes (row order of ``A``).
    """
    rows, cols, nodes = [], [], []
    offset = 0
    for level in AGGREGATE_LEVELS:
        ids = make_node_ids(level, bottom_keys)
        codes, uniques = pd.factorize(ids, sort=True)
        rows.append(codes + offset)
        cols.append(np.arange(len(bottom_keys)))
        nodes.append(pd.DataFrame({"node_id": uniques, "level": level}))
        offset += len(uniques)
    data = np.ones(sum(len(r) for r in rows))
    matrix = sparse.csr_matrix((data, (np.concatenate(rows), np.concatenate(cols))), shape=(offset, len(bottom_keys)))
    return matrix, pd.concat(nodes, ignore_index=True)


def bottom_up(bottom_forecasts: pd.DataFrame, levels=None, value: str = "forecast") -> pd.DataFrame:
    """Aggregate bottom forecasts through the explicit hierarchy."""
    frames = [aggregate_bottom_up(bottom_forecasts, level, value) for level in (levels or LEVEL_COLUMNS)]
    return pd.concat(frames, ignore_index=True)


def historical_proportions(bottom_history: pd.DataFrame, value: str = "sales") -> pd.DataFrame:
    """Gross-Sohl method A: each bottom node's average share of the daily total."""
    daily_total = bottom_history.groupby("date")[value].transform("sum")
    shares = bottom_history.assign(share=bottom_history[value] / daily_total.where(daily_total > 0))
    result = shares.groupby("node_id", as_index=False, observed=True)["share"].mean().rename(columns={"share": "proportion"})
    result["proportion"] = result["proportion"].fillna(0)
    result["proportion"] /= result["proportion"].sum()
    return result


def top_down(proportions: pd.DataFrame, total_forecast: pd.DataFrame, target_level: str = "item") -> pd.DataFrame:
    """Disaggregate a total forecast (``date``, ``total_forecast``) by fixed proportions."""
    required = {"node_id", "proportion"}.difference(proportions.columns)
    if required:
        raise ValueError(f"Missing columns: {required}")
    result = proportions[["node_id", "proportion"]].merge(total_forecast[["date", "total_forecast"]], how="cross")
    result["forecast"] = result["proportion"] * result["total_forecast"]
    result["level"] = target_level
    return result[["date", "node_id", "level", "forecast"]]


def mint(base_forecasts: np.ndarray, summing_matrix: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Dense trace-minimising reconciliation ``S (S' W^-1 S)^-1 S' W^-1 y``.

    Suitable for small hierarchies; use :func:`mint_diagonal` at SKU scale.
    """
    inv_cov = np.linalg.pinv(covariance)
    projection = summing_matrix @ np.linalg.pinv(summing_matrix.T @ inv_cov @ summing_matrix) @ summing_matrix.T @ inv_cov
    return projection @ base_forecasts


def mint_diagonal(
    base_aggregate: np.ndarray,
    base_bottom: np.ndarray,
    aggregation: sparse.spmatrix,
    var_aggregate: np.ndarray,
    var_bottom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MinT with a diagonal error covariance (a.k.a. WLS-variance), solved via Woodbury.

    With ``S = [A; I]`` and ``W = diag(w_a, w_b)``, the reconciled bottom level is
    ``M^-1 (A' W_a^-1 y_a + W_b^-1 y_b)`` where ``M = W_b^-1 + A' W_a^-1 A``.
    Woodbury gives ``M^-1 = W_b - W_b A' (W_a + A W_b A')^-1 A W_b``, so the only
    dense inverse is k x k for k aggregate nodes, not m x m for m bottom series.

    Inputs are (nodes x horizon) arrays. Returns reconciled (aggregate, bottom).
    """
    var_aggregate = np.maximum(np.asarray(var_aggregate, dtype=float), 1e-9)
    var_bottom = np.maximum(np.asarray(var_bottom, dtype=float), 1e-9)
    base_aggregate = np.asarray(base_aggregate, dtype=float).reshape(len(var_aggregate), -1)
    base_bottom = np.asarray(base_bottom, dtype=float).reshape(len(var_bottom), -1)
    A = sparse.csr_matrix(aggregation)
    rhs = A.T @ (base_aggregate / var_aggregate[:, None]) + base_bottom / var_bottom[:, None]
    scaled = var_bottom[:, None] * rhs
    core = np.diag(var_aggregate) + (A.multiply(var_bottom[None, :]) @ A.T).toarray()
    correction = var_bottom[:, None] * (A.T @ np.linalg.solve(core, A @ scaled))
    bottom = scaled - correction
    return np.asarray(A @ bottom), bottom
