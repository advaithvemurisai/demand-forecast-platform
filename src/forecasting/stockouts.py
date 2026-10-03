"""Probable-stockout detection for censored demand.

M5 records sales, not demand: a day with no stock shows as a zero. A zero run is *probable
stockout* when, at the product's normal selling rate, that many consecutive zeros would be
very unlikely (Poisson: ``P = exp(-rate * run_length)``) for a product that sells steadily. Days before a product's first sale
are not flagged: it had not launched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def probable_stockouts(history: np.ndarray, p_threshold: float = 0.001, min_run: int = 2, min_rate: float = 1.0) -> np.ndarray:
    """Boolean matrix (series x days): zero-sales days inside an unusually long zero run.

    Only products selling at least ``min_rate`` units/day are judged. For slower, clumpier products a
    long zero run is ordinary, and the Poisson test over-flags it (about 19% of days at p < 0.01 on the
    California data, which biased the model upward). At p < 0.001 and 1 unit/day, about 4% are flagged.
    """
    sales = np.nan_to_num(np.asarray(history, dtype=float))
    zero = sales == 0
    started = np.cumsum(sales > 0, axis=1) > 0
    n_days = sales.shape[1]
    forward = np.zeros(sales.shape, dtype=np.int32)
    backward = np.zeros(sales.shape, dtype=np.int32)
    for day in range(n_days):
        previous = forward[:, day - 1] if day else 0
        forward[:, day] = np.where(zero[:, day], previous + 1, 0)
    for day in range(n_days - 1, -1, -1):
        following = backward[:, day + 1] if day < n_days - 1 else 0
        backward[:, day] = np.where(zero[:, day], following + 1, 0)
    run_length = forward + backward - 1
    rate = sales.sum(axis=1, keepdims=True) / np.maximum(started.sum(axis=1, keepdims=True), 1)
    probability = np.exp(-rate * run_length)
    return zero & started & (run_length >= min_run) & (probability < p_threshold) & (rate >= min_rate)


def censoring_summary(flags: np.ndarray, keys: pd.DataFrame, history: np.ndarray) -> pd.DataFrame:
    """Share of on-the-shelf product-days flagged, by store and category (the diagnostic layer)."""
    started = np.cumsum(np.nan_to_num(history) > 0, axis=1) > 0
    frame = keys[["store_id", "cat_id"]].astype(str).copy()
    frame["flagged_days"] = flags.sum(axis=1)
    frame["observed_days"] = started.sum(axis=1)
    summary = frame.groupby(["store_id", "cat_id"], as_index=False)[["flagged_days", "observed_days"]].sum()
    summary["censored_share"] = summary["flagged_days"] / summary["observed_days"].clip(lower=1)
    return summary
