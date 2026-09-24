"""LightGBM global and local model adapters.

* Global: one model pooled across every series in the training frame.
* Local: one model per ``group_column`` (store by default). Per-series models are
  not practical for 30k intermittent M5 series and generalise poorly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.features import ID_FEATURES, feature_columns

DEFAULT_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}


def _design(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    design = frame[columns].copy()
    for column in ID_FEATURES:
        if column in design:
            design[column] = design[column].astype("category")
    return design


def fit_global(frame: pd.DataFrame, target: str = "sales", quantile: float | None = None, params: dict | None = None):
    """Fit a pooled LightGBM model. ``quantile`` switches to a pinball objective."""
    from lightgbm import LGBMRegressor

    settings = {**DEFAULT_PARAMS, **(params or {})}
    if quantile is not None:
        settings.update(objective="quantile", alpha=quantile)
        settings.pop("tweedie_variance_power", None)
    train = frame[frame[target].notna()]
    columns = feature_columns(train)
    model = LGBMRegressor(**settings)
    model.fit(_design(train, columns), train[target])
    model.feature_names_used_ = columns
    return model


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(_design(frame, model.feature_names_used_)), 0, None)


def fit_local(frame: pd.DataFrame, target: str = "sales", group_column: str = "store_id", params: dict | None = None) -> dict[str, object]:
    return {
        str(key): fit_global(group, target=target, params=params)
        for key, group in frame.groupby(group_column, observed=True)
    }


def predict_local(models: dict[str, object], frame: pd.DataFrame, group_column: str = "store_id") -> np.ndarray:
    output = np.zeros(len(frame))
    keys = frame[group_column].astype(str).to_numpy()
    for key, model in models.items():
        mask = keys == key
        if mask.any():
            output[mask] = predict(model, frame.loc[mask])
    return output
