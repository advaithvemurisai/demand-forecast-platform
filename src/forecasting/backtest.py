"""Rolling-origin evaluation and MLflow logging."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def wmape(actual: pd.Series, forecast: pd.Series) -> float:
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denominator = np.abs(actual).sum()
    return float(np.abs(actual - forecast).sum() / denominator) if denominator else 0.0


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    """MAPE over non-zero actuals only (undefined at zero; M5 items are intermittent)."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    mask = actual != 0
    return float((np.abs(actual[mask] - forecast[mask]) / np.abs(actual[mask])).mean()) if mask.any() else 0.0


def pinball_loss(actual: pd.Series, forecast: pd.Series, quantile: float = 0.5) -> float:
    error = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    return float(np.maximum(quantile * error, (quantile - 1) * error).mean())


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    """Signed total error relative to total actuals (positive = over-forecast)."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denominator = np.abs(actual).sum()
    return float((forecast - actual).sum() / denominator) if denominator else 0.0


def wrmsse(actual: np.ndarray, forecast: np.ndarray, scale_sq: np.ndarray, weights: np.ndarray) -> float:
    """Weighted RMSSE for one hierarchy level (the M5 accuracy metric).

    ``actual``/``forecast`` are (nodes x horizon). Each node's RMSE is scaled by the
    in-sample one-step naive RMSE (``scale_sq`` is its square) and weighted by its
    share of recent revenue. Nodes with no sales history get zero weight.
    """
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    scale_sq, weights = np.asarray(scale_sq, dtype=float), np.asarray(weights, dtype=float)
    valid = scale_sq > 0
    if not valid.any() or weights[valid].sum() == 0:
        return float("nan")
    rmsse = np.sqrt(np.mean((actual[valid] - forecast[valid]) ** 2, axis=1) / scale_sq[valid])
    return float(np.sum(weights[valid] / weights[valid].sum() * rmsse))


def score(frame: pd.DataFrame, group_columns: list[str], actual: str = "actual", forecast: str = "forecast") -> pd.DataFrame:
    """WMAPE, MAPE, median pinball loss, and bias for every group."""
    def metrics(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "wmape": wmape(group[actual], group[forecast]),
            "mape": mape(group[actual], group[forecast]),
            "pinball_loss": pinball_loss(group[actual], group[forecast]),
            "bias": bias(group[actual], group[forecast]),
            "n": len(group),
        })
    return frame.groupby(group_columns, observed=True)[[actual, forecast]].apply(metrics).reset_index()


def rolling_origin(
    frame: pd.DataFrame,
    forecast_fn,
    origins: list[pd.Timestamp],
    horizon: int = 7,
    key_columns: tuple[str, ...] = ("series_id",),
) -> pd.DataFrame:
    """Evaluate ``forecast_fn(train, horizon)`` at each origin.

    ``forecast_fn`` must return ``key_columns`` + ``date`` + ``forecast``; forecasts
    are joined to actuals on those keys, so row order never matters.
    """
    keys = [*key_columns, "date"]
    rows = []
    for origin in origins:
        train = frame[frame["date"] <= origin]
        test = frame[(frame["date"] > origin) & (frame["date"] <= origin + pd.Timedelta(days=horizon))]
        predicted = forecast_fn(train, horizon)
        joined = test[keys + ["sales"]].merge(predicted[keys + ["forecast"]], on=keys, how="left", validate="one_to_one")
        if joined["forecast"].isna().any():
            raise ValueError(f"Missing forecasts for {int(joined['forecast'].isna().sum())} test rows at origin {origin}")
        rows.append({
            "origin": origin,
            "wmape": wmape(joined["sales"], joined["forecast"]),
            "mape": mape(joined["sales"], joined["forecast"]),
            "pinball_loss": pinball_loss(joined["sales"], joined["forecast"]),
        })
    return pd.DataFrame(rows)


def configure_mlflow(tracking_dir: str | Path, experiment: str) -> None:
    import os
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    tracking_dir = Path(tracking_dir).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(tracking_dir / "artifacts").as_uri())
    mlflow.set_experiment(experiment)


def log_run(
    metrics: dict[str, float],
    params: dict[str, object],
    artifact_path: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    import mlflow

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact_path:
            mlflow.log_artifact(artifact_path)
