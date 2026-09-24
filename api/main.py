"""FastAPI serving layer for forecast and allocation outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Hierarchical Demand Forecast Platform", version="0.2.0")
GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold"
METRIC_TABLES = {"model_metrics", "reconciliation_metrics", "interval_coverage", "allocation_backtest", "drift"}


def read_gold(name: str) -> pd.DataFrame:
    path = GOLD_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Gold output not found: {name}. Run `python -m forecasting.pipeline`.")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def page(frame: pd.DataFrame, limit: int, offset: int, **filters) -> dict:
    for column, value in filters.items():
        if value is not None:
            if column not in frame:
                raise HTTPException(status_code=400, detail=f"Cannot filter on {column}")
            frame = frame[frame[column].eq(value)]
    window = frame.iloc[offset: offset + limit].replace({np.nan: None})
    return {"total": int(len(frame)), "limit": limit, "offset": offset, "records": window.to_dict("records")}


@app.get("/health")
def health():
    return {"status": "ok", "gold_tables": sorted(path.stem for path in GOLD_DIR.glob("*.parquet"))}


@app.get("/get-forecast")
def get_forecast(
    series_id: str | None = None,
    level: str | None = None,
    method: str | None = None,
    limit: int = Query(1000, ge=1, le=50_000),
    offset: int = Query(0, ge=0),
):
    """Production forecasts. ``series_id`` is a hierarchy node id, e.g. ``store:store_id=CA_1``.

    Without ``method`` only the served reconciliation method is returned.
    """
    frame = read_gold("forecasts.parquet")
    if method is None and "served" in frame:
        frame = frame[frame["served"]]  # default: the reconciliation method selected on backtests
    return page(frame, limit, offset, series_id=series_id, level=level, method=method)


@app.get("/get-prediction-interval")
def get_prediction_interval(
    series_id: str | None = None,
    level: str | None = None,
    limit: int = Query(1000, ge=1, le=50_000),
    offset: int = Query(0, ge=0),
):
    return page(read_gold("prediction_intervals.parquet"), limit, offset, series_id=series_id, level=level)


@app.get("/get-allocation")
def get_allocation(store_id: str | None = None, limit: int = Query(1000, ge=1, le=50_000), offset: int = Query(0, ge=0)):
    frame = read_gold("allocation.parquet")
    return page(frame, limit, offset, store_id=store_id if "store_id" in frame else None)


@app.get("/get-metrics")
def get_metrics(table: str = "reconciliation_metrics", fold: str | None = None, level: str | None = None, limit: int = Query(1000, ge=1, le=50_000), offset: int = Query(0, ge=0)):
    if table not in METRIC_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown metrics table {table!r}; choose from {sorted(METRIC_TABLES)}")
    frame = read_gold(f"{table}.parquet")
    return page(frame, limit, offset, fold=fold if "fold" in frame else None, level=level if "level" in frame else None)
