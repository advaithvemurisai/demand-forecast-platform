"""FastAPI serving layer: forecasts, allocation, metrics, and live inventory-twin what-ifs."""
import json
import os
import sys
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the API needs forecasting.twin only, not the modelling stack

from forecasting import twin  # noqa: E402

app = FastAPI(title="Hierarchical Demand Forecast Platform", version="0.3.0")
# DATA_DIR points at the gold tables (locally) or the committed data/dashboard extract (hosted).
GOLD_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "gold"))
TWIN_DIR = Path(os.environ.get("TWIN_DIR", ROOT / "data" / "dashboard" / "twin_inputs"))
METRIC_TABLES = {"model_metrics", "reconciliation_metrics", "interval_coverage", "allocation_backtest", "drift"}
TWIN_TABLES = {
    "twin_validation", "twin_timeline", "twin_frontier", "twin_stress", "twin_exceptions", "override_fva", "decision_accuracy",
    "weekday_bias", "event_accuracy", "bias_exceptions", "probable_stockouts", "interval_coverage_segment", "allocation_node_fill", "planning_cycle",
}
RATE_LIMIT, RATE_WINDOW = int(os.environ.get("TWIN_RATE_LIMIT", 30)), 60.0
_hits: dict[str, deque] = defaultdict(deque)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:4173").split(",") if origin.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


class Page(BaseModel):
    total: int
    limit: int
    offset: int
    records: list[dict]


class Scenario(BaseModel):
    demand_scale: float = Field(1.0, ge=0.2, le=3.0)
    category: Literal["FOODS", "HOUSEHOLD", "HOBBIES"] | None = None
    days: tuple[int, int] | None = None
    dc_factor: float = Field(1.0, ge=0.1, le=1.5)
    dc_days: tuple[int, int] | None = None
    delay: int = Field(0, ge=0, le=14)


class TwinRequest(BaseModel):
    store_id: str = Field(pattern=r"^[A-Z]{2}_\d$")
    policy: Literal["forecast_reorder", "last_week_reorder"] = "forecast_reorder"
    service: float | Literal["current"] = "current"
    rationing: Literal["proportional", "days_of_cover", "value"] = "days_of_cover"
    scenario: Scenario | None = None
    reps: int = Field(20, ge=1, le=50)
    seed: int = Field(1, ge=0, le=10_000)


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


def client_ip(request: Request) -> str:
    """Real client address: behind N trusted proxies it is the Nth-from-last X-Forwarded-For entry."""
    hops = int(os.environ.get("TRUSTED_PROXY_HOPS", 0))
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    if hops > 0 and len(forwarded) >= hops:
        return forwarded[-hops]
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    hits = _hits[ip]
    while hits and now - hits[0] > RATE_WINDOW:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail=f"Rate limit: {RATE_LIMIT} simulations per minute")
    hits.append(now)


@lru_cache(maxsize=8)
def load_bundle(store_id: str) -> dict:
    path = TWIN_DIR / f"{store_id}.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No twin inputs for store {store_id}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@lru_cache(maxsize=64)
def cached_simulation(store_id: str, payload: str) -> str:
    return json.dumps(twin.simulate_bundle(load_bundle(store_id), json.loads(payload)))


@app.get("/health")
def health():
    return {"status": "ok", "gold_tables": sorted(path.stem for path in GOLD_DIR.glob("*.parquet")), "twin_stores": sorted(path.stem for path in TWIN_DIR.glob("*.npz"))}


@app.get("/get-forecast", response_model=Page)
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


@app.get("/get-prediction-interval", response_model=Page)
def get_prediction_interval(
    series_id: str | None = None,
    level: str | None = None,
    limit: int = Query(1000, ge=1, le=50_000),
    offset: int = Query(0, ge=0),
):
    return page(read_gold("prediction_intervals.parquet"), limit, offset, series_id=series_id, level=level)


@app.get("/get-allocation", response_model=Page)
def get_allocation(store_id: str | None = None, limit: int = Query(1000, ge=1, le=50_000), offset: int = Query(0, ge=0)):
    frame = read_gold("allocation.parquet")
    return page(frame, limit, offset, store_id=store_id if "store_id" in frame else None)


@app.get("/get-metrics", response_model=Page)
def get_metrics(table: str = "reconciliation_metrics", fold: str | None = None, level: str | None = None, limit: int = Query(1000, ge=1, le=50_000), offset: int = Query(0, ge=0)):
    if table not in METRIC_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown metrics table {table!r}; choose from {sorted(METRIC_TABLES)}")
    frame = read_gold(f"{table}.parquet")
    return page(frame, limit, offset, fold=fold if "fold" in frame else None, level=level if "level" in frame else None)


@app.get("/twin/tables/{name}", response_model=Page)
def get_twin_table(name: str, limit: int = Query(1000, ge=1, le=50_000), offset: int = Query(0, ge=0)):
    if name not in TWIN_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown twin table {name!r}; choose from {sorted(TWIN_TABLES)}")
    return page(read_gold(f"{name}.parquet"), limit, offset)


@app.post("/twin/simulate")
def simulate(body: TwinRequest, request: Request):
    """Run a what-if on one store: baseline and scenario share the same random demand (KPIs are mean and 90% band)."""
    check_rate_limit(client_ip(request))
    payload = body.model_dump(exclude={"store_id"}, mode="json")
    try:
        return json.loads(cached_simulation(body.store_id, json.dumps(payload, sort_keys=True)))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
