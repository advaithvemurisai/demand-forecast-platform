"""End-to-end pipeline: silver -> features -> models -> reconciliation -> intervals -> allocation -> gold.

Evaluation design
-----------------
* ``n_backtest_folds`` rolling origins, each followed by a ``horizon``-day test window.
* A holdout fold whose test window is the last ``horizon`` days with known actuals
  (the M5 validation period d_1914-d_1941 when ``sales_train_evaluation.csv`` is used).
* The bottom-level model used for reconciliation is selected on backtest folds only,
  so the holdout is never used for any modelling decision.
* Conformal scores for fold *j* come only from folds before *j*, so reported
  interval coverage is out-of-sample.
* A production run trains on all known data and forecasts the next ``horizon`` days
  (d_1942-d_1969), which is what the API and Tableau serve.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import backtest as bt
from forecasting import decision_metrics as dm
from forecasting import probabilistic as prob
from forecasting import twin_runs
from forecasting.allocation import allocate_inventory, fairness_violations, proportional_allocation
from forecasting.data import make_node_ids, prepare_m5, read_silver, write_silver_partitioned
from forecasting.drift import psi, retraining_flag
from forecasting.features import add_features, closed_mask
from forecasting.models import gbm_model
from forecasting.models.baselines import arima_forecast, arima_weekly_forecast, forecast_all
from forecasting.models.prophet_model import calendar_holidays, prophet_forecast
from forecasting.twin import TwinConfig
from forecasting.stockouts import censoring_summary, probable_stockouts
from forecasting.reconciliation import historical_proportions, mint_diagonal, summing_matrix, temporal_reconcile, top_down

log = logging.getLogger("forecasting.pipeline")

BOTTOM_KEYS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
SILVER_COLUMNS = [*BOTTOM_KEYS, "date", "sales", "sell_price", "promo", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
BOTTOM_MODELS = ["naive", "seasonal_naive", "lgbm_global", "lgbm_local"]
RECONCILIATION_METHODS = ["base", "bottom_up", "top_down", "mint_diagonal", "mint_oos"]
COHERENT_METHODS = ["bottom_up", "top_down", "mint_diagonal", "mint_oos"]
PROPHET_LEVELS = ("store", "state", "total")
HISTORY_DAYS = 420  # bottom-level history kept as a matrix for scales, SARIMA, drift, and proportions


@dataclass
class Config:
    root: Path
    states: tuple[str, ...] = ("CA",)
    horizon: int = 28
    history_days: int = 548
    n_backtest_folds: int = 3
    sarima_days: int = 364
    service_level: float = 0.95
    interval_levels: tuple[float, ...] = (0.8, 0.95)
    supply_ratio: float = 0.9
    min_fill: float = 0.5  # every department x store keeps at least this share of its forecast
    fairness_weeks: int = 2  # flag nodes below min_fill for more than this many consecutive weeks
    # Assumed gross margin by category: M5 has prices but no costs, so these are planning assumptions.
    margin_by_category: dict = field(default_factory=lambda: {"FOODS": 0.25, "HOUSEHOLD": 0.30, "HOBBIES": 0.40})
    n_scenarios: int = 25
    lead_time_days: int = 1  # DC -> store transit
    review_days_by_category: dict = field(default_factory=lambda: {"FOODS": 2, "HOUSEHOLD": 7, "HOBBIES": 7})  # days between store orders
    presentation_min: float = 2.0  # units kept on the shelf however slowly a product sells
    # Assumed weekly cost of holding stock, as a share of its value (capital, shrink and, for food, spoilage).
    holding_by_category: dict = field(default_factory=lambda: {"FOODS": 0.015, "HOUSEHOLD": 0.005, "HOBBIES": 0.005})
    case_pack: int = 1
    # Pull aggregate daily base forecasts toward weekly ones before cross-sectional MinT. Opt-in: on the California data it
    # made the base SARIMA slightly worse (state WMAPE 0.0502 vs 0.0462) and left the reconciled accuracy unchanged.
    temporal: bool = False
    twin: bool = True
    twin_reps: int = 50
    mask_stockouts: bool = True  # drop probable out-of-stock days from training and calibration
    n_estimators: int = 300
    prophet: bool = True
    mlflow: bool = True
    max_items: int | None = None
    experiment: str = "m5-hierarchical"
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def silver_dir(self) -> Path:
        return self.root / "data" / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.root / "data" / "gold"


# --------------------------------------------------------------------------- data


def sales_file(raw_dir: Path) -> str:
    """Prefer the evaluation file: it adds 28 days of actuals (d_1914-d_1941) for a true holdout."""
    return "sales_train_evaluation.csv" if (raw_dir / "sales_train_evaluation.csv").exists() else "sales_train_validation.csv"


def ensure_silver(cfg: Config) -> None:
    for state in cfg.states:
        if not (cfg.silver_dir / f"state_id={state}").exists():
            log.info("Building silver partition for %s", state)
            write_silver_partitioned(prepare_m5(cfg.raw_dir, states=[state], sales_file=sales_file(cfg.raw_dir)), cfg.silver_dir)


def assign_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach a categorical item-level ``series_id`` whose codes index the bottom matrix rows."""
    for column in BOTTOM_KEYS:
        frame[column] = frame[column].astype(str).astype("category")
    keys = frame[BOTTOM_KEYS].drop_duplicates(["item_id", "store_id"]).copy()
    keys["series_id"] = make_node_ids("item", keys).to_numpy()
    keys = keys.sort_values("series_id").reset_index(drop=True)
    n_stores = len(frame["store_id"].cat.categories)
    lookup = np.full(len(frame["item_id"].cat.categories) * n_stores, -1)
    keys_item = pd.Categorical(keys["item_id"], categories=frame["item_id"].cat.categories).codes.astype(np.int64)
    keys_store = pd.Categorical(keys["store_id"], categories=frame["store_id"].cat.categories).codes.astype(np.int64)
    lookup[keys_item * n_stores + keys_store] = np.arange(len(keys))
    position = lookup[frame["item_id"].cat.codes.astype(np.int64) * n_stores + frame["store_id"].cat.codes.astype(np.int64)]
    frame["series_id"] = pd.Categorical.from_codes(position, categories=keys["series_id"])
    return frame, keys


def future_rows(history: pd.DataFrame, keys: pd.DataFrame, raw_dir: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows for the production horizon: known calendar and prices, unknown sales."""
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    calendar = calendar[calendar["date"].isin(dates)]
    grid = keys[BOTTOM_KEYS].merge(calendar, how="cross")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["store_id"].isin(keys["store_id"].unique()) & prices["wm_yr_wk"].isin(calendar["wm_yr_wk"].unique())]
    grid = grid.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    median_price = history.groupby("series_id", observed=True)["sell_price"].median()
    grid["series_id"] = make_node_ids("item", grid).to_numpy()
    grid["promo"] = (grid["sell_price"] < grid["series_id"].map(median_price)).astype("int8")
    grid["sales"] = np.nan
    return grid[[column for column in [*SILVER_COLUMNS, "series_id"] if column in grid]]


def load_frame(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Load the modelling window, append production rows, and compute features."""
    ensure_silver(cfg)
    probe = read_silver(cfg.silver_dir, cfg.states, columns=["date"])
    last_actual = probe["date"].max()
    del probe
    holdout_origin = last_actual - pd.Timedelta(days=cfg.horizon)
    first_origin = holdout_origin - pd.Timedelta(days=cfg.horizon * cfg.n_backtest_folds)
    start = first_origin - pd.Timedelta(days=max(cfg.history_days, HISTORY_DAYS) + 90)
    frame = read_silver(cfg.silver_dir, cfg.states, columns=SILVER_COLUMNS, start=start)
    if cfg.max_items:
        chosen = sorted(frame["item_id"].astype(str).unique())[: cfg.max_items]
        frame = frame[frame["item_id"].astype(str).isin(chosen)].copy()
    frame["sales"] = frame["sales"].astype("float32")
    frame, keys = assign_series(frame)
    future_dates = pd.date_range(last_actual + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")
    future = future_rows(frame, keys, cfg.raw_dir, future_dates)
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype) and column in future:
            categories = frame[column].cat.categories.union(pd.Index(future[column].dropna().astype(str).unique()))
            frame[column] = frame[column].cat.set_categories(categories)
            future[column] = pd.Categorical(future[column], categories=categories)
    combined = pd.concat([frame, future], ignore_index=True)
    del frame, future
    features = add_features(combined, min_lag=cfg.horizon)
    features["series_id"] = pd.Categorical(features["series_id"], categories=keys["series_id"])
    origins = {f"backtest_{i + 1}": first_origin + pd.Timedelta(days=cfg.horizon * i) for i in range(cfg.n_backtest_folds)}
    origins["holdout"] = holdout_origin
    origins["production"] = last_actual
    log.info("Loaded %s rows, %s series, last actual %s", f"{len(features):,}", f"{len(keys):,}", last_actual.date())
    return features, keys, origins


def to_matrix(frame: pd.DataFrame, value: str, dates: pd.DatetimeIndex, n_series: int) -> np.ndarray:
    """Pivot a long frame into a (series x date) matrix using series codes."""
    matrix = np.full((n_series, len(dates)), np.nan)
    rows = frame["series_id"].cat.codes.to_numpy()
    cols = dates.get_indexer(frame["date"])
    keep = (rows >= 0) & (cols >= 0)
    matrix[rows[keep], cols[keep]] = frame[value].to_numpy(dtype=float)[keep]
    return matrix


# --------------------------------------------------------------------------- models


def mad_scale(history: np.ndarray, days: int = 56, floor: float = 0.1) -> np.ndarray:
    """Per-node error scale: mean absolute weekly-seasonal difference over recent history."""
    recent = history[:, -(days + 7):]
    return np.maximum(np.nanmean(np.abs(recent[:, 7:] - recent[:, :-7]), axis=1), floor)


def weekly(matrix: np.ndarray) -> np.ndarray:
    weeks = matrix.shape[1] // 7
    return matrix[:, : weeks * 7].reshape(matrix.shape[0], weeks, 7).sum(axis=2)


def fit_bottom_models(features: pd.DataFrame, origin: pd.Timestamp, cfg: Config, dates: pd.DatetimeIndex, n_series: int, models: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Forecast every bottom series with the requested models; returns forecasts and in-sample residual variances."""
    window = features[(features["date"] > origin - pd.Timedelta(days=cfg.history_days)) & (features["date"] <= origin)]
    train = window[window["sell_price"].notna() & window["rolling_mean_56"].notna()]
    train = train[~closed_mask(train)]  # closure days are zero by closure, not by demand
    if cfg.mask_stockouts:
        window_dates = pd.date_range(origin - pd.Timedelta(days=cfg.history_days - 1), origin, freq="D")
        flags = probable_stockouts(to_matrix(window, "sales", window_dates, n_series))
        rows, cols = train["series_id"].cat.codes.to_numpy(), window_dates.get_indexer(train["date"])
        train = train[~flags[rows, cols]]
        log.info("  masked %.1f%% of training rows as probable stockouts", 100 * (1 - len(train) / max(len(window), 1)))
    test = features[features["date"].isin(dates)]
    calib = window[window["date"] > origin - pd.Timedelta(days=cfg.horizon)]
    on_shelf = test["sell_price"].notna().to_numpy()
    forecasts, variances = {}, {}
    params = {"n_estimators": cfg.n_estimators}

    if {"naive", "seasonal_naive"} & set(models):
        recent = features.loc[(features["date"] <= origin) & (features["date"] > origin - pd.Timedelta(days=14)), ["series_id", "date", "sales"]]
        floors = forecast_all(recent.assign(series_id=recent["series_id"].astype(str)), cfg.horizon)
        floors["series_id"] = pd.Categorical(floors["series_id"], categories=features["series_id"].cat.categories)
        for model in ("naive", "seasonal_naive"):
            if model in models:
                forecasts[model] = to_matrix(floors[floors["model"] == model], "forecast", dates, n_series)

    def residual_variance(predicted: np.ndarray) -> np.ndarray:
        residual = calib.assign(residual=calib["sales"].to_numpy() - predicted)
        return residual.groupby("series_id", observed=False)["residual"].var().fillna(0).to_numpy()

    if "lgbm_global" in models:
        started = time.time()
        model = gbm_model.fit_global(train, params=params)
        predicted = gbm_model.predict(model, test) * on_shelf * ~closed_mask(test)
        forecasts["lgbm_global"] = to_matrix(test.assign(forecast=predicted), "forecast", dates, n_series)
        variances["lgbm_global"] = residual_variance(gbm_model.predict(model, calib) * calib["sell_price"].notna().to_numpy())
        log.info("  lgbm_global fit+predict in %.0fs", time.time() - started)
    if "lgbm_local" in models:
        started = time.time()
        local = gbm_model.fit_local(train, params=params)
        predicted = gbm_model.predict_local(local, test) * on_shelf * ~closed_mask(test)
        forecasts["lgbm_local"] = to_matrix(test.assign(forecast=predicted), "forecast", dates, n_series)
        variances["lgbm_local"] = residual_variance(gbm_model.predict_local(local, calib) * calib["sell_price"].notna().to_numpy())
        log.info("  lgbm_local fit+predict in %.0fs", time.time() - started)
    return forecasts, variances


def fit_aggregate_models(history_agg: np.ndarray, nodes: pd.DataFrame, history_dates: pd.DatetimeIndex, cfg: Config, holidays: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """SARIMA base forecasts for every aggregate node, Prophet for the top levels."""
    sarima = np.zeros((len(nodes), cfg.horizon))
    variance = np.zeros(len(nodes))
    for row in range(len(nodes)):
        series = pd.Series(history_agg[row, -cfg.sarima_days:])
        sarima[row], variance[row] = arima_forecast(series, cfg.horizon)
    prophet = {}
    if cfg.prophet:
        for row in np.flatnonzero(nodes["level"].isin(PROPHET_LEVELS).to_numpy()):
            history = pd.DataFrame({"date": history_dates, "sales": history_agg[row]})
            prophet[row] = prophet_forecast(history, cfg.horizon, holidays=holidays)["forecast"].to_numpy()
    return sarima, variance, prophet


def fit_weekly_models(history_agg: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Weekly-total ARIMA base forecasts and residual variances for every aggregate node."""
    weeks = cfg.horizon // 7
    usable = (min(cfg.sarima_days, history_agg.shape[1]) // 7) * 7
    forecast, variance = np.zeros((len(history_agg), weeks)), np.zeros(len(history_agg))
    for row in range(len(history_agg)):
        weekly_history = history_agg[row, -usable:].reshape(-1, 7).sum(axis=1)
        forecast[row], variance[row] = arima_weekly_forecast(weekly_history, weeks)
    return forecast, variance


# --------------------------------------------------------------------------- evaluation helpers


def segment_labels(history_bottom: np.ndarray, keys: pd.DataFrame, nodes: pd.DataFrame, rows_by_level: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calibration segments per level: item = velocity x category, department = category, others one segment."""
    mean_daily = history_bottom[:, -56:].mean(axis=1)
    labels = {"item": np.char.add(np.char.add(prob.velocity_class(mean_daily).astype(str), "|"), keys["cat_id"].astype(str).to_numpy().astype(str))}
    for level, rows in rows_by_level.items():
        if level == "department":
            labels[level] = nodes.iloc[rows]["node_id"].str.extract(r"dept_id=([A-Z]+)_")[0].to_numpy().astype(str)
        else:
            labels[level] = np.full(len(rows), "all")
    return labels


def protection_days(keys: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Days of demand a store order must cover: lead time plus the days until the next order."""
    review = keys["cat_id"].astype(str).map(cfg.review_days_by_category).fillna(7).to_numpy()
    return (cfg.lead_time_days + review).astype(int)


def window_sums(matrix: np.ndarray, days: int) -> np.ndarray:
    """Sums over consecutive non-overlapping ``days``-day windows: (series, windows)."""
    windows = matrix.shape[1] // days
    return matrix[:, : windows * days].reshape(len(matrix), windows, days).sum(axis=2)


def item_safety_stock(cfg: Config, keys: pd.DataFrame, production: dict, serve_bottom: np.ndarray, protect: dict, service_override: float | None = None) -> pd.DataFrame:
    """Per-product stock target over the protection window, calibrated by segment and service class."""
    n = len(keys)
    days = protection_days(keys, cfg)
    history = production["history_bottom"]
    abc = prob.abc_class(production["scaling"][3])
    xyz = prob.xyz_class(weekly(history[:, -56:]))
    target = prob.service_targets(abc, xyz) if service_override is None else np.full(n, float(service_override))
    segments = production["segments"]["item"]
    point, safety, order_up_to, risk, fill = (np.zeros(n) for _ in range(5))
    for window in np.unique(days):
        rows = days == window
        scores, groups = np.concatenate(protect["scores"][window]), np.concatenate(protect["groups"][window])
        scale = np.maximum(history[rows][:, -56:].mean(axis=1) * window, 1.0)
        forecast = window_sums(serve_bottom[rows], window)[:, 0]
        point[np.flatnonzero(rows)] = forecast
        for segment in np.unique(segments[rows]):
            for service in np.unique(target[rows]):
                block = (segments == segment) & (target == service) & rows
                if not block.any():
                    continue
                calibration = scores[groups == segment]
                if np.isfinite(calibration).sum() < 500:
                    calibration = scores
                inside = np.flatnonzero(rows)
                local = np.isin(inside, np.flatnonzero(block))
                block_scale, block_point = scale[local], forecast[local]
                ss = prob.service_level_safety_stock(block_point, calibration, float(service), block_scale)
                target_stock = prob.round_to_case(np.maximum(ss["order_up_to"].to_numpy(), cfg.presentation_min), cfg.case_pack)
                idx = np.flatnonzero(block)
                safety[idx] = ss["safety_stock"].to_numpy()
                order_up_to[idx] = target_stock
                risk[idx] = prob.stockout_risk(block_point, target_stock, calibration, block_scale)
                fill[idx] = prob.expected_fill_rate(block_point, target_stock, calibration, block_scale)
    return pd.DataFrame({
        "forecast": point, "safety_stock": safety, "order_up_to": order_up_to, "stockout_risk": risk,
        "expected_fill_rate": fill, "service_level": target, "abc": abc, "xyz": xyz, "protection_days": days,
    })


def twin_phase(cfg: Config, keys: pd.DataFrame, folds: dict, evaluated: list[str], daily_scores: dict, protect_snaps: list, protect: dict, served_method: str) -> dict[str, pd.DataFrame]:
    """Phase 4: the inventory twin on every fold with calibration data, the holdout, and the production window."""
    n = len(keys)
    keys = keys.reset_index(drop=True)
    tcfg = TwinConfig(lead_time_days=cfg.lead_time_days, presentation_min=cfg.presentation_min, case_pack=cfg.case_pack, seed=0)
    review = protection_days(keys, cfg) - cfg.lead_time_days

    def past(upto: int) -> tuple[np.ndarray, np.ndarray]:
        """What sold and what was forecast in the first ``upto`` windows: the history the demand bootstrap resamples."""
        actual = np.hstack([np.nan_to_num(folds[evaluated[k]]["actual_bottom"]) for k in range(upto)])
        forecast = np.hstack([folds[evaluated[k]]["reconciled"][served_method][1] for k in range(upto)])
        return actual, forecast

    def inputs(state, snap, actual, upto):
        serve_bottom = state["reconciled"][served_method][1]
        safety = item_safety_stock(cfg, keys, state, serve_bottom, snap)["safety_stock"].to_numpy()
        units = state["history_bottom"][:, -28:].sum(axis=1)
        price = np.divide(state["scaling"][3], units, out=np.zeros(n), where=units > 0)
        price = np.where(price > 0, price, np.median(price[price > 0]) if (price > 0).any() else 1.0)
        return twin_runs.FoldInputs(keys, state["history_bottom"], serve_bottom, actual, np.ones(n), safety, review, price, state["dates"], *past(upto))

    validation = []
    for index, fold in enumerate(evaluated):
        if index == 0:
            continue  # nothing earlier to calibrate on
        inp = inputs(folds[fold], protect_snaps[index], folds[fold]["actual_bottom"], index)
        validation += twin_runs.validation_rows(fold, inp, tcfg, twin_runs.POLICY_LABELS, cfg.twin_reps, None, seed=100 * index)
    last = len(evaluated) - 1
    holdout = inputs(folds[evaluated[last]], protect_snaps[last], folds[evaluated[last]]["actual_bottom"], last)
    timeline = twin_runs.timeline_frame(holdout, tcfg, "forecast_reorder", seed=7)
    state_h, snap_h = folds[evaluated[last]], protect_snaps[last]
    safety_for = lambda level: item_safety_stock(cfg, keys, state_h, state_h["reconciled"][served_method][1], snap_h, service_override=level)["safety_stock"].to_numpy()
    frontier = twin_runs.frontier_frame(holdout, tcfg, safety_for, holdout.safety, cfg.margin_by_category, cfg.holding_by_category, cfg.twin_reps, None, seed=300)
    stress = twin_runs.stress_frame(holdout, tcfg, twin_runs.POLICY_LABELS, cfg.twin_reps, None, seed=500)
    production = inputs(folds["production"], protect, None, len(evaluated))
    exceptions = twin_runs.exceptions_frame(production, tcfg, "last_week_reorder", cfg.twin_reps, None, seed=700)
    write_twin_inputs(cfg, keys, production, folds["production"], protect)
    return {
        "twin_validation": pd.DataFrame(validation), "twin_timeline": timeline, "twin_frontier": frontier,
        "twin_stress": stress, "twin_exceptions": exceptions,
    }


def write_twin_inputs(cfg: Config, keys: pd.DataFrame, production, state: dict, protect: dict) -> None:
    """Per-store input bundles for what-if runs (API, build script, browser): the production window's forecast and calibration."""
    out = cfg.root / "data" / "dashboard" / "twin_inputs"
    out.mkdir(parents=True, exist_ok=True)
    served = production.forecast
    by_service = np.stack([
        item_safety_stock(cfg, keys, state, served, protect, service_override=level)["safety_stock"].to_numpy() for level in twin_runs.SERVICE_GRID
    ], axis=1).astype("float32")
    store = keys["store_id"].astype(str).to_numpy()
    for name in sorted(set(store)):
        rows = np.flatnonzero(store == name)
        np.savez_compressed(
            out / f"{name}.npz",
            forecast=served[rows].astype("float32"), history=production.history[rows][:, -28:].astype("float32"),
            scale=production.scale[rows].astype("float32"), current_safety=production.safety[rows].astype("float32"),
            safety_by_service=by_service[rows], review=production.review[rows].astype("int8"), price=production.price[rows].astype("float32"),
            past_actual=production.past_actual[rows].astype("float16"), past_forecast=production.past_forecast[rows].astype("float16"), dept_ids=keys["dept_id"].astype(str).to_numpy().astype("U16")[rows], cat_ids=keys["cat_id"].astype(str).to_numpy().astype("U16")[rows],
            item_ids=keys["item_id"].astype(str).to_numpy().astype("U24")[rows], dates=production.dates.strftime("%Y-%m-%d").to_numpy().astype("U10"),
            lead_time_days=np.int8(cfg.lead_time_days), presentation_min=np.float32(cfg.presentation_min), case_pack=np.int8(cfg.case_pack),
        )


def twin_summary(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Headline twin numbers for the run summary, README and drift check."""
    validation = tables["twin_validation"]
    network = validation[(validation["store_id"] == "all") & (validation["policy"] == "forecast_reorder")]
    latest = network[network["fold"] == network["fold"].iloc[-1]].set_index("metric")
    frontier = tables["twin_frontier"]
    stress = tables["twin_stress"]
    lost = stress[(stress["metric"] == "lost_sales_value") & (stress["policy"] == "forecast_reorder")].set_index("scenario")["delta"]
    error = (network["predicted"] - network["realised"]).abs()
    miss = {
        metric: float(error[network["metric"] == metric].mean() / (1 if metric in ("fill_rate", "in_stock_pct") else network.loc[network["metric"] == metric, "realised"].abs().mean()))
        for metric in network["metric"].unique()
    }
    return {
        "validation_fold": str(network["fold"].iloc[-1]),
        "fill_rate": {"predicted": float(latest.loc["fill_rate", "predicted"]), "realised": float(latest.loc["fill_rate", "realised"]),
                      "lower": float(latest.loc["fill_rate", "lower"]), "upper": float(latest.loc["fill_rate", "upper"])},
        "share_realised_in_band": float(network["in_band"].mean()),
        # Mean absolute miss across windows: points for rates, share of the realised value otherwise.
        "typical_miss": miss,
        "fill_rate_gap": float(abs(latest.loc["fill_rate", "predicted"] - latest.loc["fill_rate", "realised"])),
        "recommended_service": {row.category: float(row.service_level) for row in frontier[frontier["recommended"]].itertuples()},
        "stress_added_lost_sales": {name: float(value) for name, value in lost.items()},
    }


def level_rows(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = {level: np.flatnonzero(nodes["level"].to_numpy() == level) for level in nodes["level"].unique()}
    return rows


def level_metrics(actual_agg, forecast_agg, actual_bottom, forecast_bottom, rows_by_level, levels=None, scaling=None) -> list[dict]:
    """Per-level accuracy. ``scaling`` = (scale_sq_agg, scale_sq_bottom, weight_agg, weight_bottom) adds WRMSSE."""
    output = []
    for level, rows in rows_by_level.items():
        if levels is None or level in levels:
            row = {"level": level, **_metrics(actual_agg[rows].ravel(), forecast_agg[rows].ravel())}
            if scaling is not None:
                row["wrmsse"] = bt.wrmsse(actual_agg[rows], forecast_agg[rows], scaling[0][rows], scaling[2][rows])
            output.append(row)
    if forecast_bottom is not None and (levels is None or "item" in levels):
        row = {"level": "item", **_metrics(actual_bottom.ravel(), forecast_bottom.ravel())}
        if scaling is not None:
            row["wrmsse"] = bt.wrmsse(actual_bottom, forecast_bottom, scaling[1], scaling[3])
        output.append(row)
    return output


def naive_scale_sq(history: np.ndarray) -> np.ndarray:
    """M5 RMSSE denominator: mean squared one-step naive error since each series' first sale."""
    started = np.cumsum(history > 0, axis=1) > 0
    diffs = np.diff(history, axis=1) ** 2
    valid = started[:, :-1]
    counts = valid.sum(axis=1)
    scale = np.divide((diffs * valid).sum(axis=1), counts, out=np.zeros(len(history)), where=counts > 0)
    return scale


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    return {
        "wmape": bt.wmape(actual, forecast),
        "mape": bt.mape(actual, forecast),
        "pinball_loss": bt.pinball_loss(actual, forecast),
        "bias": bt.bias(actual, forecast),
    }


def reconcile(sarima, sarima_var, bottom, bottom_var, A, history_bottom, keys, dates, oos_var=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return (aggregate, bottom) forecasts for each reconciliation method.

    ``mint_diagonal`` weights by in-sample residual variance. ``mint_oos`` weights by
    the base forecasts' mean squared error in the preceding out-of-sample window
    (``oos_var`` = (aggregate, bottom)); it falls back to in-sample when none exists.
    """
    total_row = A.shape[0] - 1  # AGGREGATE_LEVELS ends with "total"
    results = {"base": (sarima, bottom), "bottom_up": (np.asarray(A @ bottom), bottom)}
    recent = history_bottom[:, -91:]
    long = pd.DataFrame({
        "node_id": np.repeat(keys["series_id"].to_numpy(), recent.shape[1]),
        "date": np.tile(np.arange(recent.shape[1]), recent.shape[0]),
        "sales": recent.ravel(),
    })
    proportions = historical_proportions(long)
    total = pd.DataFrame({"date": dates, "total_forecast": sarima[total_row]})
    td = top_down(proportions, total)
    td["series_id"] = pd.Categorical(td["node_id"], categories=keys["series_id"])
    td_bottom = to_matrix(td, "forecast", dates, len(keys))
    results["top_down"] = (np.asarray(A @ td_bottom), td_bottom)
    _, mint_bottom = mint_diagonal(sarima, bottom, A, sarima_var, bottom_var)
    mint_bottom = np.clip(mint_bottom, 0, None)  # re-aggregate after clipping so the result stays coherent
    results["mint_diagonal"] = (np.asarray(A @ mint_bottom), mint_bottom)
    oos_agg, oos_bottom = oos_var if oos_var is not None else (sarima_var, bottom_var)
    _, oos = mint_diagonal(sarima, bottom, A, oos_agg, oos_bottom)
    oos = np.clip(oos, 0, None)
    results["mint_oos"] = (np.asarray(A @ oos), oos)
    return results


# --------------------------------------------------------------------------- pipeline


def run(cfg: Config) -> dict[str, object]:
    started = time.time()
    features, keys, origins = load_frame(cfg)
    cfg.timings["load_and_features"] = time.time() - started
    A, nodes = summing_matrix(keys)
    rows_by_level = level_rows(nodes)
    n_series = len(keys)
    calendar = pd.read_csv(cfg.raw_dir / "calendar.csv", parse_dates=["date"])
    holidays = calendar_holidays(calendar) if cfg.prophet else None
    weights = dept_prices(features, origins["production"], nodes, keys)
    evaluated = [name for name in origins if name != "production"]

    # Phase 1: fit every candidate model on every evaluation fold.
    folds, model_rows = {}, []
    for fold in [*evaluated, "production"]:
        if fold == "production":
            backtest = pd.DataFrame([row for row in model_rows if row["fold"].startswith("backtest") and row["level"] == "item" and row["model"] in BOTTOM_MODELS])
            selected_model = backtest.groupby("model")["wmape"].mean().idxmin()
            log.info("Selected bottom model on backtest folds only: %s", selected_model)
        fold_started = time.time()
        origin = origins[fold]
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")
        history_dates = pd.date_range(origin - pd.Timedelta(days=HISTORY_DAYS - 1), origin, freq="D")
        history_bottom = np.nan_to_num(to_matrix(features[features["date"].isin(history_dates)], "sales", history_dates, n_series))
        history_agg = np.asarray(A @ history_bottom)
        log.info("Fold %s: origin %s", fold, origin.date())
        models = [selected_model] if fold == "production" else BOTTOM_MODELS
        forecasts, variances = fit_bottom_models(features, origin, cfg, dates, n_series, models)
        sarima, sarima_var, prophet = fit_aggregate_models(history_agg, nodes, history_dates, cfg, holidays)
        sarima_daily = sarima
        if cfg.temporal and cfg.horizon % 7 == 0:
            weekly_forecast, weekly_var = fit_weekly_models(history_agg, cfg)
            sarima = temporal_reconcile(sarima_daily, weekly_forecast, sarima_var, weekly_var)
        recent = features[(features["date"] > origin - pd.Timedelta(days=28)) & (features["date"] <= origin)]
        revenue_bottom = np.nansum(to_matrix(recent.assign(revenue=recent["sales"] * recent["sell_price"].fillna(0)), "revenue", history_dates[-28:], n_series), axis=1)
        scaling = (naive_scale_sq(history_agg), naive_scale_sq(history_bottom), np.asarray(A @ revenue_bottom), revenue_bottom)
        state = {
            "dates": dates, "history_dates": history_dates, "history_bottom": history_bottom, "history_agg": history_agg,
            "forecasts": forecasts, "variances": variances, "sarima": sarima, "sarima_var": sarima_var, "scaling": scaling,
        }
        if fold != "production":
            actual_bottom = np.nan_to_num(to_matrix(features[features["date"].isin(dates)], "sales", dates, n_series))
            actual_agg = np.asarray(A @ actual_bottom)
            state.update(actual_bottom=actual_bottom, actual_agg=actual_agg)
            state["censored"] = probable_stockouts(np.concatenate([history_bottom, actual_bottom], axis=1))[:, -cfg.horizon:]
            for model, bottom in forecasts.items():
                for row in level_metrics(actual_agg, np.asarray(A @ bottom), actual_bottom, bottom, rows_by_level, scaling=scaling):
                    model_rows.append({"fold": fold, "model": model, "approach": "bottom_up", **row})
            for row in level_metrics(actual_agg, sarima, None, None, rows_by_level, scaling=scaling):
                model_rows.append({"fold": fold, "model": "sarima", "approach": "direct", **row})
            if sarima is not sarima_daily:  # keep the untouched daily SARIMA for an honest comparison
                for row in level_metrics(actual_agg, sarima_daily, None, None, rows_by_level, scaling=scaling):
                    model_rows.append({"fold": fold, "model": "sarima_daily_only", "approach": "direct", **row})
            if prophet:
                prophet_matrix = np.zeros_like(sarima)
                for index, values in prophet.items():
                    prophet_matrix[index] = values
                for row in level_metrics(actual_agg, prophet_matrix, None, None, rows_by_level, levels=PROPHET_LEVELS, scaling=scaling):
                    model_rows.append({"fold": fold, "model": "prophet", "approach": "direct", **row})
        folds[fold] = state
        cfg.timings[f"models_{fold}"] = time.time() - fold_started
        log.info("Fold %s models done in %.0fs", fold, cfg.timings[f"models_{fold}"])

    # Phase 2: reconcile with the selected model. Folds are contiguous 28-day windows,
    # so the previous fold's test errors are exactly the out-of-sample errors before
    # each origin; mint_oos weights by them.
    recon_rows, backtest_forecasts = [], []
    dept_rows = rows_by_level["department"]
    previous = None
    for fold in [*evaluated, "production"]:
        state = folds[fold]
        dates = state["dates"]
        oos_var = None
        if previous is not None:
            oos_var = (
                np.mean((previous["actual_agg"] - previous["sarima"]) ** 2, axis=1),
                np.mean((previous["actual_bottom"] - previous["forecasts"][selected_model]) ** 2, axis=1),
            )
        reconciled = reconcile(state["sarima"], state["sarima_var"], state["forecasts"][selected_model], state["variances"][selected_model], A, state["history_bottom"], keys, dates, oos_var)
        state["reconciled"] = reconciled
        state["scale_agg"], state["scale_bottom"] = mad_scale(state["history_agg"]), mad_scale(state["history_bottom"])
        state["segments"] = segment_labels(state["history_bottom"], keys, nodes, rows_by_level)
        state["week_scale_dept"] = np.maximum(weekly(state["history_agg"][dept_rows][:, -56:]).mean(axis=1), 1.0)
        state["week_scale_item"] = np.maximum(weekly(state["history_bottom"][:, -56:]).mean(axis=1), 1.0)
        if fold != "production":
            for method, (agg, bottom) in reconciled.items():
                for row in level_metrics(state["actual_agg"], agg, state["actual_bottom"], bottom, rows_by_level, scaling=state["scaling"]):
                    recon_rows.append({"fold": fold, "method": method, "bottom_model": selected_model, "oos_weights": previous is not None, **row})
                backtest_forecasts.append(long_forecasts(nodes, agg, dates, fold, method, state["actual_agg"]))
            previous = state

    # Serve the coherent method with the best mean WRMSSE on backtest folds where every
    # method had out-of-sample weights (never the holdout).
    candidates = pd.DataFrame(recon_rows)
    backtest = candidates["fold"].str.startswith("backtest") & candidates["method"].isin(COHERENT_METHODS)
    if (backtest & candidates["oos_weights"]).any():
        backtest &= candidates["oos_weights"]
    served_method = candidates[backtest].groupby("method")["wrmsse"].mean().idxmin()
    log.info("Served reconciliation method (backtest WRMSSE): %s", served_method)

    # Phase 3: intervals and allocation for the served method, calibrated on earlier folds only.
    coverage_rows, allocation_rows, node_fill_rows, segment_rows = [], [], [], []
    daily_scores = {level: [] for level in [*rows_by_level, "item"]}
    daily_groups = {level: [] for level in daily_scores}
    weekly_scores = {"department": [], "item": []}
    item_days = protection_days(keys, cfg)
    protect = {"scores": {int(w): [] for w in np.unique(item_days)}, "groups": {int(w): [] for w in np.unique(item_days)}}
    event_dates = pd.DatetimeIndex(calendar.loc[calendar["event_name_1"].notna(), "date"])
    decision_rows, weekday_rows, event_rows, week_actual, week_forecast, fva_rows = [], [], [], [], [], []
    protect_snaps = []
    for fold in evaluated:
        protect_snaps.append({key: {window: list(parts) for window, parts in protect[key].items()} for key in protect})  # calibration available before this fold
        state = folds[fold]
        actual_agg, actual_bottom = state["actual_agg"], state["actual_bottom"]
        serve_agg, serve_bottom = state["reconciled"][served_method]
        for level in daily_scores:
            if level == "item":
                actual, forecast, scale = actual_bottom, serve_bottom, state["scale_bottom"]
            else:
                rows = rows_by_level[level]
                actual, forecast, scale = actual_agg[rows], serve_agg[rows], state["scale_agg"][rows]
            groups = np.repeat(state["segments"][level], forecast.shape[1])
            if daily_scores[level]:
                prior, prior_groups = np.concatenate(daily_scores[level]), np.concatenate(daily_groups[level])
                for nominal in cfg.interval_levels:
                    interval = prob.mondrian_interval(forecast.ravel(), prior, prior_groups, groups, nominal, np.repeat(scale, forecast.shape[1]))
                    covered = ((actual.ravel() >= interval["lower"].to_numpy()) & (actual.ravel() <= interval["upper"].to_numpy()))
                    coverage_rows.append({
                        "fold": fold, "level": level, "nominal": nominal,
                        "coverage": float(covered.mean()),
                        "mean_width": float((interval["upper"] - interval["lower"]).mean()),
                    })
                    if level in ("item", "department"):
                        for segment in np.unique(groups):
                            mask = groups == segment
                            segment_rows.append({"fold": fold, "level": level, "segment": segment, "nominal": nominal, "coverage": float(covered[mask].mean()), "n": int(mask.sum())})
            scores = prob.conformal_scores(actual, forecast, scale[:, None])
            if level == "item" and cfg.mask_stockouts:
                scores = np.where(state["censored"], np.nan, scores)  # zero sales on a stockout day understate demand
            daily_scores[level].append(scores.ravel())
            daily_groups[level].append(groups)
        dept_actual_week, dept_forecast_week = weekly(actual_agg[dept_rows]), weekly(serve_agg[dept_rows])
        if weekly_scores["department"]:
            allocation_rows.extend(evaluate_allocation(fold, nodes.iloc[dept_rows], dept_forecast_week, dept_actual_week, state["week_scale_dept"], np.concatenate(weekly_scores["department"]), weights, cfg, node_fill_rows))
        weekly_scores["department"].append(prob.conformal_scores(dept_actual_week, dept_forecast_week, state["week_scale_dept"][:, None]).ravel())
        item_week_scores = prob.conformal_scores(weekly(actual_bottom), weekly(serve_bottom), state["week_scale_item"][:, None])
        if cfg.mask_stockouts:
            item_week_scores = np.where(weekly(state["censored"].astype(float)) > 0, np.nan, item_week_scores)
        weekly_scores["item"].append(item_week_scores.ravel())
        velocity = np.array([label.split("|")[0] for label in state["segments"]["item"]])
        decision_rows.append(dm.decision_accuracy(actual_bottom, serve_bottom, keys, item_days, velocity, fold))
        weekday_rows.append(dm.bias_by_weekday(actual_bottom, serve_bottom, state["dates"], fold))
        if week_actual:  # override demo: correct products the forecast has been missing in one direction
            multiplier, flagged = dm.bias_correction(np.hstack(week_actual), np.hstack(week_forecast))
            a7, f7 = weekly(actual_bottom), weekly(serve_bottom)
            for scope, mask in (("flagged products", flagged), ("all products", None)):
                fva_rows.append({"fold": fold, "scope": scope, **dm.forecast_value_added(a7, f7, f7 * multiplier[:, None], mask)})
        week_actual.append(weekly(actual_bottom))
        week_forecast.append(weekly(serve_bottom))
        total = rows_by_level["total"]
        event_days = np.asarray(state["dates"].isin(event_dates))
        for day_type, mask in (("event", event_days), ("normal", ~event_days)):
            if mask.any():
                item_wmape, item_bias = dm.wmape_bias(actual_bottom[:, mask], serve_bottom[:, mask])
                total_wmape, total_bias = dm.wmape_bias(actual_agg[total][:, mask], serve_agg[total][:, mask])
                event_rows.append({"fold": fold, "day_type": day_type, "days": int(mask.sum()), "item_wmape": item_wmape, "item_bias": item_bias, "total_wmape": total_wmape, "total_bias": total_bias})
        for window in protect["scores"]:
            rows = item_days == window
            window_scale = np.maximum(state["history_bottom"][rows][:, -56:].mean(axis=1) * window, 1.0)[:, None]
            window_scores = prob.conformal_scores(window_sums(actual_bottom[rows], window), window_sums(serve_bottom[rows], window), window_scale)
            if cfg.mask_stockouts:
                window_scores = np.where(window_sums(state["censored"][rows].astype(float), window) > 0, np.nan, window_scores)
            protect["scores"][window].append(window_scores.ravel())
            protect["groups"][window].append(np.repeat(state["segments"]["item"][rows], window_scores.shape[1]))

    outputs = build_gold(cfg, keys, nodes, rows_by_level, folds["production"], daily_scores, weekly_scores, weights, model_rows, recon_rows, coverage_rows, allocation_rows, backtest_forecasts, served_method, origins, node_fill_rows, daily_groups, segment_rows, protect)
    folds_production = folds["production"]
    units_28 = folds_production["history_bottom"][:, -28:].sum(axis=1)
    unit_price = np.divide(folds_production["scaling"][3], units_28, out=np.zeros(n_series), where=units_28 > 0)
    outputs.update({
        "decision_accuracy": pd.concat(decision_rows, ignore_index=True),
        "weekday_bias": pd.concat(weekday_rows, ignore_index=True),
        "event_accuracy": pd.DataFrame(event_rows),
        "override_fva": pd.DataFrame(fva_rows),
        "bias_exceptions": dm.bias_exceptions(np.hstack(week_actual), np.hstack(week_forecast), keys, unit_price),
    })
    twin_info = None
    if cfg.twin:
        twin_started = time.time()
        outputs.update(twin_phase(cfg, keys, folds, evaluated, daily_scores, protect_snaps, protect, served_method))
        twin_info = twin_summary(outputs)
        if twin_info["fill_rate_gap"] > 0.05:  # the twin no longer matches reality: recalibrate before trusting what-ifs
            drift = outputs["drift"]
            reason = f"twin fill-rate gap {twin_info['fill_rate_gap']:.1%} vs realised"
            drift["retrain"] = True
            drift["retrain_reasons"] = np.where(drift["retrain_reasons"] == "stable", reason, drift["retrain_reasons"] + "; " + reason)
        cfg.timings["twin"] = time.time() - twin_started
    cfg.timings["total"] = time.time() - started
    summary = summarise(cfg, outputs, selected_model, origins, keys)
    summary["served_method"] = served_method
    if twin_info:
        summary["twin"] = twin_info
    write_gold(cfg, outputs, summary)
    if cfg.mlflow:
        log_mlflow(cfg, outputs, summary)
    return summary


def dept_prices(features: pd.DataFrame, origin: pd.Timestamp, nodes: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """Revenue per unit for each store x department node (last 28 days of known sales)."""
    recent = features[(features["date"] > origin - pd.Timedelta(days=28)) & (features["date"] <= origin)]
    revenue = (recent["sales"] * recent["sell_price"].fillna(0)).groupby([recent["dept_id"], recent["store_id"]], observed=True).sum()
    units = recent.groupby(["dept_id", "store_id"], observed=True)["sales"].sum()
    price = (revenue / units.where(units > 0)).rename("unit_value").reset_index()
    price["node_id"] = make_node_ids("department", price).to_numpy()
    return price.set_index("node_id")["unit_value"].reindex(nodes.loc[nodes["level"] == "department", "node_id"]).fillna(1.0)


def margin_weights(price: pd.Series, cfg: Config) -> pd.Series:
    """Margin per unit for each department node: price x the category's assumed gross margin."""
    category = price.index.to_series().str.extract(r"dept_id=([A-Z]+)_")[0]
    return price * category.map(cfg.margin_by_category).fillna(np.mean(list(cfg.margin_by_category.values()))).to_numpy()


def evaluate_allocation(fold, dept_nodes, forecast_week, actual_week, scale, prior_scores, weights, cfg, node_log: list | None = None) -> list[dict]:
    """Allocate a constrained weekly supply with the LP and a pro-rata rule; score against actuals.

    The LP maximises expected *margin* with every node held to at least ``cfg.min_fill`` of its
    forecast. Results are in units and dollars lost, not just a count of short nodes.
    """
    rows = []
    price = weights.reindex(dept_nodes["node_id"])
    unit_value = price.to_numpy()
    margin = margin_weights(price, cfg).to_numpy()
    for week in range(forecast_week.shape[1]):
        demand = pd.DataFrame({"node_id": dept_nodes["node_id"].to_numpy(), "forecast": forecast_week[:, week], "unit_value": unit_value, "margin_value": margin})
        supply = cfg.supply_ratio * demand["forecast"].sum()
        scenarios = prob.demand_scenarios(demand["forecast"], prior_scores, scale, cfg.n_scenarios)
        lp = allocate_inventory(demand, supply, scenarios, value_column="margin_value", min_fill=cfg.min_fill)["allocated_quantity"].to_numpy()
        pro_rata = proportional_allocation(demand, supply)
        actual = actual_week[:, week]
        for policy, allocated in (("lp_scenario", lp), ("pro_rata", pro_rata)):
            sold = np.minimum(actual, allocated)
            node_fill = np.divide(sold, actual, out=np.ones(len(actual)), where=actual > 0)
            rows.append({
                "fold": fold, "week": week + 1, "policy": policy, "supply": supply,
                "units_fulfilled": sold.sum(), "units_demanded": actual.sum(),
                "fill_rate": sold.sum() / actual.sum(), "revenue_fulfilled": float((sold * unit_value).sum()),
                "units_lost": float((actual - sold).sum()), "lost_revenue": float(((actual - sold) * unit_value).sum()),
                "min_node_fill": float(node_fill.min()),
                "stockout_nodes": int((actual > allocated + 1e-9).sum()),
            })
            if node_log is not None:
                node_log.extend({"fold": fold, "week": week + 1, "policy": policy, "node_id": node, "fill_rate": float(fill)} for node, fill in zip(demand["node_id"], node_fill))
    return rows


def long_forecasts(nodes: pd.DataFrame, values: np.ndarray, dates: pd.DatetimeIndex, fold: str, method: str, actual: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "fold": fold, "method": method,
        "series_id": np.repeat(nodes["node_id"].to_numpy(), len(dates)),
        "level": np.repeat(nodes["level"].to_numpy(), len(dates)),
        "date": np.tile(dates, len(nodes)),
        "forecast": values.ravel(),
    })
    if actual is not None:
        frame["actual"] = actual.ravel()
    return frame


def build_gold(cfg, keys, nodes, rows_by_level, production, daily_scores, weekly_scores, weights, model_rows, recon_rows, coverage_rows, allocation_rows, backtest_forecasts, served_method, origins, node_fill_rows=(), daily_groups=None, segment_rows=(), protect=None) -> dict[str, pd.DataFrame]:
    dates = production["dates"]
    all_nodes = pd.concat([nodes, pd.DataFrame({"node_id": keys["series_id"], "level": "item"})], ignore_index=True)

    forecasts = []
    for method, (agg, bottom) in production["reconciled"].items():
        forecasts.append(long_forecasts(all_nodes, np.vstack([agg, bottom]), dates, "production", method))
    forecasts = pd.concat(forecasts, ignore_index=True).drop(columns="fold")
    forecasts["served"] = forecasts["method"] == served_method

    serve_agg, serve_bottom = production["reconciled"][served_method]
    intervals = []
    for level in daily_scores:
        if level == "item":
            forecast, scale, level_nodes = serve_bottom, production["scale_bottom"], all_nodes[all_nodes["level"] == "item"]
        else:
            rows = rows_by_level[level]
            forecast, scale, level_nodes = serve_agg[rows], production["scale_agg"][rows], nodes.iloc[rows]
        frame = long_forecasts(level_nodes, forecast, dates, "production", served_method).drop(columns="fold")
        scores = np.concatenate(daily_scores[level])
        score_groups = np.concatenate(daily_groups[level])
        groups = np.repeat(production["segments"][level], forecast.shape[1])
        for nominal in cfg.interval_levels:
            interval = prob.mondrian_interval(forecast.ravel(), scores, score_groups, groups, nominal, np.repeat(scale, forecast.shape[1]))
            tag = int(round(nominal * 100))
            frame[f"lower_{tag}"], frame[f"upper_{tag}"] = interval["lower"].to_numpy(), interval["upper"].to_numpy()
        intervals.append(frame)
    intervals = pd.concat(intervals, ignore_index=True)

    safety = item_safety_stock(cfg, keys, production, serve_bottom, protect)
    safety.insert(0, "series_id", keys["series_id"].to_numpy())
    for column in BOTTOM_KEYS:
        safety.insert(1, column, keys[column].to_numpy())
    safety["week_start"] = dates[0]

    dept_rows = rows_by_level["department"]
    dept_nodes = nodes.iloc[dept_rows]
    week_forecast = weekly(serve_agg[dept_rows])[:, 0]
    dept_scores = np.concatenate(weekly_scores["department"])
    price = weights.reindex(dept_nodes["node_id"])
    demand = pd.DataFrame({"node_id": dept_nodes["node_id"].to_numpy(), "forecast": week_forecast, "unit_value": price.to_numpy(), "margin_value": margin_weights(price, cfg).to_numpy()})
    parts = demand["node_id"].str.extract(r"dept_id=(?P<dept_id>[^|]+)\|store_id=(?P<store_id>.+)")
    demand = pd.concat([demand, parts], axis=1)
    supply = cfg.supply_ratio * demand["forecast"].sum()
    scenarios = prob.demand_scenarios(demand["forecast"], dept_scores, production["week_scale_dept"], cfg.n_scenarios)
    allocation = allocate_inventory(demand, supply, scenarios, value_column="margin_value", min_fill=cfg.min_fill)
    allocation["pro_rata_quantity"] = proportional_allocation(demand, supply)
    allocation["supply"] = supply
    allocation["week_start"] = dates[0]
    ss = prob.service_level_safety_stock(week_forecast, dept_scores, cfg.service_level, production["week_scale_dept"])
    allocation["safety_stock"], allocation["order_up_to"] = ss["safety_stock"].to_numpy(), ss["order_up_to"].to_numpy()

    drift = drift_table(nodes, keys, production, daily_scores["item"])
    recon = pd.DataFrame(recon_rows)
    store_wmape = recon[(recon["method"] == served_method) & (recon["level"] == "store")].set_index("fold")["wmape"]
    flag = retraining_flag(
        drift.set_index("node_id")["psi"],
        recent_wmape=float(store_wmape.get("holdout", np.nan)),
        reference_wmape=float(store_wmape[store_wmape.index.str.startswith("backtest")].mean()),
    )
    drift["retrain"], drift["retrain_reasons"] = flag["retrain"], flag["reasons"]

    return {
        "forecasts": forecasts,
        "prediction_intervals": intervals,
        "safety_stock": safety,
        "allocation": allocation,
        "allocation_backtest": pd.DataFrame(allocation_rows),
        "allocation_node_fill": pd.DataFrame(node_fill_rows),
        "model_metrics": pd.DataFrame(model_rows),
        "reconciliation_metrics": recon,
        "interval_coverage": pd.DataFrame(coverage_rows),
        "interval_coverage_segment": pd.DataFrame(segment_rows),
        "probable_stockouts": censoring_summary(probable_stockouts(production["history_bottom"]), keys, production["history_bottom"]),
        "backtest_forecasts": pd.concat(backtest_forecasts, ignore_index=True),
        "drift": drift,
    }


def drift_table(nodes, keys, production, item_scores) -> pd.DataFrame:
    """Two PSI signals per store.

    * ``demand_psi``: item-level daily sales (log1p) in the last 28 days vs the
      preceding year - a shift in the demand mix, robust to overall growth.
    * ``residual_psi``: scaled forecast errors in the holdout fold vs the backtest
      folds - whether the model's error distribution has moved.
    """
    history = production["history_bottom"]
    horizon = production["dates"].size
    stores = keys["store_id"].astype(str).to_numpy()
    reference_scores = np.concatenate([scores.reshape(len(keys), -1) for scores in item_scores[:-1]], axis=1)
    latest_scores = item_scores[-1].reshape(len(keys), -1)
    rows = []
    for store in sorted(set(stores)):
        mask = stores == store
        reference, current = np.log1p(history[mask, -392:-28]).ravel(), np.log1p(history[mask, -28:]).ravel()
        rows.append({
            "node_id": f"store:store_id={store}",
            "demand_psi": psi(reference, current),
            "residual_psi": psi(reference_scores[mask].ravel(), latest_scores[mask].ravel()),
            "reference_mean_daily": history[mask, -392:-28].sum(axis=0).mean(),
            "current_mean_daily": history[mask, -28:].sum(axis=0).mean(),
        })
    table = pd.DataFrame(rows)
    table["psi"] = table[["demand_psi", "residual_psi"]].max(axis=1)
    table["drift"] = table["psi"] >= 0.2
    return table


def summarise(cfg: Config, outputs: dict[str, pd.DataFrame], selected_model: str, origins, keys) -> dict[str, object]:
    recon = outputs["reconciliation_metrics"]
    models = outputs["model_metrics"]
    alloc = outputs["allocation_backtest"]
    coverage = outputs["interval_coverage"]
    summary = {
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(cfg).items() if key != "timings"},
        "origins": {name: str(origin.date()) for name, origin in origins.items()},
        "n_series": int(len(keys)),
        "selected_bottom_model": selected_model,
        "reconciliation_wmape": recon.pivot_table(index="level", columns=["method"], values="wmape", aggfunc="mean").round(4).to_dict(),
        "reconciliation_wmape_holdout": recon[recon["fold"] == "holdout"].pivot_table(index="level", columns="method", values="wmape").round(4).to_dict(),
        "model_wmape_item": models[models["level"] == "item"].groupby("model")["wmape"].mean().round(4).to_dict(),
        # Equal weight per level, as in M5 (6 levels here: the CA-only hierarchy has no item/dept/cat totals across stores).
        "wrmsse_holdout": recon[recon["fold"] == "holdout"].groupby("method")["wrmsse"].mean().round(4).to_dict(),
        "wrmsse_backtest": recon[recon["fold"].str.startswith("backtest")].groupby("method")["wrmsse"].mean().round(4).to_dict(),
        "interval_coverage": coverage.groupby(["level", "nominal"])["coverage"].mean().round(4).unstack().to_dict() if not coverage.empty else {},
        "allocation": alloc.groupby("policy")[["units_fulfilled", "units_demanded", "units_lost", "revenue_fulfilled", "lost_revenue"]].sum().assign(fill_rate=lambda f: f["units_fulfilled"] / f["units_demanded"]).round(4).to_dict("index") if not alloc.empty else {},
        "allocation_fairness": {policy: fairness_violations(group, cfg.min_fill, cfg.fairness_weeks)["node_id"].tolist() for policy, group in outputs["allocation_node_fill"].groupby("policy")} if not outputs["allocation_node_fill"].empty else {},
        "allocation_vs_pro_rata_ci": allocation_ci(alloc),
        "drift": outputs["drift"][["node_id", "psi", "drift", "retrain", "retrain_reasons"]].to_dict("records"),
        "timings_seconds": {key: round(value, 1) for key, value in cfg.timings.items()},
    }
    return summary


def allocation_ci(alloc: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Paired weekly difference (optimiser minus pro-rata) with a 95% t-interval."""
    if alloc.empty:
        return {}
    pivot = alloc.pivot_table(index=["fold", "week"], columns="policy", values=["revenue_fulfilled", "lost_revenue", "units_lost"])
    return {metric: dm.paired_ci(pivot[metric]["lp_scenario"].to_numpy(), pivot[metric]["pro_rata"].to_numpy()) for metric in ("revenue_fulfilled", "lost_revenue", "units_lost")}


def write_gold(cfg: Config, outputs: dict[str, pd.DataFrame], summary: dict[str, object]) -> None:
    from forecasting.export import export_dashboard_extract, export_gold

    export_gold(outputs, cfg.gold_dir)
    (cfg.gold_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    export_dashboard_extract(cfg.gold_dir, cfg.root / "data" / "dashboard")
    log.info("Wrote %d gold tables to %s", len(outputs), cfg.gold_dir)


def log_mlflow(cfg: Config, outputs: dict[str, pd.DataFrame], summary: dict[str, object]) -> None:
    import mlflow

    bt.configure_mlflow(cfg.root / "mlruns", cfg.experiment)
    params = {key: value for key, value in summary["config"].items() if key not in ("root",)}
    with mlflow.start_run(run_name=f"pipeline-{'-'.join(cfg.states)}"):
        mlflow.log_params({**params, "selected_bottom_model": summary["selected_bottom_model"], "served_method": summary["served_method"], "n_series": summary["n_series"]})
        mlflow.log_metrics({f"wrmsse.holdout.{method}": value for method, value in summary["wrmsse_holdout"].items()})
        for table, key in (("model_metrics", "model"), ("reconciliation_metrics", "method")):
            frame = outputs[table]
            for (fold, name), group in frame.groupby(["fold", key]):
                with mlflow.start_run(run_name=f"{table}:{name}:{fold}", nested=True):
                    mlflow.log_params({"fold": fold, key: name, "table": table})
                    mlflow.log_metrics({f"{metric}.{row.level}": getattr(row, metric) for row in group.itertuples() for metric in ("wmape", "mape", "pinball_loss", "bias", "wrmsse") if pd.notna(getattr(row, metric, np.nan))})
        mean_recon = outputs["reconciliation_metrics"].groupby(["method", "level"])["wmape"].mean()
        mlflow.log_metrics({f"wmape.{method}.{level}": value for (method, level), value in mean_recon.items()})
        for policy, values in summary["allocation"].items():
            mlflow.log_metric(f"allocation.{policy}.fill_rate", values["fill_rate"])
        for table in ("model_metrics", "reconciliation_metrics", "interval_coverage", "allocation_backtest", "drift"):
            mlflow.log_artifact(str(cfg.gold_dir / f"{table}.csv"))
        mlflow.log_artifact(str(cfg.gold_dir / "run_summary.json"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the hierarchical forecasting pipeline end to end.")
    parser.add_argument("--states", nargs="+", default=["CA"], help="M5 states to model (CA, TX, WI)")
    parser.add_argument("--history-days", type=int, default=548)
    parser.add_argument("--folds", type=int, default=3, help="Rolling-origin backtest folds before the holdout")
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--max-items", type=int, default=None, help="Subsample items for a quick run")
    parser.add_argument("--temporal", action="store_true", help="Temporally reconcile aggregate base forecasts (experimental, off by default)")
    parser.add_argument("--no-twin", action="store_true", help="Skip the inventory twin phase")
    parser.add_argument("--no-prophet", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config(
        root=Path(__file__).resolve().parents[2], states=tuple(args.states), history_days=args.history_days,
        n_backtest_folds=args.folds, n_estimators=args.estimators, max_items=args.max_items,
        prophet=not args.no_prophet, mlflow=not args.no_mlflow, temporal=args.temporal, twin=not args.no_twin,
    )
    summary = run(cfg)
    print(json.dumps({key: summary[key] for key in ("selected_bottom_model", "served_method", "wrmsse_holdout", "reconciliation_wmape_holdout", "interval_coverage", "allocation")}, indent=2, default=str))


if __name__ == "__main__":
    main()
