"""Run the inventory twin on pipeline folds: validation, timeline, cost frontier, stress tests, exceptions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting import twin
from forecasting.twin import TwinConfig

ADDITIVE = ("units_demanded", "units_sold", "units_lost", "lost_sales_value", "sales_value", "inventory_value", "holding_cost")
REPORT = ("fill_rate", "in_stock_pct", "lost_sales_value", "inventory_value", "units_lost", "units_demanded")
POLICY_LABELS = {"forecast_reorder": "forecast_reorder", "last_week_reorder": "last_week_reorder"}
SERVICE_GRID = twin.SERVICE_GRID
SCENARIOS = {  # day indices count from the start of the 28-day window
    "event_spike": {"label": "Holiday-style spike: FOODS demand +50% for a week", "demand_scale": 1.5, "category": "FOODS", "days": (7, 14)},
    "supplier_delay": {"label": "Supplier lead time 7 -> 14 days", "delay": 7},
    "dc_cut": {"label": "Warehouse receives 30% less stock for two weeks", "dc_factor": 0.7, "dc_days": (7, 21)},
}


@dataclass
class FoldInputs:
    keys: pd.DataFrame  # series_id, item_id, dept_id, cat_id, store_id (rows align with the arrays)
    history: np.ndarray  # (n, days) actual sales before the window
    forecast: np.ndarray  # (n, horizon)
    actual: np.ndarray | None
    scale: np.ndarray  # (n,) the scale the conformal scores were divided by
    safety: np.ndarray  # (n,)
    review: np.ndarray  # (n,) days between store orders
    price: np.ndarray  # (n,)
    dates: pd.DatetimeIndex
    past_actual: np.ndarray | None = None  # (n, 28 x folds) what sold in earlier windows (for the demand bootstrap)
    past_forecast: np.ndarray | None = None  # (n, 28 x folds) what was forecast for them


def run_sim(inp: FoldInputs, rows: np.ndarray, tcfg: TwinConfig, policy: str, reps: int, seed: int, score_paths: np.ndarray | None,
            shock_spec: dict | None = None, replay: bool = False, groups: np.ndarray | None = None, safety: np.ndarray | None = None,
            risk_days: int | None = None, chunk: int = 25, per_series: bool = True) -> dict:
    """Simulate one store (or subset) of a fold: realised demand when ``replay``, else sampled from past errors."""
    safety = inp.safety if safety is None else safety
    shock = twin.make_shock(shock_spec, inp.keys["cat_id"].astype(str).to_numpy()[rows], tcfg.warmup_days)
    return twin.run_store(
        inp.history[rows], inp.forecast[rows], None if inp.actual is None else inp.actual[rows], inp.scale[rows], safety[rows], inp.review[rows],
        inp.price[rows], tcfg, policy, reps, seed, None if score_paths is None else score_paths[rows], shock, replay, groups, risk_days, chunk, per_series,
        None if inp.past_actual is None else inp.past_actual[rows], None if inp.past_forecast is None else inp.past_forecast[rows],
    )


def combine(parts: list[dict]) -> dict[str, np.ndarray]:
    """Network KPIs from store runs: add units and dollars, recompute ratios."""
    total = {name: sum(part["kpis"][name] for part in parts) for name in ADDITIVE}
    weights = np.array([part["n"] for part in parts], dtype=float)
    demand = np.array([part["kpis"]["units_demanded"].mean() for part in parts])
    total["fill_rate"] = total["units_sold"] / np.maximum(total["units_demanded"], 1e-9)
    total["in_stock_pct"] = sum(w * part["kpis"]["in_stock_pct"] for w, part in zip(weights, parts)) / weights.sum()
    total["weeks_of_supply"] = sum(w * part["kpis"]["weeks_of_supply"] for w, part in zip(demand, parts)) / max(demand.sum(), 1e-9)
    return total


def _band(values: np.ndarray, band: float = 0.90) -> tuple[float, float, float]:
    low, high = (1 - band) / 2 * 100, (1 + band) / 2 * 100
    return float(np.mean(values)), float(np.percentile(values, low)), float(np.percentile(values, high))


def stores_of(inp: FoldInputs) -> dict[str, np.ndarray]:
    store = inp.keys["store_id"].astype(str).to_numpy()
    return {name: np.flatnonzero(store == name) for name in sorted(set(store))}


def validation_rows(fold: str, inp: FoldInputs, tcfg: TwinConfig, policies, reps: int, score_paths: np.ndarray, seed: int) -> list[dict]:
    """Predicted KPI band (sampled demand) vs the realised KPI (replayed actual demand)."""
    rows = []
    for policy in policies:
        predicted, realised = {}, {}
        for index, (store, idx) in enumerate(stores_of(inp).items()):
            predicted[store] = run_sim(inp, idx, tcfg, policy, reps, seed + index, score_paths)
            realised[store] = run_sim(inp, idx, tcfg, policy, 1, seed, None, replay=True)
        scopes = {"all": (combine(list(predicted.values())), combine(list(realised.values())))}
        scopes.update({store: (predicted[store]["kpis"], realised[store]["kpis"]) for store in predicted})
        for scope, (pred, real) in scopes.items():
            for metric in REPORT:
                mean, lower, upper = _band(pred[metric])
                actual = float(real[metric][0])
                rows.append({
                    "fold": fold, "store_id": scope, "policy": policy, "metric": metric, "predicted": mean, "lower": lower, "upper": upper,
                    "realised": actual, "in_band": bool(lower <= actual <= upper),
                    "divergence": (mean - actual) / max(abs(actual), 1e-9),
                })
    return rows


def timeline_frame(inp: FoldInputs, tcfg: TwinConfig, policy: str, seed: int) -> pd.DataFrame:
    """Daily on-hand, demand and lost units per store x department for a replayed window."""
    frames = []
    for store, idx in stores_of(inp).items():
        labels = inp.keys["dept_id"].astype(str).to_numpy()[idx]
        timeline = run_sim(inp, idx, tcfg, policy, 1, seed, None, replay=True, groups=labels)["timeline"]
        for column, dept in enumerate(timeline["group"]):
            frames.append(pd.DataFrame({
                "date": inp.dates, "store_id": store, "dept_id": dept, "policy": policy,
                "on_hand": timeline["on_hand"][:, column], "demand": timeline["demand"][:, column], "lost": timeline["lost"][:, column],
            }))
    return pd.concat(frames, ignore_index=True)


def frontier_frame(inp: FoldInputs, tcfg: TwinConfig, safety_for_service, current_safety: np.ndarray, margins: dict, holding: dict, reps: int, score_paths: np.ndarray, seed: int) -> pd.DataFrame:
    """Cost of each service level per category: lost margin + holding cost; the cheapest is recommended."""
    categories = inp.keys["cat_id"].astype(str).to_numpy()
    options = [(float(level), safety_for_service(level)) for level in SERVICE_GRID] + [(float("nan"), current_safety)]
    rows = []
    for category in sorted(set(categories)):
        for level, safety in options:
            parts = []
            for index, (store, idx) in enumerate(stores_of(inp).items()):
                subset = idx[categories[idx] == category]
                parts.append(run_sim(inp, subset, tcfg, "forecast_reorder", reps, seed + index, score_paths, safety=safety))
            kpis = combine(parts)
            lost_margin = kpis["lost_sales_value"] * margins.get(category, 0.3)
            holding_cost = kpis["inventory_value"] * holding.get(category, 0.005) * 28 / 7  # category-specific: perishables cost more to hold
            mean, lower, upper = _band(lost_margin + holding_cost)
            rows.append({
                "category": category, "service_level": level, "label": "segment targets (current)" if np.isnan(level) else f"{level:.0%}",
                "fill_rate": float(kpis["fill_rate"].mean()), "in_stock_pct": float(kpis["in_stock_pct"].mean()),
                "inventory_value": float(kpis["inventory_value"].mean()), "lost_sales_value": float(kpis["lost_sales_value"].mean()),
                "lost_margin": float(lost_margin.mean()), "holding_cost": float(holding_cost.mean()),
                "total_cost": mean, "total_cost_lower": lower, "total_cost_upper": upper,
            })
    frame = pd.DataFrame(rows)
    grid = frame[frame["service_level"].notna()]
    best = grid.loc[grid.groupby("category")["total_cost"].idxmin()].index
    frame["recommended"] = frame.index.isin(best)
    return frame


def stress_frame(inp: FoldInputs, tcfg: TwinConfig, policies, reps: int, score_paths: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    for policy in policies:
        baseline = combine([run_sim(inp, idx, tcfg, policy, reps, seed + i, score_paths) for i, idx in enumerate(stores_of(inp).values())])
        for name, spec in SCENARIOS.items():
            shocked = combine([run_sim(inp, idx, tcfg, policy, reps, seed + i, score_paths, shock_spec=spec) for i, idx in enumerate(stores_of(inp).values())])
            for metric in REPORT:
                b_mean, b_low, b_high = _band(baseline[metric])
                s_mean, s_low, s_high = _band(shocked[metric])
                rows.append({
                    "scenario": name, "label": spec["label"], "policy": policy, "metric": metric,
                    "baseline": b_mean, "baseline_lower": b_low, "baseline_upper": b_high,
                    "scenario_value": s_mean, "scenario_lower": s_low, "scenario_upper": s_high, "delta": s_mean - b_mean,
                })
    return pd.DataFrame(rows)


def exceptions_frame(inp: FoldInputs, tcfg: TwinConfig, policy: str, reps: int, score_paths: np.ndarray, seed: int, days: int = 7, top: int = 500) -> pd.DataFrame:
    """Products most likely to stock out in the next ``days`` days under ``policy``, ranked by lost dollars."""
    frames = []
    for index, (store, idx) in enumerate(stores_of(inp).items()):
        risk = run_sim(inp, idx, tcfg, policy, reps, seed + index, score_paths, risk_days=days)["risk"]
        frame = inp.keys.iloc[idx][["series_id", "item_id", "dept_id", "cat_id", "store_id"]].astype(str).copy()
        frame["expected_lost_units"] = risk["expected_lost"]
        frame["stockout_probability"] = risk["stockout_prob"]
        frame["expected_lost_value"] = risk["expected_lost"] * inp.price[idx]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["policy"] = policy
    table["window_days"] = days
    return table.sort_values("expected_lost_value", ascending=False).head(top).reset_index(drop=True)
