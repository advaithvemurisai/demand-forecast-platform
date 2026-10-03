"""Run the inventory twin on pipeline folds: validation, timeline, cost frontier, policy curve, stress tests, exceptions.

Every run simulates all stores together, so they draw on one DC (stock held per product) and share each sampled demand
week; per-store numbers are read off the network run rather than simulated separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting import twin
from forecasting.twin import TwinConfig

REPORT = ("fill_rate", "in_stock_pct", "lost_sales_value", "lost_margin", "inventory_value", "holding_cost", "units_lost", "units_demanded")
POLICY_LABELS = {"forecast_reorder": "forecast_reorder", "last_week_reorder": "last_week_reorder"}
SERVICE_GRID = twin.SERVICE_GRID
SAFETY_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SCENARIOS = {  # day indices count from the start of the 28-day window
    "event_spike": {"label": "Holiday-style spike: FOODS demand +50% for a week", "demand_scale": 1.5, "category": "FOODS", "days": (7, 14)},
    "late_shipment": {"label": "One supplier delivery arrives 7 days late", "delay": 7, "delay_days": (0, 7)},
    "supplier_delay": {"label": "Supplier lead time 7 → 14 days; the planner adjusts after a week", "delay": 7, "replan_after": 7},
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
    price: np.ndarray  # (n,) shelf price
    dates: pd.DatetimeIndex
    past_actual: np.ndarray | None = None  # (n, 28 x folds) what sold in earlier windows (for the demand bootstrap)
    past_forecast: np.ndarray | None = None  # (n, 28 x folds) what was forecast for them
    cost: np.ndarray | None = None  # (n,) unit cost: price x (1 - assumed category margin)
    holding: np.ndarray | None = None  # (n,) weekly holding cost as a share of cost

    def column(self, name: str) -> np.ndarray:
        return self.keys[name].astype(str).to_numpy()


def run_sim(inp: FoldInputs, rows: np.ndarray | None, tcfg: TwinConfig, policy: str, reps: int, seed: int, score_paths: np.ndarray | None,
            shock_spec: dict | None = None, replay: bool = False, groups: np.ndarray | None = None, safety: np.ndarray | None = None,
            risk_days: int | None = None, chunk: int = 25, per_series: bool = True, by_store: bool = False) -> dict:
    """Simulate ``rows`` (default: the whole network) of a fold: realised demand when ``replay``, else sampled."""
    rows = np.arange(len(inp.forecast)) if rows is None else rows
    safety = inp.safety if safety is None else safety
    pick = lambda array: None if array is None else array[rows]
    shock = twin.make_shock(shock_spec, inp.column("cat_id")[rows], tcfg.warmup_days)
    return twin.run_store(
        inp.history[rows], inp.forecast[rows], pick(inp.actual), inp.scale[rows], safety[rows], inp.review[rows],
        inp.price[rows], tcfg, policy, reps, seed, pick(score_paths), shock, replay, groups, risk_days, chunk, per_series,
        pick(inp.past_actual), pick(inp.past_forecast), products=inp.column("item_id")[rows], cost=pick(inp.cost), holding_rate=pick(inp.holding),
        kpi_groups=inp.column("store_id")[rows] if by_store else None,
    )


def _band(values: np.ndarray, band: float = 0.90) -> tuple[float, float, float]:
    low, high = (1 - band) / 2 * 100, (1 + band) / 2 * 100
    return float(np.mean(values)), float(np.percentile(values, low)), float(np.percentile(values, high))


def scopes(result: dict) -> dict[str, dict[str, np.ndarray]]:
    return {"all": result["kpis"], **result.get("kpis_by", {})}


def validation_rows(fold: str, inp: FoldInputs, tcfg: TwinConfig, policies, reps: int, score_paths: np.ndarray, seed: int) -> list[dict]:
    """Predicted KPI band (sampled demand) vs the realised KPI (replayed actual demand), for the network and each store."""
    rows = []
    for policy in policies:
        predicted = scopes(run_sim(inp, None, tcfg, policy, reps, seed, score_paths, by_store=True))
        realised = scopes(run_sim(inp, None, tcfg, policy, 1, seed, None, replay=True, by_store=True))
        for scope, pred in predicted.items():
            for metric in REPORT:
                mean, lower, upper = _band(pred[metric])
                actual = float(realised[scope][metric][0])
                rows.append({
                    "fold": fold, "store_id": scope, "policy": policy, "metric": metric, "predicted": mean, "lower": lower, "upper": upper,
                    "realised": actual, "in_band": bool(lower <= actual <= upper),
                    "divergence": (mean - actual) / max(abs(actual), 1e-9),
                })
    return rows


def timeline_frame(inp: FoldInputs, tcfg: TwinConfig, policy: str, seed: int) -> pd.DataFrame:
    """Daily on-hand, demand and lost units per store x department for a replayed window."""
    labels = np.char.add(np.char.add(inp.column("store_id"), "|"), inp.column("dept_id"))
    timeline = run_sim(inp, None, tcfg, policy, 1, seed, None, replay=True, groups=labels)["timeline"]
    frames = []
    for column, label in enumerate(timeline["group"]):
        store, dept = str(label).split("|")
        frames.append(pd.DataFrame({
            "date": inp.dates, "store_id": store, "dept_id": dept, "policy": policy,
            "on_hand": timeline["on_hand"][:, column], "demand": timeline["demand"][:, column], "lost": timeline["lost"][:, column],
        }))
    return pd.concat(frames, ignore_index=True)


def frontier_frame(inp: FoldInputs, tcfg: TwinConfig, safety_for_service, current_safety: np.ndarray, reps: int, score_paths: np.ndarray, seed: int) -> pd.DataFrame:
    """Cost of each service level per category: lost margin + holding cost (at cost); the cheapest is recommended.

    Every option sees the same simulated futures, so the saving against today's segment targets is measured per future
    and reported with a 90% band: a saving whose band includes zero is not a clear win.
    """
    categories = inp.column("cat_id")
    options = [(float(level), safety_for_service(level)) for level in SERVICE_GRID] + [(float("nan"), current_safety)]
    rows = []
    for category in sorted(set(categories)):
        subset = np.flatnonzero(categories == category)
        costs = {}
        for level, safety in options:
            kpis = run_sim(inp, subset, tcfg, "forecast_reorder", reps, seed, score_paths, safety=safety)["kpis"]
            cost = kpis["lost_margin"] + kpis["holding_cost"]
            costs[level if np.isfinite(level) else "current"] = cost
            mean, lower, upper = _band(cost)
            rows.append({
                "category": category, "service_level": level, "label": "segment targets (current)" if np.isnan(level) else f"{level:.1%}".replace(".0%", "%"),
                "fill_rate": float(kpis["fill_rate"].mean()), "in_stock_pct": float(kpis["in_stock_pct"].mean()),
                "inventory_value": float(kpis["inventory_value"].mean()), "lost_sales_value": float(kpis["lost_sales_value"].mean()),
                "lost_margin": float(kpis["lost_margin"].mean()), "holding_cost": float(kpis["holding_cost"].mean()),
                "total_cost": mean, "total_cost_lower": lower, "total_cost_upper": upper,
            })
        for row in rows:
            if row["category"] == category:
                key = row["service_level"] if np.isfinite(row["service_level"]) else "current"
                saving, low, high = _band(costs["current"] - costs[key])
                row.update(saving_vs_current=saving, saving_lower=low, saving_upper=high)
    frame = pd.DataFrame(rows)
    grid = frame[frame["service_level"].notna()]
    best = grid.loc[grid.groupby("category")["total_cost"].idxmin()]
    frame["recommended"] = frame.index.isin(best.index)
    frame["at_grid_edge"] = frame["recommended"] & frame["service_level"].isin([min(SERVICE_GRID), max(SERVICE_GRID)])
    frame["clear_saving"] = frame["saving_lower"] > 0
    return frame


def policy_curve_frame(inp: FoldInputs, tcfg: TwinConfig, seed: int, multipliers=SAFETY_MULTIPLIERS) -> pd.DataFrame:
    """Fill rate against inventory for both policies, replayed on what actually sold, sweeping the safety stock.

    Comparing the policies at one safety setting mixes up a better forecast with simply holding more stock; along
    these curves they can be compared at equal inventory.
    """
    rows = []
    for policy in POLICY_LABELS:
        for multiplier in multipliers:
            kpis = run_sim(inp, None, tcfg, policy, 1, seed, None, replay=True, safety=inp.safety * multiplier)["kpis"]
            rows.append({"policy": policy, "safety_multiplier": multiplier, **{metric: float(kpis[metric][0]) for metric in REPORT}})
    return pd.DataFrame(rows)


def equal_inventory(curve: pd.DataFrame, metric: str = "fill_rate") -> dict[str, float]:
    """Each policy's ``metric`` at the forecast policy's standard (1x safety) inventory, read off the curves."""
    forecast = curve[curve["policy"] == "forecast_reorder"].sort_values("inventory_value")
    naive = curve[curve["policy"] == "last_week_reorder"].sort_values("inventory_value")
    standard = forecast[forecast["safety_multiplier"] == 1.0].iloc[0]
    inside = naive["inventory_value"].min() <= standard["inventory_value"] <= naive["inventory_value"].max()
    return {
        "inventory_value": float(standard["inventory_value"]), "forecast": float(standard[metric]),
        "current_practice": float(np.interp(standard["inventory_value"], naive["inventory_value"], naive[metric])) if inside else float("nan"),
        "current_practice_inventory_for_same": float(np.interp(standard[metric], naive[metric], naive["inventory_value"])) if naive[metric].max() >= standard[metric] else float("nan"),
    }


def stress_frame(inp: FoldInputs, tcfg: TwinConfig, policies, reps: int, score_paths: np.ndarray, seed: int) -> pd.DataFrame:
    """Each scenario vs business as usual on the same futures; the warehouse cut is also run under each rationing rule."""
    rows = []

    def compare(name: str, spec: dict, policy: str, config: TwinConfig) -> None:
        baseline = run_sim(inp, None, config, policy, reps, seed, score_paths)["kpis"]
        shocked = run_sim(inp, None, config, policy, reps, seed, score_paths, shock_spec={k: v for k, v in spec.items() if k != "label"})["kpis"]
        for metric in REPORT:
            b_mean, b_low, b_high = _band(baseline[metric])
            s_mean, s_low, s_high = _band(shocked[metric])
            d_mean, d_low, d_high = _band(shocked[metric] - baseline[metric])
            rows.append({
                "scenario": name, "label": spec["label"], "policy": policy, "rationing": config.rationing, "metric": metric,
                "baseline": b_mean, "baseline_lower": b_low, "baseline_upper": b_high,
                "scenario_value": s_mean, "scenario_lower": s_low, "scenario_upper": s_high,
                "delta": d_mean, "delta_lower": d_low, "delta_upper": d_high,
            })

    for policy in policies:
        for name, spec in SCENARIOS.items():
            compare(name, spec, policy, tcfg)
    for mode in twin.RATIONING:
        if mode != tcfg.rationing:
            compare("dc_cut", SCENARIOS["dc_cut"], "forecast_reorder", TwinConfig(**{**tcfg.__dict__, "rationing": mode}))
    return pd.DataFrame(rows)


def exceptions_frame(inp: FoldInputs, tcfg: TwinConfig, policy: str, reps: int, score_paths: np.ndarray, seed: int, days: int = 7, top: int = 500) -> pd.DataFrame:
    """Products most likely to stock out in the next ``days`` days under ``policy``, ranked by lost dollars."""
    risk = run_sim(inp, None, tcfg, policy, reps, seed, score_paths, risk_days=days)["risk"]
    table = inp.keys[["series_id", "item_id", "dept_id", "cat_id", "store_id"]].astype(str).copy()
    table["expected_lost_units"] = risk["expected_lost"]
    table["stockout_probability"] = risk["stockout_prob"]
    table["expected_lost_value"] = risk["expected_lost"] * inp.price
    table["policy"] = policy
    table["window_days"] = days
    return table.sort_values("expected_lost_value", ascending=False).head(top).reset_index(drop=True)
