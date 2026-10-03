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

REPORT = ("fill_rate", "in_stock_pct", "lost_sales_value", "lost_margin", "inventory_value", "holding_cost", "units_lost", "units_demanded", "sales_value")
NETWORK_REPORT = REPORT + ("dc_inventory_value", "dc_on_order_value", "dc_fill_rate", "dc_holding_cost", "response_cost")
POLICY_LABELS = {"forecast_reorder": "forecast_reorder", "last_week_reorder": "last_week_reorder"}
SERVICE_GRID = twin.SERVICE_GRID
SAFETY_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SCENARIOS = {  # day indices count from the start of the 28-day window
    "event_spike": {"label": "Holiday-style spike: FOODS demand +50% for a week", "demand_scale": 1.5, "category": "FOODS", "days": (7, 14)},
    "late_shipment": {"label": "One supplier delivery arrives 7 days late", "delay": 7, "delay_days": (0, 7)},
    "supplier_delay": {"label": "Supplier lead time 7 → 14 days; the planner adjusts after a week", "delay": 7, "replan_after": 7},
    "dc_cut": {"label": "Warehouse receives 30% less stock for two weeks", "dc_factor": 0.7, "dc_days": (7, 21)},
}
# What a planner can do about each scenario, and what it costs. Premiums are assumptions (share of unit cost).
RESPONSES = {
    "supplier_delay": {
        "replan_now": {"label": "Plan on the 14-day lead time at once", "detail": "no week of waiting before the warehouse orders for the longer lead time", "replan_after": 0},
        "expedite": {"label": "Expedite only what bridges the gap", "detail": "the units the warehouse would run short of before a late order lands; +20% of unit cost on them", "expedite_share": 1.0, "premium": 0.20},
        "prebuild": {"label": "Pre-build a week of warehouse stock", "detail": "needs two weeks' notice; costs holding on the extra stock", "prebuild_days": 7},
    },
    "late_shipment": {
        "expedite": {"label": "Expedite only what bridges the gap", "detail": "+20% of unit cost on the expedited units", "expedite_share": 1.0, "premium": 0.20},
        "prebuild": {"label": "Pre-build a week of warehouse stock", "detail": "two weeks' notice, run down once the delivery lands", "prebuild_days": 7, "prebuild_until": 14},
    },
    "dc_cut": {
        "backup": {"label": "A second supplier makes up half the shortfall", "detail": "+10% of unit cost on the second supplier's units", "backup_share": 0.5, "premium": 0.10},
        "prebuild": {"label": "Pre-build a week of warehouse stock", "detail": "two weeks' notice, run down once the cut ends", "prebuild_days": 7, "prebuild_until": 21},
    },
    "event_spike": {
        "planned": {"label": "Put the spike in the forecast", "detail": "stores and warehouse order for it; costs holding on the extra stock", "planned": True},
    },
}
# How much of the simulated lost sales a shortage really costs the store. The simulator overstates lost sales against
# replayed actuals (``bias``), and shoppers facing an empty shelf often substitute or come back: Gruen & Corsten (2002)
# found 31% buy at another store and 9% don't buy, so about 40% of out-of-stock encounters are lost to the retailer.
SHOPPER_LOST_SHARE = 0.40


def lost_sale_assumptions(bias: float) -> dict[str, dict]:
    """Scales applied to simulated lost margin when pricing a service target; ``bias`` = simulated / realised lost sales."""
    bias = bias if np.isfinite(bias) and bias > 0 else 1.0
    return {
        "every_unit": {"label": "Every unmet unit is a lost sale (as simulated)", "scale": 1.0},
        "bias_corrected": {"label": f"Corrected for the simulator overstating lost sales {bias:.1f}×", "scale": 1.0 / bias},
        "shopper_response": {"label": f"Corrected, and only {SHOPPER_LOST_SHARE:.0%} of stockouts lose the sale (shoppers substitute or wait)", "scale": SHOPPER_LOST_SHARE / bias},
    }


DEFAULT_ASSUMPTION = "bias_corrected"


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
            risk_days: int | None = None, chunk: int = 25, per_series: bool = True, by_store: bool = False, kpi_groups: np.ndarray | None = None) -> dict:
    """Simulate ``rows`` (default: the whole network) of a fold: realised demand when ``replay``, else sampled.

    KPIs also come back per store with ``by_store``, or per label of ``kpi_groups`` (one per row of the fold)."""
    rows = np.arange(len(inp.forecast)) if rows is None else rows
    safety = inp.safety if safety is None else safety
    pick = lambda array: None if array is None else array[rows]
    shock = twin.make_shock(shock_spec, inp.column("cat_id")[rows], tcfg.warmup_days)
    return twin.run_store(
        inp.history[rows], inp.forecast[rows], pick(inp.actual), inp.scale[rows], safety[rows], inp.review[rows],
        inp.price[rows], tcfg, policy, reps, seed, pick(score_paths), shock, replay, groups, risk_days, chunk, per_series,
        pick(inp.past_actual), pick(inp.past_forecast), products=inp.column("item_id")[rows], cost=pick(inp.cost), holding_rate=pick(inp.holding),
        kpi_groups=inp.column("store_id")[rows] if by_store else (None if kpi_groups is None else np.asarray(kpi_groups)[rows]),
    )


def _band(values: np.ndarray, band: float = 0.90) -> tuple[float, float, float]:
    low, high = (1 - band) / 2 * 100, (1 + band) / 2 * 100
    return float(np.mean(values)), float(np.percentile(values, low)), float(np.percentile(values, high))


def scopes(result: dict) -> dict[str, dict[str, np.ndarray]]:
    return {"all": result["kpis"], **result.get("kpis_by", {})}


def validation_rows(fold: str, inp: FoldInputs, tcfg: TwinConfig, policies, reps: int, score_paths: np.ndarray, seed: int,
                    replay_seed: int | None = None) -> list[dict]:
    """Predicted KPI band (sampled demand) vs the realised KPI (replayed actual demand), for the network and each store.

    ``replay_seed`` fixes the supplier lead-time draws of the realised replay, so it can match the other replays of a window.
    """
    rows = []
    for policy in policies:
        predicted = scopes(run_sim(inp, None, tcfg, policy, reps, seed, score_paths, by_store=True))
        realised = scopes(run_sim(inp, None, tcfg, policy, 1, seed if replay_seed is None else replay_seed, None, replay=True, by_store=True))
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


def frontier_frame(inp: FoldInputs, tcfg: TwinConfig, safety_for_service, current_safety: np.ndarray, reps: int, score_paths: np.ndarray, seed: int,
                   assumptions: dict[str, dict] | None = None, groups: np.ndarray | None = None) -> pd.DataFrame:
    """Cost of each service level per category: lost margin + holding cost (at cost), under each lost-sale assumption.

    The lost margin is scaled by each assumption (see ``lost_sale_assumptions``), since pricing every unmet unit as a
    lost sale biases the curve toward more stock. Every option sees the same simulated futures, so the saving against
    today's segment targets is measured per future with a 90% band: a saving whose band includes zero is not a clear
    win. ``recommended`` is the cheapest target; ``within_budget`` the cheapest that holds no more stock than today.
    Rows are grouped by category unless ``groups`` (one label per row, e.g. product speed class) is given; the label
    lands in the ``category`` column either way.
    """
    assumptions = assumptions or {"every_unit": {"label": "Every unmet unit is a lost sale", "scale": 1.0}}
    categories = inp.column("cat_id") if groups is None else np.asarray(groups).astype(str)
    options = [(float(level), safety_for_service(level)) for level in SERVICE_GRID] + [(float("nan"), current_safety)]
    rows = []
    for category in sorted(set(categories)):
        subset = np.flatnonzero(categories == category)
        runs = {}
        for level, safety in options:
            runs[level if np.isfinite(level) else "current"] = (level, run_sim(inp, subset, tcfg, "forecast_reorder", reps, seed, score_paths, safety=safety)["kpis"])
        for name, assumption in assumptions.items():
            price = lambda kpis: kpis["lost_margin"] * assumption["scale"] + kpis["holding_cost"]
            current = price(runs["current"][1])
            for key, (level, kpis) in runs.items():
                cost = price(kpis)
                mean, lower, upper = _band(cost)
                saving, low, high = _band(current - cost)
                rows.append({
                    "assumption": name, "assumption_label": assumption["label"], "lost_scale": assumption["scale"],
                    "category": category, "service_level": level, "label": "segment targets (current)" if np.isnan(level) else f"{level:.1%}".replace(".0%", "%"),
                    "fill_rate": float(kpis["fill_rate"].mean()), "in_stock_pct": float(kpis["in_stock_pct"].mean()),
                    "inventory_value": float(kpis["inventory_value"].mean()), "lost_sales_value": float(kpis["lost_sales_value"].mean()),
                    "lost_margin": float(kpis["lost_margin"].mean() * assumption["scale"]), "holding_cost": float(kpis["holding_cost"].mean()),
                    "total_cost": mean, "total_cost_lower": lower, "total_cost_upper": upper,
                    "saving_vs_current": saving, "saving_lower": low, "saving_upper": high,
                    "inventory_change": float(kpis["inventory_value"].mean() - runs["current"][1]["inventory_value"].mean()),
                })
    frame = pd.DataFrame(rows)
    grid = frame[frame["service_level"].notna()]
    best = grid.loc[grid.groupby(["assumption", "category"])["total_cost"].idxmin()]
    frame["recommended"] = frame.index.isin(best.index)
    frame["at_grid_edge"] = frame["recommended"] & frame["service_level"].isin([min(SERVICE_GRID), max(SERVICE_GRID)])
    frame["clear_saving"] = frame["saving_lower"] > 0
    affordable = grid[grid["inventory_change"] <= 0]
    budget = affordable.loc[affordable.groupby(["assumption", "category"])["total_cost"].idxmin()] if len(affordable) else affordable
    frame["within_budget"] = frame.index.isin(budget.index)
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
        for metric in NETWORK_REPORT:
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


def response_frame(inp: FoldInputs, tcfg: TwinConfig, reps: int, score_paths: np.ndarray, seed: int, lost_scale: float = 1.0) -> pd.DataFrame:
    """Each scenario without and with each planner response, against business as usual, on the same futures.

    ``net_benefit`` = lost margin recovered (scaled by ``lost_scale``, the default lost-sale assumption) minus the
    response's cost: expedite / second-supplier premiums plus the extra holding cost at the stores and the DC.
    """
    rows = []
    baseline = run_sim(inp, None, tcfg, "forecast_reorder", reps, seed, score_paths)["kpis"]
    cost = lambda k: k["holding_cost"] + k["dc_holding_cost"] + k["response_cost"]
    for name, responses in RESPONSES.items():
        spec = {k: v for k, v in SCENARIOS[name].items() if k != "label"}
        shocked = run_sim(inp, None, tcfg, "forecast_reorder", reps, seed, score_paths, shock_spec=spec)["kpis"]
        for key, response in {"none": {"label": "No response", "detail": ""}, **responses}.items():
            kpis = shocked if key == "none" else run_sim(
                inp, None, tcfg, "forecast_reorder", reps, seed, score_paths,
                shock_spec={**spec, **{k: v for k, v in response.items() if k not in ("label", "detail")}})["kpis"]
            added_lost = kpis["lost_sales_value"] - baseline["lost_sales_value"]
            recovered = (shocked["lost_margin"] - kpis["lost_margin"]) * lost_scale
            extra_cost = cost(kpis) - cost(shocked)
            row = {"scenario": name, "scenario_label": SCENARIOS[name]["label"], "response": key, "label": response["label"], "detail": response["detail"]}
            for metric, values in {"added_lost_sales": added_lost, "response_cost": extra_cost, "net_benefit": recovered - extra_cost}.items():
                row[metric], row[f"{metric}_lower"], row[f"{metric}_upper"] = _band(values)
            row["dc_fill_rate"] = float(kpis["dc_fill_rate"].mean())
            row["premium"] = float(kpis["response_cost"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def health_frame(inp: FoldInputs, tcfg: TwinConfig, seed: int, velocity: np.ndarray | None = None) -> pd.DataFrame:
    """Planner KPIs from the holdout replay: weeks of supply, turns, GMROI, lost sales as a share of demand, warehouse stock.

    Store stock comes from the per-row replay; DC stock is held per product, so it is rolled up to categories only.
    Rates are annualised from the replayed window.
    """
    def run(groups):
        return run_sim(inp, None, tcfg, "forecast_reorder", 1, seed, None, replay=True, kpi_groups=groups)

    by_category = run(inp.column("cat_id"))
    dc = by_category["dc_by_product"]
    product_category = dict(zip(inp.column("item_id"), inp.column("cat_id")))
    dc_value = pd.Series(dc["value"], index=[product_category[str(p)] for p in dc["products"]]).groupby(level=0).sum()
    window_days = inp.forecast.shape[1]
    tables = [("category", "all", by_category["kpis"], float(dc_value.sum()))]
    tables += [("category", label, k, float(dc_value.get(label, 0.0))) for label, k in by_category["kpis_by"].items()]
    if velocity is not None:
        tables += [("speed", label, k, float("nan")) for label, k in run(np.asarray(velocity).astype(str))["kpis_by"].items()]
    rows = []
    for kind, label, k, dc_stock in tables:
        sales, margin, lost, store_stock = (float(k[m][0]) for m in ("sales_value", "margin_value", "lost_sales_value", "inventory_value"))
        cogs_week = (sales - margin) / window_days * 7
        stock = store_stock + (dc_stock if np.isfinite(dc_stock) else 0.0)
        rows.append({
            "group_type": kind, "group": label, "fill_rate": float(k["fill_rate"][0]), "in_stock_pct": float(k["in_stock_pct"][0]),
            "sales_value": sales, "lost_sales_value": lost, "lost_share": lost / max(sales + lost, 1e-9),
            "store_inventory_value": store_stock, "dc_inventory_value": dc_stock,
            "store_weeks_of_supply": store_stock / max(cogs_week, 1e-9), "weeks_of_supply": stock / max(cogs_week, 1e-9),
            "turns": cogs_week * 52 / max(stock, 1e-9), "gmroi": margin / window_days * 365 / max(stock, 1e-9),
            "dc_fill_rate": float(k["dc_fill_rate"][0]) if "dc_fill_rate" in k else float("nan"),
            "dc_on_order_value": float(k["dc_on_order_value"][0]) if "dc_on_order_value" in k else float("nan"),
        })
    return pd.DataFrame(rows)
