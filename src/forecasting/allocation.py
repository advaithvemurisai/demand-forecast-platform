"""Constrained inventory allocation using PuLP.

Scenario-based newsvendor LP. With demand scenarios ``d[i, k]`` (equally likely) and
stock already at each location ``h[i]``, allocate ``x[i]`` units of a shared supply to
maximise expected value of fulfilled demand::

    max  sum_i v_i * (1/K) * sum_k s[i, k]
    s.t. s[i, k] <= h[i] + x[i],  s[i, k] <= d[i, k]   (sales cannot exceed stock or demand)
         sum_i x[i] <= inventory
         h[i] + x[i] >= min_fill * forecast[i]         (optional service floor)

Allocating against net need (what is already on the shelf counts) rather than gross
forecast stops the LP sending stock where it is already sitting.

Expected sales ``E[min(D, x)]`` are concave in ``x``, so the LP is exact for the
scenario distribution and naturally spreads stock toward uncertain, high-value demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def allocate_inventory(
    demand: pd.DataFrame,
    inventory: float,
    scenarios: np.ndarray | None = None,
    quantity_column: str = "forecast",
    value_column: str | None = None,
    min_fill: float = 0.0,
    on_hand: np.ndarray | None = None,
) -> pd.DataFrame:
    """Allocate ``inventory`` across the rows of ``demand``.

    ``scenarios`` is (rows x K); without it the point forecast is used as a single
    deterministic scenario. ``value_column`` weights each unit (e.g. sell price).
    ``on_hand`` is stock already at each row (default none).
    """
    import pulp

    if inventory < 0:
        raise ValueError("inventory must be non-negative")
    point = demand[quantity_column].to_numpy(dtype=float)
    if scenarios is None:
        scenarios = point[:, None]
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.shape[0] != len(demand):
        raise ValueError("scenarios must have one row per demand row")
    weights = demand[value_column].to_numpy(dtype=float) if value_column else np.ones(len(demand))
    held = np.zeros(len(point)) if on_hand is None else np.maximum(np.asarray(on_hand, dtype=float), 0)
    floor = np.maximum(min_fill * point - held, 0)
    if floor.sum() > inventory + 1e-9:
        raise ValueError(f"min_fill requires {floor.sum():.1f} units but only {inventory:.1f} are available")

    rows, n_scenarios = scenarios.shape
    problem = pulp.LpProblem("inventory_allocation", pulp.LpMaximize)
    allocation = [pulp.LpVariable(f"x_{i}", lowBound=float(floor[i]), upBound=float(max(scenarios[i].max() - held[i], floor[i], 0))) for i in range(rows)]
    sold = [[pulp.LpVariable(f"s_{i}_{k}", lowBound=0, upBound=float(scenarios[i, k])) for k in range(n_scenarios)] for i in range(rows)]
    problem += pulp.lpSum(weights[i] / n_scenarios * sold[i][k] for i in range(rows) for k in range(n_scenarios))
    for i in range(rows):
        for k in range(n_scenarios):
            problem += sold[i][k] <= allocation[i] + float(held[i])
    problem += pulp.lpSum(allocation) <= inventory
    problem.solve(solver())
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"Allocation LP finished with status {status}")

    allocated = np.array([variable.value() or 0.0 for variable in allocation])
    result = demand.copy()
    result["allocated_quantity"] = allocated
    stock = (allocated + held)[:, None]
    expected_sales = np.minimum(scenarios, stock).mean(axis=1)
    expected_demand = scenarios.mean(axis=1)
    result["expected_fill_rate"] = np.divide(expected_sales, expected_demand, out=np.ones(rows), where=expected_demand > 0)
    result["stockout_risk"] = (scenarios > stock + 1e-9).mean(axis=1)
    return result


def solver():
    """Prefer HiGHS (native wheels via ``highspy``); PuLP's bundled CBC is x86-only on macOS."""
    import pulp

    available = pulp.listSolvers(onlyAvailable=True)
    if "HiGHS" in available:
        return pulp.HiGHS(msg=False)
    if "PULP_CBC_CMD" in available:
        return pulp.PULP_CBC_CMD(msg=False)
    raise RuntimeError("No LP solver available; install one with `pip install highspy`")


def proportional_allocation(demand: pd.DataFrame, inventory: float, quantity_column: str = "forecast", need: np.ndarray | None = None) -> np.ndarray:
    """Benchmark rule: split supply pro rata to net need (default: the point forecast), never shipping more than is needed."""
    point = demand[quantity_column].to_numpy(dtype=float) if need is None else np.maximum(np.asarray(need, dtype=float), 0)
    shipped = inventory if need is None else min(inventory, point.sum())
    return point / point.sum() * shipped if point.sum() else np.zeros(len(point))


def fairness_violations(fill: pd.DataFrame, floor: float = 0.5, max_weeks: int = 2) -> pd.DataFrame:
    """Nodes whose fill rate stayed below ``floor`` for more than ``max_weeks`` consecutive weeks.

    ``fill`` has one row per (node_id, period) with a ``fill_rate`` column, ordered by period.
    Returns one row per offending node with its longest run of short weeks.
    """
    rows = []
    for node, group in fill.groupby("node_id", sort=False):
        longest = run = 0
        for short in (group["fill_rate"].to_numpy() < floor - 1e-9):
            run = run + 1 if short else 0
            longest = max(longest, run)
        if longest > max_weeks:
            rows.append({"node_id": node, "longest_short_run": longest})
    return pd.DataFrame(rows, columns=["node_id", "longest_short_run"])
