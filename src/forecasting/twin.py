"""Inventory digital twin: a daily, stateful simulation of supplier -> DC -> stores -> shoppers.

numpy only, so the same file runs under CPython (pipeline, tests) and in the browser (Pyodide).

Each day, for every product x store and every replication:
    receive in-transit stock -> sell (lost sales when stock runs out) -> on review days order up
    to a policy target -> the DC rations each product's stock across the stores that ordered it ->
    shipments enter the transit pipeline.
The DC holds stock per product and serves every store in the run, so all stores of a network are simulated together:
they compete for the same DC units and share the same demand week. It reorders each product weekly from its
supplier against that product's network forecast, with a pooled (square-root) safety stock that also covers
supplier lead-time variability. Each supplier order draws its own lead time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RATIONING = ("proportional", "days_of_cover", "value")
POLICIES = ("forecast_reorder", "last_week_reorder")
SERVICE_GRID = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998)


@dataclass
class TwinConfig:
    lead_time_days: int = 1  # DC -> store transit
    supplier_lead_days: int = 7  # supplier -> DC, as planned
    supplier_lead_sd_days: float = 1.0  # spread of the actual supplier lead time around the plan (each order draws its own)
    dc_review_days: int = 7
    dc_service_z: float = 1.65  # DC safety stock: z for ~95% cycle service over the DC's cover window
    dc_safety_factor: float = 1.0  # scales the pooled DC safety stock (1 = the textbook amount)
    presentation_min: float = 2.0
    case_pack: int = 1
    rationing: str = "days_of_cover"
    warmup_days: int = 14
    holding_cost_weekly: float = 0.005  # share of unit *cost* per week of storage, when no per-product rate is given
    seed: int = 0


@dataclass
class Shock:
    """A what-if: scale demand on some products and days, cut DC supply, or delay the supplier."""

    demand_scale: float = 1.0
    demand_rows: np.ndarray | None = None  # boolean (n,): which products; None = all
    demand_days: tuple[int, int] | None = None  # [start, stop) day indices; None = all
    dc_supply_factor: float = 1.0  # share of ordered stock the DC actually receives
    dc_days: tuple[int, int] | None = None  # days the cut applies to (by receipt day)
    supplier_delay_days: int = 0  # extra supplier lead time
    delay_days: tuple[int, int] | None = None  # days whose supplier orders are delayed (by order day); None = all
    replan_after: int | None = None  # days after the delay starts that the DC plans on the longer lead time; None = never
    # Responses a planner can take, each with its own cost:
    demand_planned: bool = False  # the spike is known in advance, so stores and the DC order for it
    prebuild_days: int = 0  # extra days of forecast demand the DC holds from the start of the run (pre-build)
    prebuild_until: int | None = None  # day the extra cover is dropped (the disruption is over); None = kept
    expedite_share: float = 0.0  # share of the shortfall a late order leaves that goes by a faster route (planned lead time)
    backup_share: float = 0.0  # share of a DC supply cut a second supplier makes up
    premium: float = 0.0  # extra cost of expedited or second-supplier units, as a share of unit cost


class Groups:
    """Rows labelled 0..m-1, with fast per-group sums over the last axis (rows sorted once, then reduceat)."""

    def __init__(self, labels: np.ndarray):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.perm = np.argsort(self.labels, kind="stable")
        ordered = self.labels[self.perm]
        self.starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if len(ordered) else np.zeros(0, dtype=int)
        self.sizes = np.diff(np.r_[self.starts, len(ordered)])
        self.m = len(self.starts)
        self.ordered = ordered

    @classmethod
    def of(cls, keys) -> "Groups":
        return cls(np.unique(np.asarray(keys), return_inverse=True)[1].ravel())

    def sum(self, values: np.ndarray) -> np.ndarray:
        """(..., n) -> (..., m)."""
        return np.add.reduceat(values[..., self.perm], self.starts, axis=-1)


def apply_shock(demand: np.ndarray, shock: Shock | None) -> np.ndarray:
    """Scale demand for the shocked products and days (returns a copy)."""
    if shock is None or shock.demand_scale == 1.0:
        return demand
    out = demand.copy()
    rows = np.ones(demand.shape[1], dtype=bool) if shock.demand_rows is None else np.asarray(shock.demand_rows, dtype=bool)
    start, stop = shock.demand_days if shock.demand_days else (0, demand.shape[2])
    out[:, rows, start:stop] *= shock.demand_scale
    return out


def sample_paths(forecast: np.ndarray, scale: np.ndarray, score_paths: np.ndarray, reps: int, rng: np.random.Generator, block: int = 7, per_series: bool = False) -> np.ndarray:
    """Demand scenarios (reps, n, days): forecast + scale x a resampled block of past errors.

    ``score_paths`` rows are time-ordered error paths (the pipeline stores raw unit errors and passes scale = 1). Whole ``block``-day pieces are resampled so
    within-week error correlation is kept. With ``per_series`` row *i* holds product *i*'s own past errors, so
    every product keeps its own noise level; otherwise rows are pooled across products, which gives slow,
    intermittent sellers the wild swings of fast ones.
    """
    n, days = forecast.shape
    rows_available, length = score_paths.shape
    if per_series and rows_available != n:
        raise ValueError("per_series score_paths needs one row per product")
    block = min(block, length)
    n_blocks = -(-days // block)
    starts_per_path = length // block
    if per_series:
        row = np.broadcast_to(np.arange(n)[None, :, None], (reps, n, n_blocks))
    else:
        row = rng.integers(0, rows_available, size=(reps, n, n_blocks))
    start = rng.integers(0, starts_per_path, size=(reps, n, n_blocks)) * block
    index = start[..., None] + np.arange(block)
    scores = score_paths[row[..., None], index].reshape(reps, n, n_blocks * block)[:, :, :days]
    return np.clip(forecast[None] + scale[None, :, None] * scores, 0, None).astype(np.float32)


def sample_demand(forecast: np.ndarray, past_actual: np.ndarray, past_forecast: np.ndarray, reps: int, rng: np.random.Generator,
                  block: int = 7, floor: float = 0.5, ratio_cap: tuple[float, float] = (0.25, 4.0)) -> np.ndarray:
    """Demand scenarios (reps, n, days) by a time-block bootstrap of what actually sold.

    For each week of the window, draw one historical week (the same one for every product and store, so shared shocks
    such as a holiday week that beat the forecast everywhere are kept) and replay each product's actual daily sales from
    that week, rescaled by ``(forecast now + floor) / (forecast then + floor)`` for the week. Compared with adding past
    errors to today's forecast this keeps zero days and bursts for intermittent products, never goes negative, and
    carries any systematic over- or under-forecast the model really had. ``past_*`` are (n, days) with whole weeks
    laid out so that column 0 falls on the same weekday as column 0 of ``forecast``.
    """
    n, days = forecast.shape
    length = past_actual.shape[1]
    block = min(block, length)
    n_blocks = -(-days // block)
    n_past = length // block
    pick = rng.integers(0, n_past, size=(reps, 1, n_blocks))
    out = np.empty((reps, n, n_blocks * block), dtype=np.float32)
    padded = np.concatenate([forecast, np.repeat(forecast[:, -1:], n_blocks * block - days, axis=1)], axis=1) if n_blocks * block > days else forecast
    now = padded.reshape(n, n_blocks, block).sum(axis=2)  # (n, blocks): forecast for each week of the window
    past_a = past_actual[:, : n_past * block].reshape(n, n_past, block)
    past_f = past_forecast[:, : n_past * block].reshape(n, n_past, block).sum(axis=2)
    for b in range(n_blocks):
        chosen = pick[:, 0, b]  # (reps,)
        ratio = np.clip((now[:, b][None, :] + floor) / (past_f[:, chosen].T + floor), *ratio_cap)  # (reps, n)
        out[:, :, b * block:(b + 1) * block] = past_a[:, chosen, :].transpose(1, 0, 2) * ratio[:, :, None]
    return out[:, :, :days]


def round_up(quantity: np.ndarray, case_pack: int) -> np.ndarray:
    return np.ceil(np.maximum(quantity, 0) / case_pack - 1e-9) * case_pack if case_pack > 1 else np.maximum(quantity, 0)


def ration(order: np.ndarray, available: np.ndarray, mode: str, cover: np.ndarray, value: np.ndarray, groups: Groups | None = None) -> np.ndarray:
    """Split DC stock across store orders (reps, n).

    ``available`` is (reps, m): units of each product the DC holds, with ``groups`` mapping rows (product x store) to
    products. Without ``groups`` every row draws on one shared pool, ``available`` (reps,).

    proportional: every store gets the same share of what it asked for.
    days_of_cover: the store with the least stock cover is served first (the retail norm).
    value: highest margin per unit first (the 'optimiser' view; exact for the deterministic allocation LP).
    """
    if mode not in RATIONING:
        raise ValueError(f"rationing must be one of {RATIONING}")
    reps, n = order.shape
    groups = groups or Groups(np.zeros(n, dtype=np.int64))
    available = np.asarray(available, dtype=np.float64).reshape(reps, -1)
    if mode == "proportional":
        total = groups.sum(order)
        share = np.minimum(np.divide(available, total, out=np.ones_like(total), where=total > 0), 1.0)
        return order * share[:, groups.labels]
    key = cover if mode == "days_of_cover" else -np.broadcast_to(value, order.shape)
    ranked = np.lexsort((key, np.broadcast_to(groups.labels, order.shape)), axis=-1)  # by product, then priority
    sorted_order = np.take_along_axis(order, ranked, axis=1)
    before = np.cumsum(sorted_order, axis=1) - sorted_order
    before = before - np.repeat(before[:, groups.starts], groups.sizes, axis=1)  # units promised earlier in the same product
    sorted_alloc = np.clip(available[:, groups.ordered] - before, 0, sorted_order)
    allocation = np.empty_like(order)
    np.put_along_axis(allocation, ranked, sorted_alloc, axis=1)
    return allocation


def window_targets(forecast: np.ndarray, cover_days: np.ndarray) -> np.ndarray:
    """Forecast demand over the next ``cover_days`` days for every product and day: (n, days)."""
    n, days = forecast.shape
    pad = int(cover_days.max()) + 1
    tail = np.repeat(forecast[:, -7:].mean(axis=1, keepdims=True), pad, axis=1)
    padded = np.concatenate([forecast, tail], axis=1)
    cumulative = np.concatenate([np.zeros((n, 1)), np.cumsum(padded, axis=1)], axis=1)
    day = np.arange(days)[None, :]
    rows = np.arange(n)[:, None]
    return cumulative[rows, day + 1 + cover_days[:, None]] - cumulative[rows, day + 1]


def span(forecast: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Forecast demand over days [start, stop) per row, extending past the end with the last week's average."""
    days = forecast.shape[1]
    inside = forecast[:, start:min(stop, days)].sum(axis=1)
    beyond = max(stop - max(start, days), 0)
    return inside + beyond * forecast[:, -7:].mean(axis=1)


def dc_safety_stock(safety: np.ndarray, store_cover: np.ndarray, dc_forecast: np.ndarray, products: Groups, dc_cover: int, cfg: TwinConfig) -> np.ndarray:
    """Pooled DC safety stock per product.

    Store safety stocks protect ``store_cover`` days; rescaled to the DC's cover window (square root of time) and
    pooled across stores (square root of the sum of squares) they give the demand-risk part. Supplier lead-time
    variability adds ``z x sd(lead) x daily demand``. Pooling is why a DC needs less buffer than its stores combined.
    """
    demand_part = products.sum(((np.maximum(safety, 0) * np.sqrt(dc_cover / np.maximum(store_cover, 1))) ** 2)[None])[0]
    lead_part = (cfg.dc_service_z * cfg.supplier_lead_sd_days * dc_forecast.mean(axis=1)) ** 2
    return np.sqrt(demand_part + lead_part) * cfg.dc_safety_factor


def simulate(
    demand: np.ndarray,
    forecast: np.ndarray,
    safety_stock: np.ndarray,
    review_days: np.ndarray,
    unit_value: np.ndarray,
    config: TwinConfig | None = None,
    policy: str = "forecast_reorder",
    shock: Shock | None = None,
    unlimited_stock: bool = False,
    groups: np.ndarray | None = None,
    risk_days: int | None = None,
    products: np.ndarray | None = None,
    unit_cost: np.ndarray | None = None,
    holding_rate: np.ndarray | None = None,
    kpi_groups: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run the twin over ``demand`` (reps, n, days) and return per-replication KPIs and a timeline.

    Rows are product x store. ``forecast`` (n, days) drives the targets; ``safety_stock`` (n,) is added to every target
    (for both policies, so they differ only in how they estimate demand). ``products`` (n,) says which rows are the same
    product, i.e. draw on the same DC stock (default: every row its own product). ``unit_value`` is the shelf price;
    ``unit_cost`` (default: the price) values inventory and holding cost; ``holding_rate`` is the weekly holding cost
    as a share of cost. ``kpi_groups`` (n,) also returns KPIs per label (e.g. per store) under ``kpis_by``.
    ``unlimited_stock`` is the replay check: with infinite stock, sales must equal demand.
    """
    cfg = config or TwinConfig()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    reps, n, days = demand.shape
    demand = apply_shock(demand, shock).astype(np.float32)
    if shock is not None and shock.demand_planned and shock.demand_scale != 1.0:  # a known event: the plan includes it
        forecast = forecast.astype(np.float64).copy()
        rows = np.ones(n, dtype=bool) if shock.demand_rows is None else np.asarray(shock.demand_rows, dtype=bool)
        start, stop = shock.demand_days if shock.demand_days else (0, days)
        forecast[rows, start:stop] *= shock.demand_scale
    lead = max(int(cfg.lead_time_days), 1)
    review = np.maximum(np.asarray(review_days, dtype=int), 1)
    cover = review + lead
    offset = np.arange(n) % review
    safety = np.asarray(safety_stock, dtype=np.float64) * np.ones(n)
    target_plan = window_targets(forecast.astype(np.float64), cover) + safety[:, None]
    target_plan = round_up(np.maximum(target_plan, cfg.presentation_min), cfg.case_pack).astype(np.float32)
    price = np.broadcast_to(np.asarray(unit_value, dtype=np.float64), (n,))
    cost = price if unit_cost is None else np.broadcast_to(np.asarray(unit_cost, dtype=np.float64), (n,))
    margin = price - cost
    holding = np.full(n, cfg.holding_cost_weekly) if holding_rate is None else np.broadcast_to(np.asarray(holding_rate, dtype=np.float64), (n,))
    priority_value = margin if unit_cost is not None else price

    # DC: one stock per product, serving every store row of that product.
    product = Groups.of(products) if products is not None else Groups(np.arange(n))
    m = product.m
    product_cost = product.sum(cost[None])[0] / product.sizes  # DC unit cost per product
    product_holding = product.sum(holding[None])[0] / product.sizes  # and its weekly holding rate
    dc_forecast = product.sum(forecast.astype(np.float64)[None].transpose(0, 2, 1))[0].T  # (m, days)
    planned_lead = max(int(cfg.supplier_lead_days), 1)
    delay = int(shock.supplier_delay_days) if shock else 0
    delay_window = (shock.delay_days or (0, days)) if delay else (0, 0)
    replan_day = delay_window[0] + shock.replan_after if delay and shock.replan_after is not None else None
    dc_review = max(int(cfg.dc_review_days), 1)
    dc_cover = planned_lead + dc_review
    dc_safety = dc_safety_stock(safety, cover, dc_forecast, product, dc_cover, cfg)
    dc_target = window_targets(dc_forecast, np.full(m, dc_cover)) + dc_safety[:, None]
    prebuild = int(shock.prebuild_days) if shock else 0
    if prebuild:  # hold extra days of cover from the start of the run, ahead of the disruption, then let it run down
        extra = np.zeros((m, days))
        extra[:, : shock.prebuild_until if shock.prebuild_until is not None else days] = span(dc_forecast, 0, prebuild)[:, None]
        dc_target = dc_target + extra
    if replan_day is not None:  # once the planner knows, the DC covers the longer lead time
        long_cover = dc_cover + delay
        long_safety = dc_safety_stock(safety, cover, dc_forecast, product, long_cover, cfg)
        dc_target_long = window_targets(dc_forecast, np.full(m, long_cover)) + long_safety[:, None]
        if prebuild:
            dc_target_long = dc_target_long + extra
    spread = int(np.ceil(3 * cfg.supplier_lead_sd_days))
    pipe_len = planned_lead + delay + spread + 1
    dc_factor = np.ones(days, dtype=np.float32)
    if shock and shock.dc_supply_factor != 1.0:
        start, stop = shock.dc_days if shock.dc_days else (0, days)
        dc_factor[start:stop] = shock.dc_supply_factor

    on_hand = np.broadcast_to(target_plan[:, 0], (reps, n)).copy()
    in_transit = np.zeros((reps, n, lead), dtype=np.float32)
    # Start the DC in steady state: safety stock plus cover until the first in-flight order lands, and earlier orders in flight.
    arrivals = sorted(a for a in range(planned_lead - dc_review, 0, -dc_review))
    landings = [*arrivals, planned_lead]
    dc_on_hand = np.broadcast_to(dc_safety + span(dc_forecast, 0, landings[0]), (reps, m)).copy()
    dc_pipeline = np.zeros((reps, m, pipe_len), dtype=np.float64)
    for index, arrival in enumerate(arrivals):
        dc_pipeline[:, :, arrival] += span(dc_forecast, arrival, landings[index + 1])
    rep_index, product_index = np.arange(reps)[:, None], np.arange(m)[None, :]
    recent = np.zeros((reps, n, 7), dtype=np.float32)  # last 7 days of observed sales (for the naive policy)
    if unlimited_stock:
        on_hand[:] = 1e9

    keep = slice(cfg.warmup_days, days)
    sold_day = np.zeros((reps, n, days), dtype=np.float32)
    lost_day = np.zeros((reps, n, days), dtype=np.float32)
    in_stock = np.zeros((reps, n, days), dtype=bool)
    end_stock = np.zeros((reps, n, days), dtype=np.float32)
    flow = {"received": 0.0, "shipped": 0.0, "ordered": 0.0}
    expedite = float(shock.expedite_share) if shock else 0.0
    backup = float(shock.backup_share) if shock else 0.0
    premium_rate = float(shock.premium) if shock else 0.0
    premium = np.zeros(reps)  # cost of the response's expedited / second-supplier units
    dc_value = np.zeros((reps, m))  # DC stock at cost, summed over the kept days
    dc_on_order = np.zeros(reps)  # stock ordered from the supplier but not yet at the DC, at cost, summed over kept days
    dc_asked = np.zeros(reps)  # store orders and what the DC shipped against them, over the kept days
    dc_sent = np.zeros(reps)
    dc_day = np.zeros((2, days))  # mean DC stock and stock on order at cost, per day
    for day in range(days):
        arrived = in_transit[:, :, 0].copy()
        on_hand += arrived
        in_transit[:, :, :-1] = in_transit[:, :, 1:].copy()
        in_transit[:, :, -1] = 0
        landing = dc_pipeline[:, :, 0]
        factor = dc_factor[day]
        if factor < 1.0 and backup > 0:  # a second supplier makes up part of the cut, at a premium
            covered = landing * (1.0 - factor) * backup
            premium += (covered * product_cost[None]).sum(axis=1) * premium_rate
            factor = factor + (1.0 - factor) * backup
        dc_on_hand += landing * factor
        flow["received"] += float(arrived.sum())
        dc_pipeline[:, :, :-1] = dc_pipeline[:, :, 1:].copy()
        dc_pipeline[:, :, -1] = 0

        in_stock[:, :, day] = on_hand > 0
        sales = np.minimum(on_hand, demand[:, :, day])
        on_hand = on_hand - sales if not unlimited_stock else on_hand
        sold_day[:, :, day] = sales
        lost_day[:, :, day] = demand[:, :, day] - sales
        recent[:, :, day % 7] = sales

        due = (day - offset) % review == 0
        if policy == "forecast_reorder":
            target = np.broadcast_to(target_plan[:, day], (reps, n))
        else:  # current practice: last week's sales rate over the same cover, plus the same safety stock
            seen = min(day + 1, 7)
            naive = recent.sum(axis=2) / seen * cover[None, :] + safety[None, :]
            target = round_up(np.maximum(naive, cfg.presentation_min), cfg.case_pack)
        position = on_hand + in_transit.sum(axis=2)
        order = np.where(due[None, :], round_up(np.maximum(target - position, 0), cfg.case_pack), 0).astype(np.float64)
        if unlimited_stock:
            order[:] = 0
        flow["ordered"] += float(order.sum())
        rate = np.maximum(forecast[:, day], 1e-3)[None, :]
        shipped = ration(order, np.maximum(dc_on_hand, 0), cfg.rationing, position / rate, priority_value, product)
        dc_on_hand = dc_on_hand - product.sum(shipped)
        in_transit[:, :, lead - 1] += shipped.astype(np.float32)
        flow["shipped"] += float(shipped.sum())
        if day >= cfg.warmup_days:
            dc_asked += order.sum(axis=1)
            dc_sent += shipped.sum(axis=1)

        if day % dc_review == 0:
            plan = dc_target_long if replan_day is not None and day >= replan_day else dc_target
            dc_position = dc_on_hand + dc_pipeline.sum(axis=2)
            quantity = np.maximum(plan[:, day][None, :] - dc_position, 0)
            late = delay if delay_window[0] <= day < delay_window[1] else 0
            noise = np.rint(rng.normal(0, cfg.supplier_lead_sd_days, (reps, m))).astype(int) if cfg.supplier_lead_sd_days > 0 else np.zeros((reps, m), dtype=int)
            arrive = np.clip(planned_lead + late + np.clip(noise, -spread, spread), 1, pipe_len)
            if late and expedite > 0:
                # Expedite only what bridges the gap: the units the DC would be short of (against its safety stock)
                # before an on-time order could land, out of this late order. The rest waits for the slow route.
                cover = span(dc_forecast, day, day + planned_lead)[None, :] + dc_safety[None, :]
                gap = cover - (np.maximum(dc_on_hand, 0) + dc_pipeline[:, :, :planned_lead].sum(axis=2))
                fast = np.clip(gap, 0, quantity) * expedite
                on_time = np.clip(planned_lead + np.clip(noise, -spread, spread), 1, pipe_len)
                dc_pipeline[rep_index, product_index, on_time - 1] += fast
                premium += (fast * product_cost[None]).sum(axis=1) * premium_rate
                quantity = quantity - fast
            dc_pipeline[rep_index, product_index, arrive - 1] += quantity
        end_stock[:, :, day] = on_hand
        stock_value = np.maximum(dc_on_hand, 0) * product_cost[None]
        order_value = (dc_pipeline.sum(axis=2) * product_cost[None]).sum(axis=1)
        dc_day[:, day] = stock_value.sum(axis=1).mean(), order_value.mean()
        if day >= cfg.warmup_days:
            dc_value += stock_value
            dc_on_order += order_value

    demand_k, sold_k, lost_k, stock_k, in_stock_k = demand[:, :, keep], sold_day[:, :, keep], lost_day[:, :, keep], end_stock[:, :, keep], in_stock[:, :, keep]
    kept_days = demand_k.shape[2]

    def kpis(rows) -> dict[str, np.ndarray]:
        d, s, lost, stock = demand_k[:, rows], sold_k[:, rows], lost_k[:, rows], stock_k[:, rows]
        total_demand = d.sum(axis=(1, 2))
        mean_stock = stock.mean(axis=2)
        inventory = (mean_stock * cost[rows][None]).sum(axis=1)
        return {
            "fill_rate": s.sum(axis=(1, 2)) / np.maximum(total_demand, 1e-9),
            "in_stock_pct": in_stock_k[:, rows].mean(axis=(1, 2)),
            "units_lost": lost.sum(axis=(1, 2)),
            "lost_sales_value": (lost.sum(axis=2) * price[rows][None]).sum(axis=1),
            "lost_margin": (lost.sum(axis=2) * margin[rows][None]).sum(axis=1),
            "sales_value": (s.sum(axis=2) * price[rows][None]).sum(axis=1),
            "margin_value": (s.sum(axis=2) * margin[rows][None]).sum(axis=1),
            "inventory_value": inventory,
            "weeks_of_supply": mean_stock.sum(axis=1) / np.maximum(total_demand / kept_days * 7, 1e-9),
            "holding_cost": (mean_stock * (cost * holding)[rows][None]).sum(axis=1) * kept_days / 7,
            "units_demanded": total_demand,
            "units_sold": s.sum(axis=(1, 2)),
        }

    network = kpis(slice(None))
    network.update({
        "dc_inventory_value": dc_value.sum(axis=1) / kept_days, "dc_on_order_value": dc_on_order / kept_days,
        "dc_holding_cost": (dc_value * product_holding[None]).sum(axis=1) / 7,
        "dc_fill_rate": dc_sent / np.maximum(dc_asked, 1e-9), "response_cost": premium,
    })
    result = {"kpis": network, "days": kept_days, "flow": flow, "dc_timeline": {"on_hand": dc_day[0, keep], "on_order": dc_day[1, keep]}}
    if products is not None:  # mean DC stock at cost per product, labelled, for per-category roll-ups
        result["dc_by_product"] = {"products": np.unique(np.asarray(products)), "value": (dc_value / kept_days).mean(axis=0)}
    if kpi_groups is not None:
        labels = np.asarray(kpi_groups)
        result["kpis_by"] = {str(label): kpis(np.flatnonzero(labels == label)) for label in np.unique(labels)}
    if risk_days:
        window = lost_k[:, :, :risk_days]
        result["risk"] = {"expected_lost": window.sum(axis=2).mean(axis=0), "stockout_prob": (window.sum(axis=2) > 0).mean(axis=0)}
    if groups is not None:
        groups = np.asarray(groups)
        labels = np.unique(groups)
        shape = (kept_days, len(labels))
        timeline = {"group": labels, "on_hand": np.zeros(shape), "demand": np.zeros(shape), "lost": np.zeros(shape)}
        for column, label in enumerate(labels):
            mask = groups == label
            timeline["on_hand"][:, column] = stock_k[:, mask].sum(axis=1).mean(axis=0)
            timeline["demand"][:, column] = demand_k[:, mask].sum(axis=1).mean(axis=0)
            timeline["lost"][:, column] = lost_k[:, mask].sum(axis=1).mean(axis=0)
        result["timeline"] = timeline
    # Units never appear or vanish: shelf stock = start + arrivals - sales, and shipments = arrivals + still in transit.
    shelf_gap = abs(float(on_hand.sum()) - (float(target_plan[:, 0].sum()) * reps + flow["received"] - float(sold_day.sum())))
    transit_gap = abs(flow["shipped"] - flow["received"] - float(in_transit.sum()))
    result["conservation_gap"] = 0.0 if unlimited_stock else shelf_gap + transit_gap
    return result


def kpi_summary(kpis: dict[str, np.ndarray], band: float = 0.90) -> dict[str, dict[str, float]]:
    """Mean and central ``band`` interval of each KPI across replications."""
    low, high = (1 - band) / 2 * 100, (1 + band) / 2 * 100
    return {
        name: {"mean": float(np.mean(values)), "lower": float(np.percentile(values, low)), "upper": float(np.percentile(values, high))}
        for name, values in kpis.items()
    }


def make_shock(spec: dict | None, categories: np.ndarray, warmup: int) -> Shock | None:
    """Build a Shock from a plain dict; day indices count from the start of the forecast window.

    Keys: demand_scale, category, days, dc_factor, dc_days, delay (extra supplier days), delay_days (which order days
    are late; default: every order from the start of the window), replan_after (days until the DC plans for the delay),
    and the responses: planned (the spike is in the forecast), prebuild_days, prebuild_until (window day the extra
    cover is dropped), expedite_share (of the shortfall a late order leaves), backup_share, premium.
    """
    if not spec:
        return None
    shift = lambda days: None if days is None else (int(days[0]) + warmup, int(days[1]) + warmup)
    category = spec.get("category")
    mask = None if category is None else (np.asarray(categories).astype(str) == category)
    delay = int(spec.get("delay", 0))
    replan = spec.get("replan_after")
    return Shock(
        demand_scale=float(spec.get("demand_scale", 1.0)), demand_rows=mask, demand_days=shift(spec.get("days")),
        dc_supply_factor=float(spec.get("dc_factor", 1.0)), dc_days=shift(spec.get("dc_days")),
        supplier_delay_days=delay, delay_days=(shift(spec.get("delay_days")) or (warmup, 10**6)) if delay else None,
        replan_after=None if replan is None else int(replan),
        demand_planned=bool(spec.get("planned", False)), prebuild_days=int(spec.get("prebuild_days", 0)),
        prebuild_until=None if spec.get("prebuild_until") is None else int(spec["prebuild_until"]) + warmup,
        expedite_share=float(spec.get("expedite_share", 0.0)), backup_share=float(spec.get("backup_share", 0.0)),
        premium=float(spec.get("premium", 0.0)),
    )


def run_store(history, forecast, actual, scale, safety, review, price, config: TwinConfig, policy: str, reps: int, seed: int,
              score_paths=None, shock: Shock | None = None, replay: bool = False, groups=None, risk_days: int | None = None, chunk: int = 25, per_series: bool = False,
              past_actual=None, past_forecast=None, products=None, cost=None, holding_rate=None, kpi_groups=None) -> dict:
    """Warm up on recent history, then simulate the forecast window with sampled (or replayed) demand.

    The shared entry point for the pipeline, the API and the browser: arrays in, per-replication KPIs out. Despite the
    name it runs whatever rows it is given; pass every store of a network (with ``products``) so they share one DC.
    """
    warm = config.warmup_days
    history = np.asarray(history, dtype=np.float32)
    recent = history[:, -warm:] if warm else history[:, :0]
    base = history[:, -28:].mean(axis=1, keepdims=True)
    extended = np.concatenate([np.repeat(base, warm, axis=1), forecast], axis=1)
    rng = np.random.default_rng(seed)
    parts, by_parts, risk_parts, timelines, dc_parts = [], [], [], [], []
    sizes = [1] if replay else [min(chunk, reps - start) for start in range(0, reps, chunk)]
    for size in sizes:
        if replay:
            demand = np.concatenate([recent, actual], axis=1)[None].astype(np.float32)
        else:
            if past_actual is not None:
                future = sample_demand(forecast, past_actual, past_forecast, size, rng)
            else:
                future = sample_paths(forecast, scale, score_paths, size, rng, per_series=per_series)
            demand = np.concatenate([np.broadcast_to(recent, (size, *recent.shape)).astype(np.float32), future], axis=2)
        result = simulate(demand, extended, safety, review, price, config, policy, shock, groups=groups, risk_days=risk_days,
                          products=products, unit_cost=cost, holding_rate=holding_rate, kpi_groups=kpi_groups, rng=rng)
        parts.append(result["kpis"])
        if "kpis_by" in result:
            by_parts.append(result["kpis_by"])
        if "risk" in result:
            risk_parts.append((size, result["risk"]))
        if "timeline" in result:
            timelines.append((size, result["timeline"]))
        dc_parts.append((size, result["dc_timeline"], result.get("dc_by_product")))
    stack = lambda dicts: {name: np.concatenate([part[name] for part in dicts]) for name in dicts[0]}
    out = {"kpis": stack(parts), "timeline": None, "n": int(len(forecast))}
    if by_parts:
        out["kpis_by"] = {label: stack([part[label] for part in by_parts]) for label in by_parts[0]}
    total = sum(sizes)
    if risk_parts:
        out["risk"] = {key: sum(size * risk[key] for size, risk in risk_parts) / total for key in ("expected_lost", "stockout_prob")}
    if timelines:  # average over every chunk, weighted by its replications
        first = timelines[0][1]
        out["timeline"] = {"group": first["group"], **{key: sum(size * t[key] for size, t in timelines) / total for key in ("on_hand", "demand", "lost")}}
    out["dc_timeline"] = {key: sum(size * t[key] for size, t, _ in dc_parts) / total for key in ("on_hand", "on_order")}
    if dc_parts[0][2] is not None:
        out["dc_by_product"] = {"products": dc_parts[0][2]["products"], "value": sum(size * p["value"] for size, _, p in dc_parts) / total}
    return out


def simulate_bundle(bundle: dict, request: dict) -> dict:
    """Run a what-if on a network's saved inputs and return JSON-ready results.

    ``bundle`` holds numpy arrays, one row per product x store (forecast, scale, safety_by_service, current_safety,
    review, price, history, score_paths or past_actual/past_forecast, dept_ids, cat_ids, and optionally store_ids,
    item_ids, cost, holding_rate); ``request`` is a plain dict (policy, service, scenario, reps, seed, rationing).
    Every store is simulated together so they share the DC; KPIs come back for the network ("all") and per store.
    The baseline and the scenario share the same random demand, so differences are caused by the shock alone.
    """
    reps = int(min(max(int(request.get("reps", 20)), 1), 50))
    policy = request.get("policy", "forecast_reorder")
    service = request.get("service", "current")
    if service == "current":
        safety = np.asarray(bundle["current_safety"], dtype=np.float64)
    else:
        levels = list(SERVICE_GRID)
        if float(service) not in levels:
            raise ValueError(f"service must be 'current' or one of {levels}")
        table = np.asarray(bundle["safety_by_service"], dtype=np.float64)
        if table.shape[1] != len(levels):
            raise ValueError(f"these inputs carry {table.shape[1]} service levels, not {len(levels)}")
        safety = table[:, levels.index(float(service))]
    config = TwinConfig(
        lead_time_days=int(bundle.get("lead_time_days", 1)), presentation_min=float(bundle.get("presentation_min", 2.0)),
        case_pack=int(bundle.get("case_pack", 1)), rationing=request.get("rationing", "days_of_cover"),
    )
    n = len(bundle["forecast"])
    stores = np.asarray(bundle["store_ids"]).astype(str) if "store_ids" in bundle else np.full(n, "all")
    depts = np.asarray(bundle["dept_ids"]).astype(str)
    multi = len(np.unique(stores)) > 1
    kwargs = dict(
        history=bundle["history"], forecast=np.asarray(bundle["forecast"], dtype=np.float64), actual=None, scale=bundle["scale"],
        safety=safety, review=bundle["review"], price=bundle["price"], config=config, policy=policy, reps=reps,
        seed=int(request.get("seed", 1)), score_paths=np.asarray(bundle.get("score_paths", np.zeros((1, 28))), dtype=np.float32),
        groups=np.char.add(np.char.add(stores, "|"), depts) if multi else depts,
        per_series=bool(np.asarray(bundle.get("score_per_series", False))),
        past_actual=None if "past_actual" not in bundle else np.asarray(bundle["past_actual"], dtype=np.float32),
        past_forecast=None if "past_forecast" not in bundle else np.asarray(bundle["past_forecast"], dtype=np.float32),
        products=None if "item_ids" not in bundle else np.asarray(bundle["item_ids"]).astype(str),
        cost=None if "cost" not in bundle else np.asarray(bundle["cost"], dtype=np.float64),
        holding_rate=None if "holding_rate" not in bundle else np.asarray(bundle["holding_rate"], dtype=np.float64),
        kpi_groups=stores if multi else None,
    )
    base = run_store(**kwargs)
    scenario = request.get("scenario")
    shocked = run_store(**kwargs, shock=make_shock(scenario, bundle["cat_ids"], config.warmup_days)) if scenario else None

    def pack(result):
        timeline = result["timeline"]
        kpis = {"all": kpi_summary(result["kpis"]), **{label: kpi_summary(values) for label, values in result.get("kpis_by", {}).items()}}
        return {
            "kpis": kpis,
            "timeline": {
                "group": [str(label) for label in timeline["group"]],
                "on_hand": np.round(timeline["on_hand"], 2).tolist(), "demand": np.round(timeline["demand"], 2).tolist(),
                "lost": np.round(timeline["lost"], 2).tolist(),
            },
            "dc": {key: np.round(values, 2).tolist() for key, values in result["dc_timeline"].items()},
        }
    return {"policy": policy, "service": service, "reps": reps, "stores": sorted(set(stores.tolist())) if multi else [], "baseline": pack(base), "scenario": pack(shocked) if shocked else None}
