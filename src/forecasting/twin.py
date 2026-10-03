"""Inventory digital twin: a daily, stateful simulation of DC -> stores -> shoppers.

numpy only, so the same file runs under CPython (pipeline, tests) and in the browser (Pyodide).

Each day, for every product x store and every replication:
    receive in-transit stock -> sell (lost sales when stock runs out) -> on review days order up
    to a policy target -> the DC rations what it has -> shipments enter the transit pipeline.
The DC reorders weekly from its supplier against the *aggregate* forecast, so the pooled
(coherent) forecast sets DC stock while store-level forecasts set store stock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RATIONING = ("proportional", "days_of_cover", "value")
POLICIES = ("forecast_reorder", "last_week_reorder")


@dataclass
class TwinConfig:
    lead_time_days: int = 1  # DC -> store transit
    supplier_lead_days: int = 7  # supplier -> DC
    dc_review_days: int = 7
    dc_safety_share: float = 0.10  # DC safety stock as a share of cover demand (risk pooling)
    presentation_min: float = 2.0
    case_pack: int = 1
    rationing: str = "days_of_cover"
    warmup_days: int = 14
    holding_cost_weekly: float = 0.005  # share of unit value per week of storage
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

    For each week of the window, draw one historical week (the same one for every product, so shared shocks such as a
    holiday week that beat the forecast everywhere are kept) and replay each product's actual daily sales from that
    week, rescaled by ``(forecast now + floor) / (forecast then + floor)`` for the week. Compared with adding past
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


def ration(order: np.ndarray, available: np.ndarray, mode: str, cover: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Split ``available`` DC units (reps,) across store orders (reps, n).

    proportional: everyone gets the same share of what they asked for.
    days_of_cover: products with the least stock cover are served first (the retail norm).
    value: highest margin per unit first (the 'optimiser' view).
    """
    if mode not in RATIONING:
        raise ValueError(f"rationing must be one of {RATIONING}")
    total = order.sum(axis=1)
    if mode == "proportional":
        share = np.divide(available, total, out=np.ones_like(total), where=total > 0)
        return order * np.minimum(share, 1.0)[:, None]
    key = cover if mode == "days_of_cover" else -np.broadcast_to(value, order.shape)
    ranked = np.argsort(key, axis=1, kind="stable")
    sorted_order = np.take_along_axis(order, ranked, axis=1)
    before = np.cumsum(sorted_order, axis=1) - sorted_order
    sorted_alloc = np.clip(available[:, None] - before, 0, sorted_order)
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
) -> dict:
    """Run the twin over ``demand`` (reps, n, days) and return per-replication KPIs and a timeline.

    ``forecast`` (n, days) drives the targets; ``safety_stock`` (n,) is added to every target.
    ``unlimited_stock`` is the replay check: with infinite stock, sales must equal demand.
    """
    cfg = config or TwinConfig()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    reps, n, days = demand.shape
    demand = apply_shock(demand, shock).astype(np.float32)
    lead = max(int(cfg.lead_time_days), 1)
    supplier = max(int(cfg.supplier_lead_days) + (shock.supplier_delay_days if shock else 0), 1)
    review = np.maximum(np.asarray(review_days, dtype=int), 1)
    cover = review + lead
    offset = np.arange(n) % review
    target_plan = window_targets(forecast.astype(np.float64), cover) + np.asarray(safety_stock, dtype=np.float64)[:, None]
    target_plan = round_up(np.maximum(target_plan, cfg.presentation_min), cfg.case_pack).astype(np.float32)
    dc_cover = max(int(cfg.supplier_lead_days), 1) + cfg.dc_review_days  # the DC plans for the normal lead time: a delay is a surprise
    dc_forecast = forecast.sum(axis=0)
    dc_window = window_targets(dc_forecast[None, :], np.array([dc_cover]))[0] * (1 + cfg.dc_safety_share)
    value = np.broadcast_to(np.asarray(unit_value, dtype=np.float32), (n,))
    dc_factor = np.ones(days, dtype=np.float32)
    if shock and shock.dc_supply_factor != 1.0:
        start, stop = shock.dc_days if shock.dc_days else (0, days)
        dc_factor[start:stop] = shock.dc_supply_factor

    on_hand = np.broadcast_to(target_plan[:, 0], (reps, n)).copy()
    in_transit = np.zeros((reps, n, lead), dtype=np.float32)
    dc_on_hand = np.full(reps, dc_window[0], dtype=np.float64)
    dc_pipeline = np.zeros((reps, supplier), dtype=np.float64)
    recent = np.zeros((reps, n, 7), dtype=np.float32)  # last 7 days of observed sales (for the naive policy)
    if unlimited_stock:
        on_hand[:] = 1e9

    keep = slice(cfg.warmup_days, days)
    sold_day = np.zeros((reps, n, days), dtype=np.float32)
    lost_day = np.zeros((reps, n, days), dtype=np.float32)
    stock_start = np.zeros((reps, n, days), dtype=np.float32)
    end_stock = np.zeros((reps, n, days), dtype=np.float32)
    flow = {"received": 0.0, "shipped": 0.0, "ordered": 0.0}
    for day in range(days):
        arrived = in_transit[:, :, 0].copy()
        on_hand += arrived
        in_transit = np.concatenate([in_transit[:, :, 1:], np.zeros((reps, n, 1), dtype=np.float32)], axis=2)
        dc_on_hand += dc_pipeline[:, 0] * dc_factor[day]
        flow["received"] += float(arrived.sum())
        dc_pipeline = np.concatenate([dc_pipeline[:, 1:], np.zeros((reps, 1))], axis=1)

        stock_start[:, :, day] = on_hand
        sales = np.minimum(on_hand, demand[:, :, day])
        on_hand = on_hand - sales if not unlimited_stock else on_hand
        sold_day[:, :, day] = sales
        lost_day[:, :, day] = demand[:, :, day] - sales
        recent[:, :, day % 7] = sales

        due = (day - offset) % review == 0
        if policy == "forecast_reorder":
            target = np.broadcast_to(target_plan[:, day], (reps, n))
        else:
            seen = min(day + 1, 7)
            naive = recent.sum(axis=2) / seen * cover[None, :]
            target = round_up(np.maximum(naive, cfg.presentation_min), cfg.case_pack)
        position = on_hand + in_transit.sum(axis=2)
        order = np.where(due[None, :], round_up(np.maximum(target - position, 0), cfg.case_pack), 0).astype(np.float64)
        if unlimited_stock:
            order[:] = 0
        flow["ordered"] += float(order.sum())
        rate = np.maximum(forecast[:, day], 1e-3)[None, :]
        shipped = ration(order, np.maximum(dc_on_hand, 0), cfg.rationing, position / rate, value)
        dc_on_hand = dc_on_hand - shipped.sum(axis=1)
        in_transit[:, :, lead - 1] += shipped.astype(np.float32)
        flow["shipped"] += float(shipped.sum())

        if day % cfg.dc_review_days == 0:
            dc_position = dc_on_hand + dc_pipeline.sum(axis=1)
            dc_pipeline[:, supplier - 1] += np.maximum(dc_window[day] - dc_position, 0)
        end_stock[:, :, day] = on_hand

    demand_k, sold_k, lost_k = demand[:, :, keep], sold_day[:, :, keep], lost_day[:, :, keep]
    kept_days = demand_k.shape[2]
    total_demand = demand_k.sum(axis=(1, 2))
    inventory_value = (end_stock[:, :, keep] * value[None, :, None]).mean(axis=2).sum(axis=1)
    lost_value = (lost_k * value[None, :, None]).sum(axis=(1, 2))
    sales_value = (sold_k * value[None, :, None]).sum(axis=(1, 2))
    kpis = {
        "fill_rate": sold_k.sum(axis=(1, 2)) / np.maximum(total_demand, 1e-9),
        "in_stock_pct": (stock_start[:, :, keep] > 0).mean(axis=(1, 2)),
        "units_lost": lost_k.sum(axis=(1, 2)),
        "lost_sales_value": lost_value,
        "sales_value": sales_value,
        "inventory_value": inventory_value,
        "weeks_of_supply": end_stock[:, :, keep].mean(axis=2).sum(axis=1) / np.maximum(total_demand / kept_days * 7, 1e-9),
        "holding_cost": inventory_value * cfg.holding_cost_weekly * kept_days / 7,
        "units_demanded": total_demand,
        "units_sold": sold_k.sum(axis=(1, 2)),
    }
    result = {"kpis": kpis, "days": kept_days, "flow": flow}
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
            timeline["on_hand"][:, column] = end_stock[:, mask][:, :, keep].sum(axis=1).mean(axis=0)
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


SERVICE_GRID = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99)


def make_shock(spec: dict | None, categories: np.ndarray, warmup: int) -> Shock | None:
    """Build a Shock from a plain dict; day indices count from the start of the forecast window."""
    if not spec:
        return None
    shift = lambda days: None if days is None else (int(days[0]) + warmup, int(days[1]) + warmup)
    category = spec.get("category")
    mask = None if category is None else (np.asarray(categories).astype(str) == category)
    return Shock(
        demand_scale=float(spec.get("demand_scale", 1.0)), demand_rows=mask, demand_days=shift(spec.get("days")),
        dc_supply_factor=float(spec.get("dc_factor", 1.0)), dc_days=shift(spec.get("dc_days")),
        supplier_delay_days=int(spec.get("delay", 0)),
    )


def run_store(history, forecast, actual, scale, safety, review, price, config: TwinConfig, policy: str, reps: int, seed: int,
              score_paths=None, shock: Shock | None = None, replay: bool = False, groups=None, risk_days: int | None = None, chunk: int = 25, per_series: bool = False,
              past_actual=None, past_forecast=None) -> dict:
    """Warm up on recent history, then simulate the forecast window with sampled (or replayed) demand.

    The shared entry point for the pipeline, the API and the browser: arrays in, per-replication KPIs out.
    """
    warm = config.warmup_days
    history = np.asarray(history, dtype=np.float32)
    recent = history[:, -warm:] if warm else history[:, :0]
    base = history[:, -28:].mean(axis=1, keepdims=True)
    extended = np.concatenate([np.repeat(base, warm, axis=1), forecast], axis=1)
    rng = np.random.default_rng(seed)
    parts, risk_parts, timeline = [], [], None
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
        result = simulate(demand, extended, safety, review, price, config, policy, shock, groups=groups, risk_days=risk_days)
        parts.append(result["kpis"])
        if "risk" in result:
            risk_parts.append((size, result["risk"]))
        timeline = result.get("timeline", timeline)
    out = {"kpis": {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}, "timeline": timeline, "n": int(len(forecast))}
    if risk_parts:
        total = sum(size for size, _ in risk_parts)
        out["risk"] = {key: sum(size * risk[key] for size, risk in risk_parts) / total for key in ("expected_lost", "stockout_prob")}
    return out


def simulate_bundle(bundle: dict, request: dict) -> dict:
    """Run a what-if on one store's saved inputs and return JSON-ready results.

    ``bundle`` holds numpy arrays (forecast, scale, safety_by_service, current_safety, review, price,
    history, score_paths, dept_ids, cat_ids); ``request`` is a plain dict (policy, service, scenario, reps,
    seed, rationing). The baseline and the scenario share the same random demand, so differences are
    caused by the shock alone.
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
        safety = np.asarray(bundle["safety_by_service"], dtype=np.float64)[:, levels.index(float(service))]
    config = TwinConfig(
        lead_time_days=int(bundle.get("lead_time_days", 1)), presentation_min=float(bundle.get("presentation_min", 2.0)),
        case_pack=int(bundle.get("case_pack", 1)), rationing=request.get("rationing", "days_of_cover"),
    )
    kwargs = dict(
        history=bundle["history"], forecast=np.asarray(bundle["forecast"], dtype=np.float64), actual=None, scale=bundle["scale"],
        safety=safety, review=bundle["review"], price=bundle["price"], config=config, policy=policy, reps=reps,
        seed=int(request.get("seed", 1)), score_paths=np.asarray(bundle.get("score_paths", np.zeros((1, 28))), dtype=np.float32), groups=bundle["dept_ids"],
        per_series=bool(np.asarray(bundle.get("score_per_series", False))),
        past_actual=None if "past_actual" not in bundle else np.asarray(bundle["past_actual"], dtype=np.float32),
        past_forecast=None if "past_forecast" not in bundle else np.asarray(bundle["past_forecast"], dtype=np.float32),
    )
    base = run_store(**kwargs)
    scenario = request.get("scenario")
    shocked = run_store(**kwargs, shock=make_shock(scenario, bundle["cat_ids"], config.warmup_days)) if scenario else None
    def pack(result):
        timeline = result["timeline"]
        return {
            "kpis": kpi_summary(result["kpis"]),
            "timeline": {
                "dept": [str(label) for label in timeline["group"]],
                "on_hand": np.round(timeline["on_hand"], 2).tolist(), "demand": np.round(timeline["demand"], 2).tolist(),
                "lost": np.round(timeline["lost"], 2).tolist(),
            },
        }
    return {"policy": policy, "service": service, "reps": reps, "baseline": pack(base), "scenario": pack(shocked) if shocked else None}
