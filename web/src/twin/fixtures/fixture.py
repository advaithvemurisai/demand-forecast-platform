"""Deterministic twin fixture shared by CPython (pytest) and Pyodide (vitest).

Built from arithmetic only (no random draws), so both runtimes start from identical arrays; the simulator's own
random sampling then uses numpy's PCG64 generator, whose stream is stable across versions.
"""
import numpy as np

N, DAYS = 24, 28
REQUESTS = {
    "baseline": {"reps": 12, "seed": 5},
    "spike": {"reps": 12, "seed": 5, "scenario": {"demand_scale": 1.5, "category": "FOODS", "days": [7, 14]}},
    "dc_cut_last_week": {"reps": 12, "seed": 5, "policy": "last_week_reorder", "service": 0.9, "scenario": {"dc_factor": 0.7, "dc_days": [7, 21]}},
    # The path the website uses: demand bootstrapped from past weeks of actual sales.
    "bootstrap_delay": {"reps": 12, "seed": 5, "bootstrap": True, "scenario": {"delay": 7, "replan_after": 7}},
    "late_shipment": {"reps": 12, "seed": 5, "service": 0.995, "scenario": {"delay": 7, "delay_days": [0, 7]}},
    "bootstrap_value_rationing": {"reps": 12, "seed": 5, "bootstrap": True, "rationing": "value", "scenario": {"dc_factor": 0.6, "dc_days": [0, 28]}},
    # Planner responses: expedite part of a lasting delay; a second supplier behind a DC cut.
    "bootstrap_expedite": {"reps": 12, "seed": 5, "bootstrap": True, "scenario": {"delay": 7, "replan_after": 7, "expedite_share": 0.5, "premium": 0.2, "prebuild_days": 7}},
    "backup_supplier": {"reps": 12, "seed": 5, "scenario": {"dc_factor": 0.6, "dc_days": [7, 21], "backup_share": 0.5, "premium": 0.1}},
}


def bundle_for(request):
    """The fixture bundle, with past weeks of sales attached when the request exercises the demand bootstrap."""
    bundle = fixture_bundle()
    if request.get("bootstrap"):
        index = np.arange(N, dtype=np.float64)[:, None]
        day = np.arange(56, dtype=np.float64)[None, :]
        rate = 1.0 + (np.arange(N) % 7)[:, None] * 0.8
        bundle["past_forecast"] = (rate * (1.0 + 0.3 * np.sin(2 * np.pi * day / 7))).astype(np.float32)
        bundle["past_actual"] = np.floor(bundle["past_forecast"] * (1.0 + 0.5 * np.sin(index * 3.1 + day * 1.7))).clip(0).astype(np.float32)
    return bundle


def fixture_bundle():
    index = np.arange(N, dtype=np.float64)
    day = np.arange(DAYS, dtype=np.float64)
    rate = 1.0 + (index % 7) * 0.8
    weekly = 1.0 + 0.3 * np.sin(2 * np.pi * day / 7)
    forecast = (rate[:, None] * weekly[None, :]).astype(np.float32)
    history = (rate[:, None] * (1.0 + 0.2 * np.cos(2 * np.pi * (day[None, :] + index[:, None]) / 7))).round().astype(np.float32)
    path_rows = np.arange(300, dtype=np.float64)
    score_paths = (np.sin(path_rows[:, None] * 12.9898 + day[None, :] * 78.233) * 1.4).astype(np.float32)
    return {
        "history": history, "forecast": forecast, "scale": np.maximum(forecast[:, 0], 1.0).astype(np.float32),
        "current_safety": (1.0 + (index % 5)).astype(np.float32),
        "safety_by_service": (np.linspace(1.0, 6.0, 8)[None, :] * (1.0 + (index % 3))[:, None]).astype(np.float32),
        "review": np.where(index % 2 == 0, 2, 7).astype(np.int8), "price": (1.0 + (index % 4) * 0.5).astype(np.float32),
        "score_paths": score_paths, "dept_ids": np.array(["FOODS_1" if i < 12 else "HOBBIES_1" for i in range(N)]),
        "cat_ids": np.array(["FOODS" if i < 12 else "HOBBIES" for i in range(N)]),
        # Two stores carrying the same six products in each department, so the DC rations each product between them.
        "store_ids": np.array(["CA_1" if i % 2 == 0 else "CA_2" for i in range(N)]),
        "item_ids": np.array([f"ITEM_{i // 2}" for i in range(N)]),
        "cost": ((1.0 + (index % 4) * 0.5) * 0.7).astype(np.float32),
        "holding_rate": np.where(index < 12, 0.015, 0.005).astype(np.float32),
    }
