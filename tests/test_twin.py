import math
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import twin
from forecasting.allocation import allocate_inventory


def _run(demand, forecast, safety, review=3, cfg=None, **kwargs):
    n = forecast.shape[0]
    return twin.simulate(demand, forecast, np.asarray(safety, dtype=float) * np.ones(n), np.full(n, review), np.ones(n), cfg or twin.TwinConfig(presentation_min=0, warmup_days=0), **kwargs)


def test_single_period_matches_newsvendor_closed_form():
    mu, sigma, z = 10.0, 3.0, 1.2816
    reps = 40_000
    rng = np.random.default_rng(0)
    demand = np.clip(rng.normal(mu, sigma, (reps, 1, 1)), 0, None).astype(np.float32)
    forecast = np.full((1, 1), mu)
    # review 1 + lead 1 -> cover 2 days, so a safety of (z*sigma - mu) puts day-0 stock at mu + z*sigma
    cfg = twin.TwinConfig(presentation_min=0, warmup_days=0, lead_time_days=1)
    result = _run(demand, forecast, z * sigma - mu, review=1, cfg=cfg)
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    upper_tail = 0.5 * (1 - math.erf(z / math.sqrt(2)))
    expected_lost = sigma * (phi - z * upper_tail)
    assert result["kpis"]["units_lost"].mean() == pytest.approx(expected_lost, rel=0.05)


def test_exact_forecast_and_deterministic_demand_never_stocks_out_after_warmup():
    n, days = 20, 60
    rate = np.linspace(1, 8, n)[:, None]
    forecast = np.repeat(rate, days, axis=1)
    demand = forecast[None].astype(np.float32)
    result = _run(demand, forecast, 0.0, cfg=twin.TwinConfig(presentation_min=0, warmup_days=14))
    assert result["kpis"]["units_lost"].max() < 1e-3
    assert result["kpis"]["fill_rate"].min() > 0.999


def test_units_are_conserved_for_every_policy_and_rationing_rule():
    rng = np.random.default_rng(1)
    n, days = 30, 50
    forecast = np.repeat(rng.uniform(0.5, 6, (n, 1)), days, axis=1)
    demand = rng.poisson(forecast, (4, n, days)).astype(np.float32)
    for policy in twin.POLICIES:
        for mode in twin.RATIONING:
            cfg = twin.TwinConfig(rationing=mode, case_pack=3, warmup_days=7)
            result = _run(demand, forecast, 3.0, cfg=cfg, policy=policy)
            assert result["conservation_gap"] < 1e-3 * max(result["flow"]["shipped"], 1)


def test_replay_with_unlimited_stock_reproduces_demand():
    rng = np.random.default_rng(2)
    forecast = np.full((10, 40), 3.0)
    demand = rng.poisson(3.0, (3, 10, 40)).astype(np.float32)
    result = _run(demand, forecast, 2.0, unlimited_stock=True)
    np.testing.assert_allclose(result["kpis"]["units_sold"], result["kpis"]["units_demanded"], rtol=1e-6)
    assert result["kpis"]["units_lost"].max() == 0


def test_value_rationing_matches_the_allocation_lp():
    order = np.array([[10.0, 8.0, 6.0, 9.0]])
    value = np.array([1.0, 5.0, 3.0, 2.0])
    available = np.array([15.0])
    greedy = twin.ration(order, available, "value", cover=np.zeros_like(order), value=value)[0]
    lp = allocate_inventory(pd.DataFrame({"forecast": order[0], "value": value}), 15.0, value_column="value")["allocated_quantity"].to_numpy()
    assert (greedy * value).sum() == pytest.approx((lp * value).sum())


def test_rationing_respects_supply_orders_and_priorities():
    order = np.array([[10.0, 10.0, 10.0]])
    cover = np.array([[5.0, 1.0, 3.0]])
    value = np.ones(3)
    for mode in twin.RATIONING:
        alloc = twin.ration(order, np.array([14.0]), mode, cover, value)
        assert alloc.sum() <= 14 + 1e-9 and (alloc <= order + 1e-9).all()
    proportional = twin.ration(order, np.array([15.0]), "proportional", cover, value)[0]
    np.testing.assert_allclose(proportional, [5.0, 5.0, 5.0])
    by_cover = twin.ration(order, np.array([14.0]), "days_of_cover", cover, value)[0]
    np.testing.assert_allclose(by_cover, [0.0, 10.0, 4.0])  # lowest cover served first
    with pytest.raises(ValueError):
        twin.ration(order, np.array([5.0]), "nope", cover, value)


def test_same_seed_same_results_and_different_seed_differs():
    forecast = np.full((5, 30), 4.0)
    paths = np.random.default_rng(0).normal(0, 1, (200, 28))
    def draw(seed):
        return twin.sample_paths(forecast, np.full(5, 2.0), paths, 6, np.random.default_rng(seed))
    np.testing.assert_array_equal(draw(7), draw(7))
    assert not np.array_equal(draw(7), draw(8))
    assert draw(7).shape == (6, 5, 30) and (draw(7) >= 0).all()
    first = _run(draw(7), forecast, 2.0)["kpis"]["fill_rate"]
    second = _run(draw(7), forecast, 2.0)["kpis"]["fill_rate"]
    np.testing.assert_array_equal(first, second)


def test_cutting_dc_supply_never_raises_service():
    rng = np.random.default_rng(3)
    n, days = 40, 70
    forecast = np.repeat(rng.uniform(1, 6, (n, 1)), days, axis=1)
    demand = rng.poisson(forecast, (6, n, days)).astype(np.float32)
    cfg = twin.TwinConfig(presentation_min=0, warmup_days=14)
    base = _run(demand, forecast, 3.0, cfg=cfg)["kpis"]
    cut = _run(demand, forecast, 3.0, cfg=cfg, shock=twin.Shock(dc_supply_factor=0.4, dc_days=(10, 70)))["kpis"]
    assert cut["in_stock_pct"].mean() <= base["in_stock_pct"].mean() + 1e-9
    assert cut["lost_sales_value"].mean() > base["lost_sales_value"].mean()
    late = _run(demand, forecast, 3.0, cfg=cfg, shock=twin.Shock(supplier_delay_days=7))["kpis"]
    assert late["lost_sales_value"].mean() >= base["lost_sales_value"].mean() - 1e-6  # a surprise delay never helps
    spike = _run(demand, forecast, 3.0, cfg=cfg, shock=twin.Shock(demand_scale=1.5, demand_days=(20, 30)))["kpis"]
    assert spike["lost_sales_value"].mean() > base["lost_sales_value"].mean()


def test_presentation_minimum_keeps_a_slow_mover_on_the_shelf():
    days = 60
    forecast = np.full((1, days), 0.1)
    demand = np.zeros((1, 1, days), dtype=np.float32)
    demand[0, 0, 20::10] = 1.0  # one unit every ten days
    stocked = _run(demand, forecast, 0.0, cfg=twin.TwinConfig(presentation_min=2, warmup_days=14))
    bare = _run(demand, forecast, 0.0, cfg=twin.TwinConfig(presentation_min=0, warmup_days=14))
    assert stocked["kpis"]["fill_rate"][0] == pytest.approx(1.0)
    assert bare["kpis"]["fill_rate"][0] < 1.0


def test_case_pack_orders_are_whole_cases():
    rng = np.random.default_rng(4)
    forecast = np.full((6, 40), 2.5)
    demand = rng.poisson(2.5, (2, 6, 40)).astype(np.float32)
    result = _run(demand, forecast, 2.0, cfg=twin.TwinConfig(case_pack=6, presentation_min=0, warmup_days=7))
    assert result["flow"]["ordered"] % 6 == pytest.approx(0, abs=1e-6) or abs(result["flow"]["ordered"] % 6 - 6) < 1e-6


def test_forecast_reorder_beats_last_week_reorder_on_service():
    rng = np.random.default_rng(5)
    n, days = 60, 84
    base = rng.uniform(1, 8, (n, 1))
    weekly = np.array([0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 1.4])
    forecast = base * np.tile(weekly, days // 7)[None, :]
    demand = rng.poisson(forecast, (8, n, days)).astype(np.float32)
    cfg = twin.TwinConfig(warmup_days=14)
    safety = 1.28 * np.sqrt(forecast.mean(axis=1) * 3)
    smart = _run(demand, forecast, safety, cfg=cfg)["kpis"]
    naive = _run(demand, forecast, safety, cfg=cfg, policy="last_week_reorder")["kpis"]
    assert smart["fill_rate"].mean() > naive["fill_rate"].mean()


def test_timeline_groups_sum_to_total():
    rng = np.random.default_rng(6)
    forecast = np.full((6, 30), 3.0)
    demand = rng.poisson(3.0, (2, 6, 30)).astype(np.float32)
    result = _run(demand, forecast, 2.0, cfg=twin.TwinConfig(warmup_days=5), groups=np.array(list("aabbcc")))
    timeline = result["timeline"]
    assert timeline["demand"].shape == (25, 3)
    np.testing.assert_allclose(timeline["demand"].sum(), demand[:, :, 5:].sum() / 2, rtol=1e-4)


def test_kpi_summary_bands_contain_the_mean():
    summary = twin.kpi_summary({"fill_rate": np.linspace(0.9, 1.0, 101)})["fill_rate"]
    assert summary["lower"] < summary["mean"] < summary["upper"]


def test_twin_module_is_standalone_numpy_only():
    """The browser (Pyodide) loads this one file, so it must not need pandas, scipy or the modelling stack."""
    path = Path(twin.__file__)
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('twin_standalone', {str(path)!r})\n"
        "module = importlib.util.module_from_spec(spec); sys.modules['twin_standalone'] = module; spec.loader.exec_module(module)\n"
        "heavy = {'pandas', 'scipy', 'pulp', 'lightgbm', 'statsmodels', 'prophet', 'sklearn'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_single_store_hosted_run_stays_inside_memory_budget():
    n, days, reps = 3000, 70, 20
    rng = np.random.default_rng(7)
    forecast = np.repeat(rng.uniform(0.2, 6, (n, 1)), days, axis=1)
    paths = rng.normal(0, 1, (5000, 28))
    tracemalloc.start()
    demand = twin.sample_paths(forecast, np.maximum(forecast[:, 0], 1), paths, reps, rng)
    twin.simulate(demand, forecast, np.full(n, 2.0), np.full(n, 3), np.ones(n), twin.TwinConfig())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 450 * 1024 * 1024


def _bundle(n=40, seed=0):
    rng = np.random.default_rng(seed)
    forecast = np.tile(rng.uniform(1, 6, (n, 1)), (1, 28)).astype(np.float32)
    return dict(
        history=rng.poisson(3, (n, 28)).astype(np.float32), forecast=forecast, scale=np.maximum(forecast[:, 0], 1),
        current_safety=np.full(n, 3.0), safety_by_service=np.tile(np.linspace(1, 6, len(twin.SERVICE_GRID)), (n, 1)), review=np.full(n, 3), price=np.ones(n),
        score_paths=rng.normal(0, 1, (500, 28)), dept_ids=np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        cat_ids=np.array(["FOODS"] * (n // 2) + ["HOBBIES"] * (n - n // 2)),
    )


def test_simulate_bundle_is_json_ready_deterministic_and_shock_hurts():
    import json

    spike = {"demand_scale": 1.6, "category": "FOODS", "days": [7, 14]}
    first = twin.simulate_bundle(_bundle(), {"reps": 10, "seed": 3, "scenario": spike})
    second = twin.simulate_bundle(_bundle(), {"reps": 10, "seed": 3, "scenario": spike})
    assert json.dumps(first) == json.dumps(second)
    assert first["scenario"]["kpis"]["all"]["lost_sales_value"]["mean"] > first["baseline"]["kpis"]["all"]["lost_sales_value"]["mean"]
    assert len(first["baseline"]["timeline"]["on_hand"]) == 28 and first["baseline"]["timeline"]["group"] == ["A", "B"]


def test_simulate_bundle_caps_reps_and_validates_service():
    assert twin.simulate_bundle(_bundle(), {"reps": 5000, "seed": 1})["reps"] == 50
    with pytest.raises(ValueError):
        twin.simulate_bundle(_bundle(), {"service": 0.5})
    high = twin.simulate_bundle(_bundle(), {"reps": 10, "service": 0.99})["baseline"]["kpis"]["all"]
    low = twin.simulate_bundle(_bundle(), {"reps": 10, "service": 0.80})["baseline"]["kpis"]["all"]
    assert high["inventory_value"]["mean"] > low["inventory_value"]["mean"]


def test_committed_browser_snapshot_matches_cpython():
    """web/src/twin/fixtures/snapshot.json is what Pyodide must reproduce; regenerate it when twin.py changes."""
    import json
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import make_twin_snapshot

    committed = json.loads((root / "web" / "src" / "twin" / "fixtures" / "snapshot.json").read_text())
    live = json.loads(json.dumps(make_twin_snapshot.compute()))
    assert committed == live, "twin.py changed: run `python scripts/make_twin_snapshot.py`"


def test_per_series_sampling_keeps_each_products_own_noise():
    rng = np.random.default_rng(0)
    n, days = 2, 56
    forecast = np.full((n, days), 10.0)
    quiet = rng.normal(0, 0.2, 84)
    wild = rng.normal(0, 3.0, 84)
    paths = np.vstack([quiet, wild])
    draw = twin.sample_paths(forecast, np.ones(n), paths, 400, np.random.default_rng(1), per_series=True)
    spread = draw.std(axis=(0, 2))
    assert spread[0] < 0.5 < 2.0 < spread[1]  # the quiet product stays quiet
    pooled = twin.sample_paths(forecast, np.ones(n), paths, 400, np.random.default_rng(1))
    assert abs(pooled.std(axis=(0, 2))[0] - pooled.std(axis=(0, 2))[1]) < 1.0  # pooling blurs them together
    with pytest.raises(ValueError):
        twin.sample_paths(forecast, np.ones(n), np.zeros((5, 28)), 3, rng, per_series=True)


def test_simulate_bundle_honours_the_per_series_flag():
    bundle = _bundle()
    bundle["score_paths"] = np.random.default_rng(2).normal(0, 1, (len(bundle["forecast"]), 56))
    bundle["score_per_series"] = np.bool_(True)
    result = twin.simulate_bundle(bundle, {"reps": 6, "seed": 2})
    assert 0 < result["baseline"]["kpis"]["all"]["fill_rate"]["mean"] <= 1


def _past(n=400, weeks=8, seed=0, rate=2.0):
    rng = np.random.default_rng(seed)
    forecast = np.full((n, 7 * weeks), rate)
    return rng.poisson(rate, forecast.shape).astype(float), forecast


def test_sample_demand_rescales_to_todays_forecast_and_stays_non_negative():
    actual, past_forecast = _past()
    same = twin.sample_demand(np.full((400, 28), 2.0), actual, past_forecast, 200, np.random.default_rng(1))
    doubled = twin.sample_demand(np.full((400, 28), 4.0), actual, past_forecast, 200, np.random.default_rng(1))
    assert same.mean() == pytest.approx(2.0, rel=0.05)
    assert doubled.mean() == pytest.approx(2 * same.mean(), rel=0.08)
    assert same.min() >= 0 and same.shape == (200, 400, 28)


def test_sample_demand_keeps_intermittent_zero_pattern():
    rng = np.random.default_rng(3)
    actual = (rng.random((300, 56)) < 0.1).astype(float) * 3  # a unit-pack sale about one day in ten
    past_forecast = np.full((300, 56), 0.3)
    sample = twin.sample_demand(np.full((300, 28), 0.3), actual, past_forecast, 50, np.random.default_rng(2))
    assert (sample == 0).mean() == pytest.approx((actual == 0).mean(), abs=0.03)


def test_sample_demand_keeps_a_shared_demand_shock_across_products():
    actual, past_forecast = _past(weeks=4)
    actual[:, 7:14] *= 2.0  # one week sold twice the forecast in every product
    sample = twin.sample_demand(np.full((400, 28), 2.0), actual, past_forecast, 400, np.random.default_rng(4))
    weekly_total = sample[:, :, :7].sum(axis=(1, 2))  # first window week, store-wide, per simulated future
    independent_noise = np.sqrt(400 * 7 * 2.0) / (400 * 7 * 2.0)  # ~1.3%: what independent draws would give
    assert weekly_total.std() / weekly_total.mean() > 5 * independent_noise
    again = twin.sample_demand(np.full((400, 28), 2.0), actual, past_forecast, 400, np.random.default_rng(4))
    np.testing.assert_array_equal(sample, again)


def test_simulate_bundle_uses_the_demand_bootstrap_when_history_is_supplied():
    bundle = _bundle(n=40)
    past, past_forecast = _past(n=40, weeks=8, rate=3.0)
    bundle.update(past_actual=past, past_forecast=past_forecast)
    result = twin.simulate_bundle(bundle, {"reps": 8, "seed": 4})
    assert 0.5 < result["baseline"]["kpis"]["all"]["fill_rate"]["mean"] <= 1
    spike = twin.simulate_bundle(bundle, {"reps": 8, "seed": 4, "scenario": {"demand_scale": 1.6, "days": [7, 14]}})
    assert spike["scenario"]["kpis"]["all"]["lost_sales_value"]["mean"] > spike["baseline"]["kpis"]["all"]["lost_sales_value"]["mean"]


def test_dc_stock_is_per_product_and_shared_across_stores():
    """Rows of the same product compete for its DC stock; one product's stock never fills another product's order."""
    order = np.array([[10.0, 10.0, 10.0, 10.0]])
    groups = twin.Groups.of(np.array(["a", "a", "b", "b"]))
    available = np.array([[5.0, 100.0]])  # product a is short, product b is not
    cover = np.array([[3.0, 1.0, 2.0, 0.5]])
    by_cover = twin.ration(order, available, "days_of_cover", cover, np.ones(4), groups)[0]
    np.testing.assert_allclose(by_cover, [0.0, 5.0, 10.0, 10.0])  # the a-store with less cover gets the 5 units
    proportional = twin.ration(order, available, "proportional", cover, np.ones(4), groups)[0]
    np.testing.assert_allclose(proportional, [2.5, 2.5, 10.0, 10.0])
    by_value = twin.ration(order, available, "value", cover, np.array([1.0, 3.0, 1.0, 1.0]), groups)[0]
    np.testing.assert_allclose(by_value, [0.0, 5.0, 10.0, 10.0])


def test_network_run_shares_one_dc_and_reports_each_store():
    rng = np.random.default_rng(8)
    items, days = 15, 56
    rate = np.repeat(rng.uniform(1, 5, (items, 1)), days, axis=1)
    forecast = np.vstack([rate, rate])  # two stores selling the same products
    demand = rng.poisson(forecast, (4, 2 * items, days)).astype(np.float32)
    products = np.tile(np.arange(items), 2)
    stores = np.repeat(["S1", "S2"], items)
    cfg = twin.TwinConfig(warmup_days=14)
    args = (demand, forecast, np.full(2 * items, 2.0), np.full(2 * items, 3), np.ones(2 * items), cfg)
    result = twin.simulate(*args, products=products, kpi_groups=stores, shock=twin.Shock(dc_supply_factor=0.5, dc_days=(14, 56)))
    assert result["conservation_gap"] < 1e-3 * result["flow"]["shipped"]
    by = result["kpis_by"]
    np.testing.assert_allclose(by["S1"]["units_lost"] + by["S2"]["units_lost"], result["kpis"]["units_lost"], rtol=1e-5)
    np.testing.assert_allclose(by["S1"]["inventory_value"] + by["S2"]["inventory_value"], result["kpis"]["inventory_value"], rtol=1e-5)


def test_pooled_dc_safety_is_below_the_sum_of_store_safety():
    forecast = np.full((4, 28), 5.0)
    products = twin.Groups.of(np.array([0, 0, 0, 0]))
    cfg = twin.TwinConfig(supplier_lead_sd_days=0.0)
    pooled = twin.dc_safety_stock(np.full(4, 10.0), np.full(4, 14), products.sum(forecast.T[None])[0].T, products, 14, cfg)[0]
    assert pooled == pytest.approx(20.0)  # sqrt(4) x 10, not 4 x 10
    shaky = twin.dc_safety_stock(np.full(4, 10.0), np.full(4, 14), products.sum(forecast.T[None])[0].T, products, 14, twin.TwinConfig(supplier_lead_sd_days=2.0))[0]
    assert shaky > pooled  # lead-time variability adds buffer


def test_inventory_and_holding_are_valued_at_cost():
    rng = np.random.default_rng(9)
    forecast = np.full((8, 40), 3.0)
    demand = rng.poisson(3.0, (2, 8, 40)).astype(np.float32)
    args = (demand, forecast, np.full(8, 2.0), np.full(8, 3), np.full(8, 4.0), twin.TwinConfig(warmup_days=7))
    retail = twin.simulate(*args)["kpis"]
    at_cost = twin.simulate(*args, unit_cost=np.full(8, 3.0), holding_rate=np.full(8, 0.01))["kpis"]
    np.testing.assert_allclose(at_cost["inventory_value"], retail["inventory_value"] * 0.75, rtol=1e-5)
    np.testing.assert_allclose(at_cost["holding_cost"], at_cost["inventory_value"] * 0.01 * 33 / 7, rtol=1e-5)
    np.testing.assert_allclose(at_cost["lost_margin"], at_cost["lost_sales_value"] * 0.25, rtol=1e-5)


def test_one_late_shipment_hurts_less_than_a_lasting_delay_and_replanning_helps():
    rng = np.random.default_rng(10)
    n, days = 40, 84
    forecast = np.repeat(rng.uniform(1, 6, (n, 1)), days, axis=1)
    demand = rng.poisson(forecast, (6, n, days)).astype(np.float32)
    cfg = twin.TwinConfig(presentation_min=0, warmup_days=14, supplier_lead_sd_days=0.0)
    lost = lambda shock: _run(demand, forecast, 3.0, cfg=cfg, shock=shock)["kpis"]["lost_sales_value"].mean()
    base = lost(None)
    once = lost(twin.make_shock({"delay": 7, "delay_days": [0, 7]}, np.full(n, "X"), 14))
    lasting = lost(twin.make_shock({"delay": 7}, np.full(n, "X"), 14))
    replanned = lost(twin.make_shock({"delay": 7, "replan_after": 7}, np.full(n, "X"), 14))
    assert base <= once < lasting
    assert replanned < lasting


def test_current_practice_carries_the_same_safety_stock():
    rng = np.random.default_rng(11)
    forecast = np.full((20, 50), 4.0)
    demand = rng.poisson(4.0, (3, 20, 50)).astype(np.float32)
    lean = _run(demand, forecast, 0.0, policy="last_week_reorder")["kpis"]
    buffered = _run(demand, forecast, 6.0, policy="last_week_reorder")["kpis"]
    assert buffered["inventory_value"].mean() > lean["inventory_value"].mean()
    assert buffered["fill_rate"].mean() > lean["fill_rate"].mean()


def test_simulate_bundle_runs_every_store_together():
    bundle = _bundle(n=40)
    bundle["store_ids"] = np.repeat(np.array(["CA_1", "CA_2"]), 20)
    bundle["item_ids"] = np.tile(np.array([f"I{i}" for i in range(20)]), 2)
    bundle["cost"] = np.full(40, 0.7)
    result = twin.simulate_bundle(bundle, {"reps": 6, "seed": 2, "scenario": {"delay": 7, "replan_after": 7}})
    assert result["stores"] == ["CA_1", "CA_2"]
    assert set(result["baseline"]["kpis"]) == {"all", "CA_1", "CA_2"}
    assert result["baseline"]["timeline"]["group"][0].startswith("CA_1|")
    assert result["scenario"]["kpis"]["all"]["lost_sales_value"]["mean"] >= result["baseline"]["kpis"]["all"]["lost_sales_value"]["mean"]


def test_each_response_cuts_the_loss_and_its_premium_is_charged():
    rng = np.random.default_rng(12)
    n, days = 40, 84
    forecast = np.repeat(rng.uniform(1, 6, (n, 1)), days, axis=1)
    demand = rng.poisson(forecast, (6, n, days)).astype(np.float32)
    products = np.arange(n) % 20  # two stores per product, sharing the DC
    cfg = twin.TwinConfig(presentation_min=0, warmup_days=14, supplier_lead_sd_days=0.0)
    run = lambda spec: twin.simulate(demand, forecast, np.full(n, 3.0), np.full(n, 3), np.full(n, 2.0), cfg, products=products, unit_cost=np.full(n, 1.5),
                                     shock=twin.make_shock(spec, np.full(n, "X"), 14))["kpis"]
    delay = {"delay": 7, "replan_after": 7}
    shocked = run(delay)
    for response in ({"replan_after": 0}, {"prebuild_days": 7}, {"expedite_share": 0.5, "premium": 0.2}):
        assert run({**delay, **response})["lost_sales_value"].mean() < shocked["lost_sales_value"].mean(), response
    assert run({**delay, "expedite_share": 0.5, "premium": 0.2})["response_cost"].mean() > 0
    assert (shocked["response_cost"] == 0).all()
    prebuilt = run({**delay, "prebuild_days": 7})
    assert prebuilt["dc_holding_cost"].mean() > shocked["dc_holding_cost"].mean()  # pre-building is paid for in holding
    cut = {"dc_factor": 0.5, "dc_days": [0, 70]}
    backed = run({**cut, "backup_share": 0.5, "premium": 0.1})
    assert backed["lost_sales_value"].mean() < run(cut)["lost_sales_value"].mean() and backed["response_cost"].mean() > 0


def test_a_planned_spike_loses_less_than_a_surprise():
    rng = np.random.default_rng(13)
    n, days = 30, 56
    forecast = np.repeat(rng.uniform(2, 6, (n, 1)), days, axis=1)
    demand = rng.poisson(forecast, (6, n, days)).astype(np.float32)
    cfg = twin.TwinConfig(presentation_min=0, warmup_days=14)
    spike = {"demand_scale": 1.6, "days": [7, 14]}
    lost = lambda spec: _run(demand, forecast, 2.0, cfg=cfg, shock=twin.make_shock(spec, np.full(n, "X"), 14))["kpis"]["lost_sales_value"].mean()
    assert lost({**spike, "planned": True}) < lost(spike)


def test_dc_kpis_report_stock_orders_and_fill():
    rng = np.random.default_rng(14)
    n, days = 20, 56
    forecast = np.full((n, days), 3.0)
    demand = rng.poisson(3.0, (4, n, days)).astype(np.float32)
    args = (demand, forecast, np.full(n, 2.0), np.full(n, 3), np.ones(n), twin.TwinConfig(warmup_days=14))
    normal = twin.simulate(*args, products=np.arange(n) % 10)
    starved = twin.simulate(*args, products=np.arange(n) % 10, shock=twin.Shock(dc_supply_factor=0.3, dc_days=(14, 56)))
    k = normal["kpis"]
    assert (k["dc_inventory_value"] > 0).all() and (k["dc_on_order_value"] > 0).all()
    assert k["dc_fill_rate"].mean() > starved["kpis"]["dc_fill_rate"].mean()
    assert 0 < starved["kpis"]["dc_fill_rate"].mean() <= 1
    assert len(normal["dc_timeline"]["on_hand"]) == days - 14
    assert normal["dc_by_product"]["value"].sum() == pytest.approx(k["dc_inventory_value"].mean(), rel=1e-6)


def test_prebuild_runs_down_after_its_stop_day_and_expedite_only_bridges_the_gap():
    rng = np.random.default_rng(15)
    n, days = 40, 84
    forecast = np.repeat(rng.uniform(1, 6, (n, 1)), days, axis=1)
    demand = rng.poisson(forecast, (6, n, days)).astype(np.float32)
    cfg = twin.TwinConfig(presentation_min=0, warmup_days=14, supplier_lead_sd_days=0.0)
    run = lambda spec: twin.simulate(demand, forecast, np.full(n, 3.0), np.full(n, 3), np.full(n, 2.0), cfg, products=np.arange(n) % 20, unit_cost=np.full(n, 1.5),
                                     shock=twin.make_shock(spec, np.full(n, "X"), 14))["kpis"]
    late = {"delay": 7, "delay_days": [0, 7]}
    kept = run({**late, "prebuild_days": 7})
    dropped = run({**late, "prebuild_days": 7, "prebuild_until": 14})
    assert dropped["dc_holding_cost"].mean() < kept["dc_holding_cost"].mean()
    # Expediting the whole gap costs far less than the premium on every late unit, yet still cuts the loss.
    bridged = run({**late, "expedite_share": 1.0, "premium": 0.2})
    assert bridged["lost_sales_value"].mean() < run(late)["lost_sales_value"].mean()
    every_unit = twin.simulate(demand, forecast, np.full(n, 3.0), np.full(n, 3), np.full(n, 2.0), cfg, products=np.arange(n) % 20, unit_cost=np.full(n, 1.5))["flow"]["ordered"]
    assert bridged["response_cost"].mean() < 0.2 * 1.5 * every_unit / 6
