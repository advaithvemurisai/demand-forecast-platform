import json

import numpy as np
import pandas as pd

from forecasting.pipeline import Config, naive_scale_sq, run


def test_pipeline_end_to_end_on_synthetic_m5(synthetic_root):
    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=20, prophet=False, mlflow=False)
    summary = run(cfg)
    gold = synthetic_root / "data" / "gold"
    for name in ("forecasts", "prediction_intervals", "safety_stock", "allocation", "allocation_backtest",
                 "model_metrics", "reconciliation_metrics", "interval_coverage", "backtest_forecasts", "drift"):
        assert (gold / f"{name}.parquet").exists() and (gold / f"{name}.csv").exists()
    assert json.loads((gold / "run_summary.json").read_text())["n_series"] == 12

    forecasts = pd.read_parquet(gold / "forecasts.parquet")
    assert forecasts["date"].min() > pd.Timestamp("2014-01-01") + pd.Timedelta(days=899)  # strictly after the last actual
    assert forecasts.loc[forecasts["served"], "method"].nunique() == 1
    assert summary["served_method"] in {"bottom_up", "top_down", "mint_diagonal", "mint_oos"}
    for method in ("bottom_up", "mint_diagonal", "mint_oos", "top_down"):
        day = forecasts[(forecasts["method"] == method) & (forecasts["date"] == forecasts["date"].min())]
        total = day.loc[day["level"] == "total", "forecast"].iloc[0]
        assert np.isclose(day.loc[day["level"] == "item", "forecast"].sum(), total)  # coherent
    assert (forecasts["forecast"] >= 0).all()

    intervals = pd.read_parquet(gold / "prediction_intervals.parquet")
    assert (intervals["lower_95"] <= intervals["lower_80"] + 1e-9).all() and (intervals["upper_95"] >= intervals["upper_80"] - 1e-9).all()
    allocation = pd.read_parquet(gold / "allocation.parquet")
    assert allocation["allocated_quantity"].sum() <= allocation["supply"].iloc[0] + 1e-6

    recon = pd.read_parquet(gold / "reconciliation_metrics.parquet")
    assert recon["wrmsse"].notna().all() and (recon["wrmsse"] > 0).all()
    assert not recon.loc[recon["fold"] == "backtest_1", "oos_weights"].any() and recon.loc[recon["fold"] == "holdout", "oos_weights"].all()
    assert set(recon["fold"]) == {"backtest_1", "backtest_2", "holdout"}
    assert summary["selected_bottom_model"] in {"naive", "seasonal_naive", "lgbm_global", "lgbm_local"}
    coverage = pd.read_parquet(gold / "interval_coverage.parquet")
    assert set(coverage["fold"]) == {"backtest_2", "holdout"}  # first fold has no earlier calibration data


def test_pipeline_logs_to_mlflow(synthetic_root):
    import mlflow

    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=True, experiment="test")
    run(cfg)
    runs = mlflow.search_runs(experiment_names=["test"])
    parent = runs[runs["tags.mlflow.runName"] == "pipeline-CA"]
    assert len(parent) == 1 and "metrics.wmape.mint_diagonal.total" in runs.columns
    assert len(runs) > 1  # nested per-fold runs


def test_naive_scale_ignores_leading_zeros():
    history = np.array([[0.0, 0.0, 0.0, 2.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # diffs counted from the first sale onward: (4-2)^2, (2-4)^2 -> mean 4
    np.testing.assert_allclose(naive_scale_sq(history), [4.0, 0.0])


def test_pipeline_writes_dashboard_extract(synthetic_root):
    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False)
    run(cfg)
    extract = synthetic_root / "data" / "dashboard"
    for name in ("production_forecast", "reconciliation_metrics", "backtest_forecasts", "allocation", "drift", "safety_stock"):
        assert (extract / f"{name}.parquet").exists()
    assert json.loads((extract / "run_summary.json").read_text())["served_method"]


def test_safety_stock_has_segments_fill_rate_and_shelf_minimum(synthetic_root):
    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False, presentation_min=2.0, case_pack=3)
    run(cfg)
    safety = pd.read_parquet(synthetic_root / "data" / "gold" / "safety_stock.parquet")
    assert {"abc", "xyz", "protection_days", "expected_fill_rate", "service_level"} <= set(safety.columns)
    assert (safety["order_up_to"] >= cfg.presentation_min - 1e-9).all()
    assert np.allclose(safety["order_up_to"] % 3, 0) or np.allclose((safety["order_up_to"] % 3).round(6) % 3, 0)
    assert safety["expected_fill_rate"].between(0, 1).all()
    foods = safety[safety["cat_id"] == "FOODS"]["protection_days"].unique().tolist()
    hobbies = safety[safety["cat_id"] == "HOBBIES"]["protection_days"].unique().tolist()
    assert foods == [cfg.lead_time_days + cfg.review_days_by_category["FOODS"]]
    assert hobbies == [cfg.lead_time_days + cfg.review_days_by_category["HOBBIES"]]
    gold = synthetic_root / "data" / "gold"
    assert (gold / "interval_coverage_segment.parquet").exists() and (gold / "probable_stockouts.parquet").exists()
    assert (gold / "allocation_node_fill.parquet").exists()
    backtest = pd.read_parquet(gold / "allocation_backtest.parquet")
    assert {"units_lost", "lost_revenue", "min_node_fill"} <= set(backtest.columns)


def test_pipeline_writes_decision_level_tables(synthetic_root):
    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False)
    summary = run(cfg)
    gold = synthetic_root / "data" / "gold"
    decision = pd.read_parquet(gold / "decision_accuracy.parquet")
    assert set(decision["window_days"]) <= {3, 8} and decision["wmape"].notna().any()
    weekday = pd.read_parquet(gold / "weekday_bias.parquet")
    assert set(weekday["weekday"]) == {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
    event = pd.read_parquet(gold / "event_accuracy.parquet")
    assert {"event", "normal"} >= set(event["day_type"]) and {"item_wmape", "total_bias"} <= set(event.columns)
    exceptions = pd.read_parquet(gold / "bias_exceptions.parquet")
    assert {"tracking_signal", "direction", "weekly_impact"} <= set(exceptions.columns)
    ci = summary["allocation_vs_pro_rata_ci"]["revenue_fulfilled"]
    assert ci["lower"] <= ci["mean"] <= ci["upper"] and ci["n"] > 0


def test_pipeline_twin_phase_writes_validated_tables(synthetic_root):
    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False, twin_reps=8)
    summary = run(cfg)
    gold = synthetic_root / "data" / "gold"
    validation = pd.read_parquet(gold / "twin_validation.parquet")
    assert set(validation["policy"]) == {"forecast_reorder", "last_week_reorder"} and "all" in set(validation["store_id"])
    assert (validation["lower"] <= validation["upper"] + 1e-9).all()
    assert set(validation["fold"]) == {"backtest_2", "holdout"}  # backtest_1 has no earlier calibration
    frontier = pd.read_parquet(gold / "twin_frontier.parquet")
    assert set(frontier["assumption"]) == {"every_unit", "bias_corrected", "shopper_response"}
    assert frontier.groupby(["assumption", "category"])["recommended"].sum().eq(1).all()
    assert frontier.groupby(["assumption", "category"])["within_budget"].sum().le(1).all()
    assert (frontier.loc[frontier["within_budget"], "inventory_change"] <= 1e-9).all()
    grid = frontier[frontier["service_level"].notna()].sort_values(["category", "service_level"])
    for _, group in grid.groupby(["assumption", "category"]):
        assert group["inventory_value"].is_monotonic_increasing
    # Pricing fewer stockouts as lost can only make the cheapest target lower or equal.
    best = frontier[frontier["recommended"]].pivot(index="category", columns="assumption", values="service_level")
    assert (best["shopper_response"] <= best["every_unit"] + 1e-9).all()
    stress = pd.read_parquet(gold / "twin_stress.parquet")
    lost = stress[(stress["metric"] == "lost_sales_value") & (stress["policy"] == "forecast_reorder") & (stress["rationing"] == "days_of_cover")]
    assert (lost["delta"] >= -1e-6).all()  # a shock never reduces lost sales
    assert {"event_spike", "late_shipment", "supplier_delay", "dc_cut"} <= set(lost["scenario"])
    assert set(stress.loc[stress["scenario"] == "dc_cut", "rationing"]) == {"proportional", "days_of_cover", "value"}
    curve = pd.read_parquet(gold / "twin_policy_curve.parquet")
    for _, group in curve.groupby("policy"):
        assert group.sort_values("safety_multiplier")["inventory_value"].is_monotonic_increasing
    assert {"saving_vs_current", "saving_lower", "saving_upper", "clear_saving", "at_grid_edge"} <= set(frontier.columns)
    responses = pd.read_parquet(gold / "twin_responses.parquet")
    assert set(responses["scenario"]) == {"event_spike", "late_shipment", "supplier_delay", "dc_cut"}
    none = responses[responses["response"] == "none"]
    assert (none["response_cost"].abs() < 1e-6).all() and (none["net_benefit"].abs() < 1e-6).all()
    speed = pd.read_parquet(gold / "twin_frontier_speed.parquet")
    assert set(speed["category"]) <= {"fast", "medium", "slow", "sporadic"} and speed.groupby(["assumption", "category"])["recommended"].sum().eq(1).all()
    health = pd.read_parquet(gold / "inventory_health.parquet")
    assert {"all", "FOODS"} <= set(health["group"]) and (health.loc[health["group_type"] == "category", "weeks_of_supply"] > 0).all()
    assert health["lost_share"].between(0, 1).all()
    allocation = pd.read_parquet(gold / "allocation.parquet")
    assert (allocation["on_hand_source"] == "twin replay of the holdout window").all() and (allocation["on_hand"] >= 0).all()
    exceptions = pd.read_parquet(gold / "twin_exceptions.parquet")
    assert exceptions["expected_lost_value"].is_monotonic_decreasing
    assert not pd.read_parquet(gold / "twin_timeline.parquet").empty
    assert {"fill_rate", "recommended_service", "share_realised_in_band", "typical_miss", "equal_inventory", "holdout_tradeoff", "service_saving", "service_by_assumption", "lost_sales_bias"} <= set(summary["twin"])
    assert 0 <= summary["twin"]["typical_miss"]["fill_rate"] <= 1


def test_replay_scores_weekly_cycle_and_stability(synthetic_root):
    from forecasting.replay import run_replay

    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False)
    table = run_replay(cfg, weeks=3)
    assert table["week"].tolist() == [1, 2, 3]
    assert table["stability_vs_previous_run"].iloc[0] != table["stability_vs_previous_run"].iloc[0]  # first run has no predecessor (NaN)
    assert (table["stability_vs_previous_run"].iloc[1:] >= 0).all()
    assert table["next_week_wmape"].notna().all()


def test_override_fva_and_temporal_options_run(synthetic_root):
    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False, twin=False, temporal=True)  # temporal is opt-in
    run(cfg)
    gold = synthetic_root / "data" / "gold"
    fva = pd.read_parquet(gold / "override_fva.parquet")
    assert set(fva["scope"]) == {"flagged products", "all products"} and {"fva", "n"} <= set(fva.columns)
    models = pd.read_parquet(gold / "model_metrics.parquet")
    assert {"sarima", "sarima_daily_only"} <= set(models["model"])


def test_twin_bundles_load_without_pickle(synthetic_root):
    """The API and the browser (Pyodide) load these with allow_pickle=False, so no object arrays are allowed."""
    import numpy as np
    from forecasting import twin

    cfg = Config(root=synthetic_root, history_days=200, n_backtest_folds=2, n_estimators=10, prophet=False, mlflow=False, twin_reps=6)
    run(cfg)
    files = sorted((synthetic_root / "data" / "dashboard" / "twin_inputs").glob("*.npz"))
    assert [path.name for path in files] == ["network.npz"]  # one bundle: every store shares the DC
    with np.load(files[0], allow_pickle=False) as data:
        bundle = {key: data[key] for key in data.files}
    assert {"past_actual", "past_forecast", "forecast", "safety_by_service", "current_safety", "dept_ids", "store_ids", "item_ids", "cost", "holding_rate"} <= set(bundle)
    assert bundle["safety_by_service"].shape[1] == len(twin.SERVICE_GRID)
    assert (bundle["cost"] < bundle["price"]).all()
    bundle = {key: (value.item() if value.ndim == 0 else value) for key, value in bundle.items()}
    bundle.pop("dates")
    result = twin.simulate_bundle(bundle, {"reps": 4, "seed": 1, "scenario": {"demand_scale": 1.4, "days": [3, 10]}})
    assert result["scenario"]["kpis"]["all"]["lost_sales_value"]["mean"] >= result["baseline"]["kpis"]["all"]["lost_sales_value"]["mean"] - 1e-9
    assert set(result["baseline"]["kpis"]) == {"all", *result["stores"]} and len(result["stores"]) > 1
