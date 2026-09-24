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
