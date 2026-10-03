"""Export gold outputs for Tableau or DuckDB ingestion."""
from pathlib import Path

import pandas as pd


def export_gold(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write each table as Parquet (typed, compact) and CSV (Tableau Public friendly)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)


# Written by newer pipeline stages; copied when present so the web app can show them.
OPTIONAL_TABLES = (
    "interval_coverage_segment", "allocation_node_fill", "decision_accuracy", "weekday_bias", "event_accuracy", "bias_exceptions",
    "override_fva", "probable_stockouts", "twin_validation", "twin_timeline", "twin_frontier", "twin_policy_curve", "twin_stress", "twin_exceptions", "planning_cycle",
)
DASHBOARD_TABLES = (
    "model_metrics", "reconciliation_metrics", "interval_coverage", "allocation",
    "allocation_backtest", "drift", "backtest_forecasts",
)


def export_dashboard_extract(gold_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Compact copy of the gold layer for the web app, the API and Looker Studio.

    Keeps every small table as-is, the served production forecast with intervals
    (all nodes), and the item-level safety stock, all small enough to version in git
    so the hosted app works without rerunning the pipeline.
    """
    import json

    gold_dir, output_dir = Path(gold_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name in DASHBOARD_TABLES:
        frame = pd.read_parquet(gold_dir / f"{name}.parquet")
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        sizes[name] = len(frame)
    for name in OPTIONAL_TABLES:
        path = gold_dir / f"{name}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
            sizes[name] = len(frame)
    intervals = pd.read_parquet(gold_dir / "prediction_intervals.parquet").drop(columns=["method"], errors="ignore")
    float_columns = intervals.select_dtypes("float").columns
    intervals[float_columns] = intervals[float_columns].astype("float32").round(3)
    intervals.to_parquet(output_dir / "production_forecast.parquet", index=False)
    sizes["production_forecast"] = len(intervals)
    safety = pd.read_parquet(gold_dir / "safety_stock.parquet")
    safety.to_parquet(output_dir / "safety_stock.parquet", index=False)
    sizes["safety_stock"] = len(safety)
    summary = json.loads((gold_dir / "run_summary.json").read_text())
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return sizes
