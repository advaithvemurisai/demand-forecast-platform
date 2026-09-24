"""Flatten the dashboard extract into CSVs shaped for Looker Studio file upload.

Looker Studio works best with one tidy table per chart family and plain types
(YYYY-MM-DD dates, ratios for percentages). Writes looker_studio/data/*.csv:

    python looker_studio/build_extract.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "dashboard"
OUT = Path(__file__).resolve().parent / "data"
LEVEL_ORDER = {"total": 1, "state": 2, "store": 3, "category": 4, "department": 5, "item": 6}
METHOD_LABELS = {
    "base": "SARIMA base (incoherent)", "bottom_up": "Bottom-up", "top_down": "Top-down",
    "mint_diagonal": "MinT, in-sample weights", "mint_oos": "MinT, out-of-sample weights",
}
MODEL_LABELS = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "lgbm_local": "LightGBM local",
                "lgbm_global": "LightGBM global", "sarima": "SARIMA", "prophet": "Prophet"}


def node_label(series_id: pd.Series) -> pd.Series:
    rest = series_id.str.split(":", n=1).str[1]
    return rest.where(rest != "total", "All CA stores").str.replace("_id=", ": ", regex=False).str.replace("|", " · ", regex=False)


def main() -> None:
    import json

    OUT.mkdir(parents=True, exist_ok=True)
    served = json.loads((SOURCE / "run_summary.json").read_text())["served_method"]
    read = lambda name: pd.read_parquet(SOURCE / f"{name}.parquet")

    # Aggregate nodes: backtest/holdout forecasts vs actuals, then the production forecast with intervals.
    backtest = read("backtest_forecasts")
    backtest["window"] = backtest["fold"].where(backtest["fold"] == "holdout", "backtest")
    production = read("production_forecast")
    aggregate = production[production["level"] != "item"].assign(fold="production", window="production", method=served)
    timeline = pd.concat([backtest, aggregate], ignore_index=True)
    timeline["method_label"] = timeline["method"].map(METHOD_LABELS)
    timeline["is_served"] = timeline["method"] == served
    timeline["node"] = node_label(timeline["series_id"])
    timeline["level_order"] = timeline["level"].map(LEVEL_ORDER)
    columns = ["date", "window", "fold", "level", "level_order", "node", "series_id", "method", "method_label", "is_served",
               "actual", "forecast", "lower_80", "upper_80", "lower_95", "upper_95"]
    timeline[columns].to_csv(OUT / "forecast_timeline.csv", index=False, date_format="%Y-%m-%d")

    items = production[production["level"] == "item"].copy()
    parts = items["series_id"].str.extract(r"item_id=(?P<item_id>[^|]+)\|store_id=(?P<store_id>.+)")
    items = pd.concat([items, parts], axis=1)
    items["dept_id"] = items["item_id"].str.rsplit("_", n=1).str[0]
    items["cat_id"] = items["item_id"].str.split("_").str[0]
    items[["date", "item_id", "dept_id", "cat_id", "store_id", "forecast", "lower_80", "upper_80", "lower_95", "upper_95"]].to_csv(
        OUT / "item_forecast.csv", index=False, date_format="%Y-%m-%d")

    recon = read("reconciliation_metrics").assign(kind="Reconciliation", name=lambda f: f["method"].map(METHOD_LABELS))
    recon["is_served"] = recon["method"] == served
    models = read("model_metrics").assign(kind="Base model", name=lambda f: f["model"].map(MODEL_LABELS), is_served=False)
    accuracy = pd.concat([recon, models], ignore_index=True)
    accuracy["window"] = accuracy["fold"].where(accuracy["fold"] == "holdout", "backtest")
    accuracy["level_order"] = accuracy["level"].map(LEVEL_ORDER)
    accuracy[["kind", "name", "is_served", "fold", "window", "level", "level_order", "wmape", "wrmsse", "mape", "bias"]].to_csv(
        OUT / "accuracy.csv", index=False)

    coverage = read("interval_coverage")
    coverage["level_order"] = coverage["level"].map(LEVEL_ORDER)
    coverage["nominal_label"] = (coverage["nominal"] * 100).round().astype(int).astype(str) + "% interval"
    coverage.to_csv(OUT / "interval_coverage.csv", index=False)

    read("allocation").to_csv(OUT / "allocation.csv", index=False, date_format="%Y-%m-%d")
    weekly = read("allocation_backtest")
    weekly["policy_label"] = weekly["policy"].map({"lp_scenario": "Scenario LP", "pro_rata": "Pro-rata"})
    weekly["week_label"] = weekly["fold"] + " · week " + weekly["week"].astype(str)
    weekly.to_csv(OUT / "allocation_backtest.csv", index=False)
    drift = read("drift")
    drift["store_id"] = drift["node_id"].str.replace("store:store_id=", "", regex=False)
    drift["status"] = drift["drift"].map({True: "Drift", False: "Stable"})
    drift.to_csv(OUT / "drift.csv", index=False)
    read("safety_stock").to_csv(OUT / "safety_stock.csv", index=False, date_format="%Y-%m-%d")

    for path in sorted(OUT.glob("*.csv")):
        print(f"{path.name:28s} {path.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
