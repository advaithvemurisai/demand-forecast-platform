"""Build the static data the web app loads: JSON tables, per-store twin inputs, and precomputed what-if presets.

    python scripts/build_web_data.py            # reads data/dashboard, writes web/public/{data,py}

The browser never sees parquet: every view reads small JSON files, split so a view fetches only what it draws.
`twin.py` is copied verbatim into web/public/py so the browser runs the same simulator as the pipeline and tests.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting import twin  # noqa: E402

SCHEMA_VERSION = 1
PRESETS = {
    "event_spike": {"label": "Holiday-style spike", "detail": "FOODS demand +50% for one week", "scenario": {"demand_scale": 1.5, "category": "FOODS", "days": [7, 14]}},
    "supplier_delay": {"label": "Supplier delay", "detail": "Supplier lead time 7 → 14 days", "scenario": {"delay": 7}},
    "dc_cut": {"label": "Warehouse shortfall", "detail": "DC receives 30% less stock for two weeks", "scenario": {"dc_factor": 0.7, "dc_days": [7, 21]}},
}
POLICIES = ("forecast_reorder", "last_week_reorder")


def clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 5)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [{key: clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def write_json(path: Path, payload) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    path.write_text(text)
    return len(text)


def read(source: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    path = source / f"{name}.parquet"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} missing; run `python -m forecasting.pipeline` first")
        return None
    return pd.read_parquet(path)


def columnar(frame: pd.DataFrame, id_column: str, value_columns: list[str], date_column: str = "date") -> dict:
    """One file per group: ids once, dates once, then a (ids x days) matrix per value (compact and easy to chart)."""
    dates = sorted(frame[date_column].unique())
    ids = sorted(frame[id_column].unique())
    wide = {column: frame.pivot(index=id_column, columns=date_column, values=column).reindex(index=ids, columns=dates) for column in value_columns}
    return {"ids": ids, "dates": [str(pd.Timestamp(d).date()) for d in dates], **{column: np.round(matrix.to_numpy(dtype=float), 2).tolist() for column, matrix in wide.items()}}


def build(source: Path, out: Path, web: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    data = out / "data"
    if data.exists():
        shutil.rmtree(data)
    summary = json.loads((source / "run_summary.json").read_text())
    sizes["summary.json"] = write_json(data / "summary.json", summary)

    for name in ("reconciliation_metrics", "model_metrics", "interval_coverage", "allocation", "allocation_backtest", "drift"):
        sizes[f"{name}.json"] = write_json(data / f"{name}.json", records(read(source, name)))
    for name in ("interval_coverage_segment", "allocation_node_fill", "decision_accuracy", "weekday_bias", "event_accuracy", "bias_exceptions",
                 "override_fva", "probable_stockouts", "twin_validation", "twin_timeline", "twin_frontier", "twin_stress", "twin_exceptions"):
        frame = read(source, name, required=False)
        if frame is not None:
            sizes[f"{name}.json"] = write_json(data / f"{name}.json", records(frame))

    safety = read(source, "safety_stock")
    keep = [c for c in ("item_id", "store_id", "dept_id", "cat_id", "forecast", "safety_stock", "order_up_to", "stockout_risk", "expected_fill_rate", "service_level", "abc", "xyz") if c in safety]
    safety = safety[keep].copy()
    for column in safety.select_dtypes("float").columns:
        safety[column] = safety[column].round(3)
    sizes["safety_stock.json"] = write_json(data / "safety_stock.json", records(safety.sort_values("forecast", ascending=False)))
    products = safety.drop_duplicates("item_id")["cat_id"].value_counts().to_dict() if "cat_id" in safety else {}

    production = read(source, "production_forecast")
    backtest = read(source, "backtest_forecasts")
    levels = {}
    for level in ("total", "state", "store", "category", "department"):
        prod = production[production["level"] == level]
        bt = backtest[backtest["level"] == level]
        payload = {
            "nodes": sorted(prod["series_id"].unique()),
            "production": columnar(prod, "series_id", ["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]),
            "backtest": records(bt[["fold", "method", "series_id", "date", "forecast", "actual"]]),
        }
        sizes[f"forecast/{level}.json"] = write_json(data / "forecast" / f"{level}.json", payload)
        levels[level] = payload["nodes"]
    items = production[production["level"] == "item"].copy()
    parts = items["series_id"].str.extract(r"^item:item_id=(?P<item>[^|]+)\|store_id=(?P<store>.+)$")
    items["store"], items["dept"] = parts["store"], parts["item"].str.rsplit("_", n=1).str[0]
    item_files = []
    for (store, dept), group in items.groupby(["store", "dept"]):
        name = f"item_{store}_{dept}.json"
        write_json(data / "forecast" / name, columnar(group, "series_id", ["forecast", "lower_95", "upper_95"]))
        sizes[f"forecast/{name}"] = (data / "forecast" / name).stat().st_size
        item_files.append({"store": store, "dept": dept, "file": name})
    sizes["forecast/index.json"] = write_json(data / "forecast" / "index.json", {"levels": levels, "item_files": item_files})

    twin_sizes = build_twin(source, out, data)
    sizes.update(twin_sizes)
    (out / "py").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "src" / "forecasting" / "twin.py", out / "py" / "twin.py")  # the browser runs the same simulator
    manifest = {
        "schema": SCHEMA_VERSION, "served_method": summary.get("served_method"), "origins": summary.get("origins"),
        "n_series": summary.get("n_series"), "products_by_category": products, "twin": bool(twin_sizes),
        "files": sorted(sizes), "bytes": int(sum(sizes.values())),
    }
    write_json(data / "manifest.json", manifest)
    return sizes


def build_twin(source: Path, out: Path, data: Path) -> dict[str, int]:
    """Copy per-store inputs for the browser worker and precompute the preset what-ifs with the same engine."""
    inputs = source / "twin_inputs"
    if not inputs.exists():
        return {}
    sizes, stores = {}, []
    for path in sorted(inputs.glob("*.npz")):
        store = path.stem
        target = data / "twin" / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sizes[f"twin/inputs/{path.name}"] = target.stat().st_size
        with np.load(path, allow_pickle=False) as raw:
            bundle = {key: raw[key] for key in raw.files}
        bundle = {key: (value.item() if value.ndim == 0 else value) for key, value in bundle.items()}
        dates = [str(d) for d in bundle.pop("dates")]
        bundle.pop("item_ids", None)
        presets = {"dates": dates, "results": {}}
        for policy in POLICIES:
            presets["results"][f"baseline|{policy}"] = twin.simulate_bundle(bundle, {"policy": policy, "reps": 50, "seed": 1})
            for key, preset in PRESETS.items():
                presets["results"][f"{key}|{policy}"] = twin.simulate_bundle(bundle, {"policy": policy, "reps": 50, "seed": 1, "scenario": preset["scenario"]})
        sizes[f"twin/presets/{store}.json"] = write_json(data / "twin" / "presets" / f"{store}.json", presets)
        stores.append(store)
    sizes["twin/presets.json"] = write_json(data / "twin" / "presets.json", {"stores": stores, "presets": PRESETS, "service_grid": list(twin.SERVICE_GRID), "policies": list(POLICIES)})
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "dashboard")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "public")
    args = parser.parse_args()
    sizes = build(args.source, args.out, args.out)
    total = sum(sizes.values()) / 1e6
    print(f"wrote {len(sizes)} files, {total:.1f} MB uncompressed to {args.out}")


if __name__ == "__main__":
    main()
