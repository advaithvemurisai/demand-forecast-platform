"""M5 ingestion, long-format preparation, and hierarchy aggregation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_FILES = ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")
ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "d", "wm_yr_wk", "weekday", "wday", "month", "year", "event_name_1",
    "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
]
LEVEL_COLUMNS = {
    "item": ["item_id", "store_id"],
    "department": ["dept_id", "store_id"],
    "category": ["cat_id", "store_id"],
    "store": ["store_id"],
    "state": ["state_id"],
    "total": [],
}


def load_m5(raw_dir: str | Path, sales_file: str = "sales_train_validation.csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three required Kaggle M5 files from ``raw_dir``."""
    raw_dir = Path(raw_dir)
    required = (sales_file, *RAW_FILES[1:])
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing M5 files in {raw_dir}: {', '.join(missing)}. "
            "Run `kaggle competitions download -c m5-forecasting-accuracy` and extract them."
        )
    sales = pd.read_csv(raw_dir / sales_file, dtype={column: "category" for column in ID_COLUMNS})
    calendar = normalise_calendar(pd.read_csv(raw_dir / "calendar.csv"))
    prices = pd.read_csv(
        raw_dir / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    return sales, calendar, prices


def normalise_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Ensure the ``d`` day key exists. Some public M5 mirrors drop it; in the official
    file ``d_1`` is the first calendar date (2011-01-29) and days are consecutive."""
    if "d" in calendar:
        return calendar
    calendar = calendar.sort_values("date").reset_index(drop=True)
    calendar.insert(0, "d", [f"d_{index + 1}" for index in range(len(calendar))])
    return calendar


def reshape_sales(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Convert M5 daily columns to one row per SKU/store/date."""
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    long = sales.melt(id_vars=ID_COLUMNS, value_vars=day_columns, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype("int16")
    date_map = calendar[["d", "date"]].copy()
    date_map["date"] = pd.to_datetime(date_map["date"])
    return long.merge(date_map, on="d", how="left", validate="many_to_one")


def prepare_m5(
    raw_dir: str | Path,
    states: Iterable[str] | None = None,
    sales_file: str = "sales_train_validation.csv",
) -> pd.DataFrame:
    """Create the silver fact table with calendar and weekly price attributes.

    ``states`` restricts the melt to a subset of states, which keeps peak memory
    manageable on a laptop (the full M5 long table is ~58M rows).
    """
    sales, calendar, prices = load_m5(raw_dir, sales_file)
    if states is not None:
        states = list(states)
        sales = sales[sales["state_id"].isin(states)]
        prices = prices[prices["store_id"].astype(str).str[:2].isin(states)]
    for column in ID_COLUMNS:
        sales[column] = sales[column].cat.remove_unused_categories()
    long = reshape_sales(sales, calendar)
    available = [column for column in CALENDAR_COLUMNS if column in calendar.columns]
    cal = calendar[available].copy()
    for column in ("event_name_1", "event_type_1", "event_name_2", "event_type_2", "weekday"):
        if column in cal:
            cal[column] = cal[column].astype("category")
    result = long.merge(cal, on="d", how="left", validate="many_to_one")
    result = result.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one",
    )
    for column in ("store_id", "item_id"):
        result[column] = result[column].astype(pd.CategoricalDtype(sales[column].cat.categories))
    # Promo proxy: the item is selling below its own median shelf price at that store.
    median_price = result.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform("median")
    result["promo"] = (result["sell_price"] < median_price).astype("int8")
    result["d"] = result["d"].astype("category")
    return result.sort_values(["state_id", "store_id", "item_id", "date"]).reset_index(drop=True)


def aggregate_bottom_up(
    bottom: pd.DataFrame,
    level: str,
    value_column: str = "sales",
    date_column: str = "date",
) -> pd.DataFrame:
    """Aggregate bottom-level observations to any explicit hierarchy level."""
    if level not in LEVEL_COLUMNS:
        raise ValueError(f"Unknown level {level!r}; expected one of {tuple(LEVEL_COLUMNS)}")
    group_columns = [date_column, *LEVEL_COLUMNS[level]]
    result = bottom.groupby(group_columns, as_index=False, observed=True)[value_column].sum()
    result["level"] = level
    result["node_id"] = make_node_ids(level, result)
    return result


def make_node_ids(level: str, frame: pd.DataFrame) -> pd.Series:
    """Vectorised stable IDs such as ``category:cat_id=FOODS|store_id=CA_1``."""
    columns = LEVEL_COLUMNS[level]
    if not columns:
        return pd.Series(f"{level}:total", index=frame.index)
    ids = pd.Series(f"{level}:", index=frame.index)
    for position, column in enumerate(columns):
        separator = "|" if position else ""
        ids = ids + f"{separator}{column}=" + frame[column].astype(str)
    return ids


def make_node_id(level: str, row: pd.Series) -> str:
    """Create a single stable ID from one row (kept for ad-hoc lookups)."""
    parts = [f"{column}={row[column]}" for column in LEVEL_COLUMNS[level]]
    return f"{level}:" + ("|".join(parts) if parts else "total")


def build_hierarchy(bottom: pd.DataFrame, value_column: str = "sales") -> pd.DataFrame:
    """Return explicit observations at all six requested hierarchy levels."""
    frames = [aggregate_bottom_up(bottom, level, value_column) for level in LEVEL_COLUMNS]
    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate_to_node(
    bottom: pd.DataFrame,
    level: str,
    node_id: str,
    value_column: str = "sales",
) -> pd.DataFrame:
    """Aggregate bottom-level data to one node, retaining the time grain."""
    if level not in LEVEL_COLUMNS:
        raise ValueError(f"Unknown level {level!r}; expected one of {tuple(LEVEL_COLUMNS)}")
    mask = pd.Series(True, index=bottom.index)
    if node_id != f"{level}:total":
        for part in node_id.split(":", 1)[1].split("|"):
            column, value = part.split("=", 1)
            mask &= bottom[column].astype(str) == value
    return aggregate_bottom_up(bottom.loc[mask], level, value_column)


def write_silver_partitioned(fact: pd.DataFrame, output_dir: str | Path) -> None:
    """Write the prepared fact table to Parquet, partitioned by state.

    Existing partitions for the states in ``fact`` are replaced, so reruns are idempotent.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for state in fact["state_id"].astype(str).unique():
        shutil.rmtree(output_dir / f"state_id={state}", ignore_errors=True)
    fact.to_parquet(output_dir, partition_cols=["state_id"], index=False)


def read_silver(
    silver_dir: str | Path,
    states: Iterable[str] | None = None,
    columns: list[str] | None = None,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read the silver fact table, optionally only for selected states and dates >= ``start``."""
    filters = []
    if states is not None:
        filters.append(("state_id", "in", list(states)))
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start)))
    frame = pd.read_parquet(silver_dir, filters=filters or None, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        if column in frame:
            frame[column] = frame[column].astype(str).astype("category")
    return frame
