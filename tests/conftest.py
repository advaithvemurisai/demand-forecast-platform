import numpy as np
import pandas as pd
import pytest


def write_synthetic_m5(raw_dir, n_days=900, horizon=28, seed=0):
    """Tiny M5-shaped raw files: 2 CA stores, 2 categories, 6 items, weekly seasonality."""
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2014-01-01", periods=n_days + horizon, freq="D")
    calendar = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": 11101 + np.arange(len(dates)) // 7,
        "weekday": dates.day_name(),
        "wday": dates.dayofweek + 1,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i + 1}" for i in range(len(dates))],
        "event_name_1": np.where(dates.strftime("%m-%d") == "12-25", "Christmas", None),
        "event_type_1": np.where(dates.strftime("%m-%d") == "12-25", "National", None),
        "event_name_2": None, "event_type_2": None,
        "snap_CA": (dates.day <= 10).astype(int), "snap_TX": 0, "snap_WI": 0,
    })
    items = [("FOODS_1_001", "FOODS_1", "FOODS"), ("FOODS_1_002", "FOODS_1", "FOODS"), ("FOODS_1_003", "FOODS_1", "FOODS"),
             ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"), ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES"), ("HOBBIES_1_003", "HOBBIES_1", "HOBBIES")]
    rows, prices = [], []
    weekly = np.array([1.0, 0.9, 0.9, 1.0, 1.2, 1.5, 1.4])
    for store in ("CA_1", "CA_2"):
        for position, (item, dept, cat) in enumerate(items):
            level = 2 + position + (store == "CA_2")
            mean = level * weekly[dates[:n_days].dayofweek]
            rows.append({"id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": dept, "cat_id": cat, "store_id": store, "state_id": "CA",
                         **{f"d_{i + 1}": int(v) for i, v in enumerate(rng.poisson(mean))}})
            for week in calendar["wm_yr_wk"].unique():
                prices.append({"store_id": store, "item_id": item, "wm_yr_wk": week, "sell_price": round(3 + position + 0.5 * rng.random(), 2)})
    pd.DataFrame(rows).to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw_dir / "calendar.csv", index=False)
    pd.DataFrame(prices).to_csv(raw_dir / "sell_prices.csv", index=False)
    return raw_dir


@pytest.fixture
def synthetic_root(tmp_path):
    write_synthetic_m5(tmp_path / "data" / "raw")
    return tmp_path
