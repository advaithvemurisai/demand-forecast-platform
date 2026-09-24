import pandas as pd

from forecasting.data import aggregate_to_node, normalise_calendar, build_hierarchy, make_node_id, prepare_m5, read_silver, write_silver_partitioned


def bottom_frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "item_id": ["i1", "i2", "i1", "i2"],
            "dept_id": ["d1"] * 4,
            "cat_id": ["c1"] * 4,
            "store_id": ["s1", "s1", "s2", "s2"],
            "state_id": ["CA", "CA", "TX", "TX"],
            "sales": [2, 3, 5, 7],
        }
    )


def test_hierarchy_has_explicit_levels_and_totals():
    hierarchy = build_hierarchy(bottom_frame())
    assert set(hierarchy["level"]) == {"item", "department", "category", "store", "state", "total"}
    total = hierarchy.query("level == 'total'")["sales"].iloc[0]
    assert total == 17
    assert hierarchy.query("level == 'store'")["sales"].sum() == total


def test_aggregate_to_node_selects_exact_node():
    node = aggregate_to_node(bottom_frame(), "state", "state:state_id=CA").iloc[0]
    assert node["sales"] == 5
    assert aggregate_to_node(bottom_frame(), "total", "total:total").iloc[0]["sales"] == 17


def test_vectorised_node_ids_match_row_ids():
    hierarchy = build_hierarchy(bottom_frame())
    for row in hierarchy.itertuples(index=False):
        assert row.node_id == make_node_id(row.level, pd.Series(row._asdict()))


def test_prepare_m5_and_idempotent_silver(synthetic_root):
    fact = prepare_m5(synthetic_root / "data" / "raw", states=["CA"], sales_file="sales_train_evaluation.csv")
    assert len(fact) == 12 * 900
    assert fact["sell_price"].notna().all() and set(fact["promo"].unique()) <= {0, 1}
    silver = synthetic_root / "data" / "silver"
    write_silver_partitioned(fact, silver)
    write_silver_partitioned(fact, silver)  # rerun must replace, not duplicate
    assert len(read_silver(silver, ["CA"])) == len(fact)
    assert read_silver(silver, ["CA"], start=pd.Timestamp("2016-01-01"))["date"].min() >= pd.Timestamp("2016-01-01")


def test_calendar_without_day_key_is_rebuilt():
    calendar = pd.DataFrame({"date": ["2011-01-30", "2011-01-29"], "wm_yr_wk": [11101, 11101]})
    result = normalise_calendar(calendar)
    assert result.set_index("date")["d"].to_dict() == {"2011-01-29": "d_1", "2011-01-30": "d_2"}
