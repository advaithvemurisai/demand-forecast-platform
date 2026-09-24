import numpy as np
import pandas as pd
import pytest

from forecasting.allocation import allocate_inventory, proportional_allocation


def test_allocation_respects_inventory_constraint():
    demand = pd.DataFrame({"store_id": ["A", "B", "C"], "forecast": [8, 7, 6]})
    result = allocate_inventory(demand, inventory=10)
    assert result["allocated_quantity"].sum() <= 10 + 1e-8
    assert (result["allocated_quantity"] <= result["forecast"] + 1e-8).all()


def test_allocation_prefers_higher_value_units():
    demand = pd.DataFrame({"forecast": [10.0, 10.0], "unit_value": [1.0, 5.0]})
    result = allocate_inventory(demand, inventory=10, value_column="unit_value")
    assert result["allocated_quantity"].tolist() == pytest.approx([0.0, 10.0])


def test_scenario_allocation_hedges_uncertain_demand():
    # Same mean demand; B is volatile. Expected fulfilment rewards stocking B above its median.
    demand = pd.DataFrame({"forecast": [10.0, 10.0]})
    scenarios = np.array([[10.0] * 4, [2.0, 6.0, 14.0, 18.0]])
    result = allocate_inventory(demand, inventory=24, scenarios=scenarios)
    assert result["allocated_quantity"].sum() == pytest.approx(24)
    assert result.loc[0, "allocated_quantity"] == pytest.approx(10)
    assert result.loc[0, "stockout_risk"] == 0
    assert 0 < result.loc[1, "stockout_risk"] < 1
    assert result["expected_fill_rate"].between(0, 1).all()


def test_scenario_lp_beats_pro_rata_in_expectation():
    rng = np.random.default_rng(3)
    demand = pd.DataFrame({"forecast": [20.0, 20.0, 20.0], "unit_value": [1.0, 2.0, 3.0]})
    scenarios = np.clip(demand["forecast"].to_numpy()[:, None] + rng.normal(0, [[2], [8], [15]], (3, 200)), 0, None)
    lp = allocate_inventory(demand, 50, scenarios, value_column="unit_value")["allocated_quantity"].to_numpy()
    pro_rata = proportional_allocation(demand, 50)
    value = lambda x: (np.minimum(scenarios, x[:, None]).mean(axis=1) * demand["unit_value"].to_numpy()).sum()
    assert value(lp) >= value(pro_rata)


def test_min_fill_floor_and_infeasible():
    demand = pd.DataFrame({"forecast": [10.0, 10.0], "unit_value": [1.0, 5.0]})
    result = allocate_inventory(demand, 10, value_column="unit_value", min_fill=0.3)
    assert result.loc[0, "allocated_quantity"] >= 3 - 1e-8
    with pytest.raises(ValueError):
        allocate_inventory(demand, 5, min_fill=0.5)
