import numpy as np
import pandas as pd
import pytest

from forecasting.reconciliation import bottom_up, mint, mint_diagonal, summing_matrix, top_down


def keys():
    return pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "i1"],
        "dept_id": ["d1", "d1", "d2", "d1"],
        "cat_id": ["c1", "c1", "c1", "c1"],
        "store_id": ["s1", "s1", "s1", "s2"],
        "state_id": ["CA", "CA", "CA", "CA"],
    })


def test_mint_projection_preserves_coherent_forecasts():
    summing = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
    base = np.array([11.0, 6.0, 7.0])
    reconciled = mint(base, summing, np.eye(3))
    assert reconciled[0] == pytest.approx(reconciled[1] + reconciled[2])


def test_summing_matrix_structure():
    A, nodes = summing_matrix(keys())
    assert A.shape == (len(nodes), 4)
    assert (np.asarray(A.sum(axis=1)).ravel() >= 1).all()
    total = nodes.index[nodes["level"] == "total"][0]
    assert A[total].sum() == 4
    assert set(nodes["level"]) == {"department", "category", "store", "state", "total"}


def test_mint_diagonal_matches_dense_mint():
    rng = np.random.default_rng(1)
    A, nodes = summing_matrix(keys())
    S = np.vstack([A.toarray(), np.eye(4)])
    base = rng.normal(10, 3, size=(S.shape[0], 5))
    variances = rng.uniform(0.5, 4, size=S.shape[0])
    dense = mint(base, S, np.diag(variances))
    agg, bottom = mint_diagonal(base[: len(nodes)], base[len(nodes):], A, variances[: len(nodes)], variances[len(nodes):])
    np.testing.assert_allclose(np.vstack([agg, bottom]), dense, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(agg, A @ bottom)  # coherent


def test_mint_diagonal_is_identity_on_coherent_input():
    A, nodes = summing_matrix(keys())
    bottom = np.array([[1.0, 2], [3, 4], [5, 6], [7, 8]])
    agg, reconciled = mint_diagonal(A @ bottom, bottom, A, np.ones(len(nodes)), np.ones(4))
    np.testing.assert_allclose(reconciled, bottom)


def test_top_down_splits_total_by_proportion():
    proportions = pd.DataFrame({"node_id": ["a", "b"], "proportion": [0.4, 0.6]})
    total = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "total_forecast": [100.0, 50.0]})
    result = top_down(proportions, total)
    assert len(result) == 4
    assert result.groupby("date")["forecast"].sum().tolist() == [100.0, 50.0]
    assert result.query("node_id == 'a'")["forecast"].tolist() == [40.0, 20.0]


def test_bottom_up_totals():
    frame = keys().assign(date=pd.Timestamp("2024-01-01"), forecast=[1.0, 2.0, 3.0, 4.0])
    result = bottom_up(frame)
    assert result.query("level == 'total'")["forecast"].iloc[0] == 10
    assert result.query("level == 'store'")["forecast"].sum() == 10


def test_temporal_reconcile_moves_weekly_total_toward_the_more_reliable_forecast():
    from forecasting.reconciliation import temporal_reconcile

    daily = np.full((1, 14), 10.0)  # weekly total 70
    weekly = np.array([[84.0, 56.0]])
    near_weekly = temporal_reconcile(daily, weekly, np.array([100.0]), np.array([1e-6]))
    np.testing.assert_allclose(near_weekly.reshape(2, 7).sum(axis=1), [84.0, 56.0], rtol=1e-4)
    near_daily = temporal_reconcile(daily, weekly, np.array([1e-6]), np.array([100.0]))
    np.testing.assert_allclose(near_daily, daily, rtol=1e-3)
    balanced = temporal_reconcile(daily, weekly, np.array([10.0]), np.array([70.0])).reshape(2, 7).sum(axis=1)
    assert 70 < balanced[0] < 84 and 56 < balanced[1] < 70


def test_temporal_reconcile_validates_shape_and_stays_non_negative():
    from forecasting.reconciliation import temporal_reconcile

    with pytest.raises(ValueError):
        temporal_reconcile(np.ones((1, 10)), np.ones((1, 1)), np.ones(1), np.ones(1))
    out = temporal_reconcile(np.full((1, 7), 1.0), np.array([[-50.0]]), np.array([1.0]), np.array([1e-9]))
    assert (out >= 0).all()
