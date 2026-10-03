import numpy as np
import pytest

from forecasting import probabilistic as prob


def test_safety_stock_increases_with_service_level():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 2000)
    point = np.array([10.0, 20.0])
    low = prob.service_level_safety_stock(point, scores, 0.80)
    high = prob.service_level_safety_stock(point, scores, 0.99)
    assert (high["safety_stock"] > low["safety_stock"]).all()
    assert high["stockout_risk"].max() < low["stockout_risk"].min()
    assert low["stockout_risk"].iloc[0] == pytest.approx(0.2, abs=0.02)


def test_stockout_risk_varies_by_row():
    scores = np.linspace(-3, 3, 601)
    risk = prob.stockout_risk([10, 10], [10, 13], scores)
    assert risk[0] == pytest.approx(0.5, abs=0.01) and risk[1] == pytest.approx(0.0, abs=0.01)


def test_conformal_interval_achieves_nominal_coverage():
    rng = np.random.default_rng(1)
    calibration = rng.standard_t(4, 5000)
    test_errors = rng.standard_t(4, 5000)
    point = np.full(5000, 50.0)
    interval = prob.conformal_interval(point, calibration, level=0.9)
    coverage = prob.empirical_coverage(point + test_errors, interval["lower"], interval["upper"])
    assert coverage == pytest.approx(0.9, abs=0.02)


def test_scaled_interval_width_follows_scale():
    interval = prob.conformal_interval([10.0, 1000.0], np.linspace(-1, 1, 101), 0.9, scale=[1.0, 100.0])
    widths = (interval["upper"] - interval["lower"]).to_numpy()
    assert widths[1] == pytest.approx(100 * widths[0])


def test_invalid_levels():
    with pytest.raises(ValueError):
        prob.service_level_safety_stock([1.0], [0.0], 1.0)
    with pytest.raises(ValueError):
        prob.conformal_interval([1.0], [0.0], 0)


def test_mondrian_interval_fixes_per_segment_coverage():
    rng = np.random.default_rng(0)
    n = 20_000
    groups = np.where(rng.random(n) < 0.5, "fast", "sporadic")
    noise = np.where(groups == "fast", 0.5, 3.0)  # same scale, very different error spread
    actual_scores = rng.normal(0, noise)
    point = np.full(n, 10.0)
    test_groups = np.where(rng.random(n) < 0.5, "fast", "sporadic")
    test_noise = np.where(test_groups == "fast", 0.5, 3.0)
    actual = point + rng.normal(0, test_noise)
    pooled = prob.conformal_interval(point, actual_scores, 0.9)
    mondrian = prob.mondrian_interval(point, actual_scores, groups, test_groups, 0.9)
    for segment in ("fast", "sporadic"):
        rows = test_groups == segment
        pooled_cov = prob.empirical_coverage(actual[rows], pooled["lower"][rows], pooled["upper"][rows])
        mondrian_cov = prob.empirical_coverage(actual[rows], mondrian["lower"][rows], mondrian["upper"][rows])
        assert abs(mondrian_cov - 0.9) < 0.02
        assert abs(mondrian_cov - 0.9) <= abs(pooled_cov - 0.9) + 1e-9


def test_mondrian_small_segment_falls_back_to_pooled_scores():
    scores = np.linspace(-1, 1, 1000)
    result = prob.mondrian_interval([5.0], scores, np.array(["a"] * 1000), ["rare"], 0.9, min_count=500)
    pooled = prob.conformal_interval([5.0], scores, 0.9)
    assert result["upper"].iloc[0] == pytest.approx(pooled["upper"].iloc[0])


def test_velocity_class_thresholds():
    assert prob.velocity_class([3.0, 1.0, 0.2, 0.01]).tolist() == ["fast", "medium", "slow", "sporadic"]


def test_abc_classes_follow_revenue_share():
    revenue = np.array([80.0, 10.0, 5.0, 3.0, 2.0])
    assert prob.abc_class(revenue).tolist() == ["A", "B", "B", "C", "C"]


def test_xyz_classes_follow_variability():
    weekly = np.array([[10, 10, 11, 9], [1, 20, 2, 30], [0, 0, 0, 40], [0, 0, 0, 0]], dtype=float)
    assert prob.xyz_class(weekly).tolist() == ["X", "Y", "Z", "Z"]


def test_service_targets_protect_staples_more_than_long_tail():
    targets = prob.service_targets(["A", "C"], ["X", "Z"])
    assert targets[0] > targets[1]


def test_round_to_case_never_under_orders():
    quantity = np.array([0.0, 0.2, 6.0, 6.1, 13.0])
    rounded = prob.round_to_case(quantity, 6)
    assert (rounded >= quantity - 1e-9).all()
    assert rounded.tolist() == [0.0, 6.0, 6.0, 12.0, 18.0]


def test_expected_fill_rate_rises_with_stock_and_is_bounded():
    rng = np.random.default_rng(2)
    scores = rng.normal(0, 3, 5000)
    fills = prob.expected_fill_rate([20.0] * 3, [10.0, 20.0, 40.0], scores)
    assert fills[0] < fills[1] < fills[2] <= 1.0
