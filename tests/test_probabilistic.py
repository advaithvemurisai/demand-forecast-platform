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
