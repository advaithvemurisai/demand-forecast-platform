import numpy as np
import pandas as pd

from forecasting.stockouts import censoring_summary, probable_stockouts


def test_zero_run_in_fast_seller_is_flagged_but_slow_seller_zeros_are_not():
    rng = np.random.default_rng(1)
    fast = rng.poisson(8, 120).astype(float)
    fast[60:65] = 0  # five dead days for a product selling ~8/day
    slow = np.zeros(120)
    slow[::15] = 1  # about 0.07/day: long zero runs are normal
    flags = probable_stockouts(np.vstack([fast, slow]))
    assert flags[0, 60:65].all()
    assert flags[0, :60].sum() == 0
    assert not flags[1].any()


def test_days_before_first_sale_are_not_flagged():
    history = np.array([[0.0] * 30 + [5.0] * 30 + [0.0] * 4 + [5.0] * 20])
    flags = probable_stockouts(history)
    assert not flags[0, :30].any()
    assert flags[0, 60:64].all()


def test_single_zero_day_is_not_flagged():
    history = np.array([[6.0] * 20 + [0.0] + [6.0] * 20])
    assert not probable_stockouts(history, min_run=2).any()


def test_censoring_summary_shares():
    history = np.array([[5.0] * 10, [5.0] * 4 + [0.0] * 6])
    flags = probable_stockouts(history)
    keys = pd.DataFrame({"store_id": ["CA_1", "CA_1"], "cat_id": ["FOODS", "FOODS"]})
    summary = censoring_summary(flags, keys, history)
    assert summary["flagged_days"].iloc[0] == flags.sum()
    assert 0 < summary["censored_share"].iloc[0] < 1


def test_slow_sellers_are_never_judged_by_the_poisson_rule():
    slow = np.zeros(200)
    slow[::40] = 1  # 0.025/day
    assert not probable_stockouts(np.array([slow]), p_threshold=0.5).any()
    assert probable_stockouts(np.array([slow]), p_threshold=0.5, min_rate=0.0).any()
