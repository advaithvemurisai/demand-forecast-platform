"""Replay the weekly planning cycle: re-forecast every week and score accuracy and stability.

The main pipeline scores four 28-day windows. A planning team re-runs the forecast every week, so
this replays that cadence (opt-in, one model fit per week): week-ahead accuracy at the decision
grain, and how much the forecast for the same target days moved from one weekly run to the next.

    python -m forecasting.replay --weeks 8
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import decision_metrics as dm
from forecasting import probabilistic as prob
from forecasting.export import export_gold
from forecasting.pipeline import Config, fit_bottom_models, load_frame, protection_days, to_matrix

log = logging.getLogger("forecasting.replay")


def run_replay(cfg: Config, weeks: int = 8, model: str = "lgbm_global") -> pd.DataFrame:
    features, keys, origins = load_frame(cfg)
    n = len(keys)
    item_days = protection_days(keys, cfg)
    rows, previous = [], None
    for week in range(weeks):
        origin = origins["holdout"] - pd.Timedelta(days=7 * (weeks - 1 - week))
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")
        forecasts, _ = fit_bottom_models(features, origin, cfg, dates, n, [model])
        forecast = forecasts[model]
        actual = np.nan_to_num(to_matrix(features[features["date"].isin(dates)], "sales", dates, n))
        history_dates = pd.date_range(origin - pd.Timedelta(days=55), origin, freq="D")
        history = np.nan_to_num(to_matrix(features[features["date"].isin(history_dates)], "sales", history_dates, n))
        velocity = prob.velocity_class(history.mean(axis=1)).astype(str)
        decision = dm.decision_accuracy(actual, forecast, keys, item_days, velocity, f"week_{week + 1}")
        week_wmape, week_bias = dm.wmape_bias(dm.window_sums(actual, 7)[:, :1], dm.window_sums(forecast, 7)[:, :1])
        total_wmape, total_bias = dm.wmape_bias(dm.window_sums(actual, 7), dm.window_sums(forecast, 7))
        rows.append({
            "week": week + 1, "origin": origin.date().isoformat(), "model": model,
            "next_week_wmape": week_wmape, "next_week_bias": week_bias,
            "four_week_wmape": total_wmape, "four_week_bias": total_bias,
            "decision_window_wmape": float(np.average(decision["wmape"].fillna(0), weights=decision["units"].clip(lower=1e-9))),
            "stability_vs_previous_run": dm.forecast_stability(previous[:, 7:], forecast[:, : cfg.horizon - 7]) if previous is not None else np.nan,
        })
        previous = forecast
        log.info("Replay week %s/%s origin %s next-week WMAPE %.3f", week + 1, weeks, origin.date(), week_wmape)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay the weekly planning cycle (one model fit per week).")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--states", nargs="+", default=["CA"])
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config(root=Path(__file__).resolve().parents[2], states=tuple(args.states), n_estimators=args.estimators, max_items=args.max_items, prophet=False, mlflow=False)
    table = run_replay(cfg, args.weeks)
    export_gold({"planning_cycle": table}, cfg.gold_dir)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
