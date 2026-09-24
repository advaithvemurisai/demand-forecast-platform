# Tableau layer

Tableau reads the gold tables in `data/gold/` directly (CSV for Tableau Public, Parquet for Desktop 2023.2+). Nothing here depends on the modelling code. Rerunning `python -m forecasting.pipeline` refreshes every table in place, so an extract refresh picks up the new run.

## Data sources

| Table | Grain | Use it for |
|---|---|---|
| `forecasts` | node × date × method | Production forecast (28 days past the last actual), every hierarchy level, all five methods; `served = true` marks the method selected on backtests |
| `prediction_intervals` | node × date | Served forecast with conformal 80% / 95% bands |
| `backtest_forecasts` | fold × method × aggregate node × date | Forecast vs actual lines for the backtest and holdout windows |
| `reconciliation_metrics` | fold × method × level | WMAPE / WRMSSE / MAPE / pinball / bias by reconciliation method |
| `model_metrics` | fold × model × level | Naive, seasonal naive, LightGBM global/local (bottom-up), SARIMA, Prophet |
| `interval_coverage` | fold × level × nominal | Empirical vs nominal coverage, mean interval width |
| `safety_stock` | item × store (week 1) | Order-up-to level at the configured service level, stockout risk |
| `allocation` | store × department (week 1) | LP allocation vs pro-rata, expected fill rate, stockout risk |
| `allocation_backtest` | fold × week × policy | Realised units / revenue fulfilled: LP vs pro-rata |
| `drift` | store | Demand-mix PSI, residual PSI, retraining flag and reasons |

`series_id` / `node_id` values are hierarchy keys such as `item:item_id=FOODS_3_090|store_id=CA_1`, `department:dept_id=FOODS_3|store_id=CA_1`, `store:store_id=CA_1`, `state:state_id=CA`, and `total:total`. Relate `prediction_intervals` to `forecasts` on `series_id` + `date` (filter `forecasts.served = true`).

## Suggested dashboard

1. **Hierarchy explorer**: a line chart of `backtest_forecasts` (actual vs `forecast`), with the level and node as filters and `method` on colour. Then continue into `prediction_intervals` for the production window, with `lower_95`/`upper_95` as a band.
2. **Reconciliation scorecard**: a highlight table of mean `wrmsse` (or `wmape`) by `level` (rows) × `method` (columns) from `reconciliation_metrics`, filtered to `fold = holdout`.
3. **Model comparison**: a bar chart of `wmape` by `model` at `level = item` from `model_metrics`.
4. **Calibration**: a dot plot of `coverage` against `nominal` by `level` from `interval_coverage`, with a reference line at `nominal`.
5. **Allocation**: store × department bars of `allocated_quantity` vs `pro_rata_quantity` from `allocation`, with `stockout_risk` as a tooltip. Add a KPI tile for `allocation_backtest` revenue uplift.
6. **Drift monitor**: a store table from `drift`, where a red mark means `retrain = true`.

Useful calculated fields:

```
// Level (from series_id)
SPLIT([series_id], ":", 1)

// Absolute percentage error per row (backtest_forecasts)
ABS([actual] - [forecast]) / NULLIF([actual], 0)

// LP revenue uplift vs pro-rata (allocation_backtest)
(SUM(IIF([policy] = "lp_scenario", [revenue_fulfilled], 0))
 / SUM(IIF([policy] = "pro_rata", [revenue_fulfilled], 0))) - 1
```
