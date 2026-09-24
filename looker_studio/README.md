# Looker Studio report

This builds a free, shareable Looker Studio version of the dashboard from the CSVs in `looker_studio/data/`. Looker Studio runs in your Google account, so the steps are clicks in the browser, about 30–40 minutes the first time.

```bash
python looker_studio/build_extract.py     # regenerates looker_studio/data/*.csv from data/dashboard
```

## 1. Create the data sources

Go to [lookerstudio.google.com](https://lookerstudio.google.com), then **Create → Data source → File Upload**. Upload each CSV as its own data source. Name each source after its file, and on the field screen set the types below; everything else can stay as detected.

| File | Rows | Set these field types |
|---|---|---|
| `forecast_timeline.csv` | ~27k | `date` → Date (YYYYMMDD); `is_served` → Boolean |
| `item_forecast.csv` | ~341k | `date` → Date |
| `accuracy.csv` | 248 | `wmape`, `mape`, `bias` → Percent; `wrmsse` → Number; `is_served` → Boolean |
| `interval_coverage.csv` | 36 | `coverage`, `nominal` → Percent |
| `allocation.csv` | 28 | `stockout_risk`, `expected_fill_rate` → Percent; `unit_value` → Currency (USD) |
| `allocation_backtest.csv` | 24 | `fill_rate` → Percent; `revenue_fulfilled` → Currency (USD) |
| `drift.csv` | 4 | (none) |
| `safety_stock.csv` | 12,196 | `stockout_risk` → Percent |

Set the aggregation for every metric field to **Average**, not Sum. The metric tables hold one row per fold, so summing them would be wrong.

## 2. Report theme

In **Theme and layout → Customise**, set:
- Background `#fcfcfb`, text `#0b0b0b`, and a light grid `#e4e3df`.
- Chart palette, in this order: `#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`, `#e87ba4`. This order was checked for colour-blind separation; keep it fixed, and don't let Looker Studio cycle colours.

## 3. Pages

### Overview (scorecards)

Use four **Scorecards**, each with a text box underneath stating what it measures:

| Scorecard | Data source | Metric | Filter |
|---|---|---|---|
| Holdout WRMSSE (served) | `accuracy` | Average of `wrmsse` | `kind = Reconciliation`, `is_served = true`, `fold = holdout` |
| Category WMAPE, holdout | `accuracy` | Average of `wmape` | as above, plus `level = category` |
| 95% coverage, items | `interval_coverage` | Average of `coverage` | `level = item`, `nominal = 0.95` |
| LP revenue vs pro-rata | `allocation_backtest` | calculated field `LP uplift` (below) | none |

`LP uplift` is a calculated field in `allocation_backtest`, with type Percent:

```
SUM(CASE WHEN policy = "lp_scenario" THEN revenue_fulfilled ELSE 0 END)
/ SUM(CASE WHEN policy = "pro_rata" THEN revenue_fulfilled ELSE 0 END) - 1
```

### Forecast explorer

- **Controls:** add drop-down lists for `level` and then `node` (the second one filters on the first), and a checkbox for `is_served` so viewers can hide the other methods.
- **Time series chart** on `forecast_timeline`:
  - Dimension `date`, breakdown dimension `method_label`.
  - Metric `forecast`, plus a second series `actual` shown in black.
  - Filter `level != item`.
- **Intervals:** Looker Studio can't shade a band between two lines. Add `lower_95` and `upper_95` as extra metrics drawn as thin dashed lines in the served colour, and state in the chart title that the band is the 95% interval.
- **Item drill-down:** a second time series on `item_forecast`, with drop-down controls for `store_id`, `dept_id` and `item_id`, and metrics `forecast`, `lower_95` and `upper_95`.

### Accuracy

- **Pivot table with heatmap** on `accuracy`:
  - Row dimension `level`, sorted by `level_order` ascending. Column dimension `name`. Metric Average of `wmape`.
  - Filter `kind = Reconciliation`, and add a drop-down control on `window` (holdout or backtest).
  - Heatmap colour: a single blue, light to dark.
- **Bar chart** of base models on `accuracy`: filter `kind = Base model` and `level = item`; dimension `name`, metric Average of `wmape`, sorted ascending.
- **Text box:** "Served method is chosen on backtests only; the SARIMA base is most accurate at aggregate levels but incoherent."

### Intervals

- **Scatter chart** on `interval_coverage`:
  - Dimension `level` (sorted by `level_order`), breakdown `nominal_label`, metric Average of `coverage`.
  - Add reference lines at 0.80 and 0.95, labelled "nominal".
- **Table** underneath: `level`, `nominal_label`, Average of `coverage`, and Average of `mean_width`.

### Allocation

- **Column chart** on `allocation`: dimension `dept_id`, metrics `allocated_quantity` (label it "Scenario LP") and `pro_rata_quantity` (label it "Pro-rata"). Add a drop-down control on `store_id`.
- **Bar chart** on `allocation_backtest`: dimension `week_label`, breakdown `policy_label`, metric `revenue_fulfilled`.
- **Table** on `safety_stock`: `item_id`, `store_id`, `forecast`, `safety_stock`, `order_up_to` and `stockout_risk`, with a search control on `item_id`.

### Drift

- **Table** on `drift`: `store_id`, `status`, `demand_psi`, `residual_psi` and `retrain_reasons`. Use conditional formatting so `status = Drift` shows red with a ⚠ symbol, so the status never relies on colour alone.

## 4. Share it

Go to **Share → Manage access** and set the link to **Anyone on the internet with the link can view**, then copy the link into the project README's **Dashboards** section. Viewers don't need a Google account.

The data is uploaded CSVs, so after a pipeline rerun you refresh the report by running `build_extract.py` again and then **Resource → Manage added data sources → Edit → Replace file** for each changed file.
