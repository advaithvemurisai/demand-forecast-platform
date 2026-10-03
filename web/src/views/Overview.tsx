import { getJson, getOptional } from '../data/load'
import type { AllocationBacktestRow, CoverageRow, Manifest, ModelRow, ReconRow, Summary, TwinValidationRow } from '../data/types'
import { Async, Kpi } from '../components/ui'
import { useData } from '../components/data'
import { MODEL_LABELS, num, pct, points, signedPct, usd } from '../lib/format'
import { holdoutWmape, mean, sum, twinHeadline, weeklyUplift } from '../lib/metrics'

const REPO_URL = 'https://github.com/advaithvemurisai/demand-forecast-platform'

export default function Overview() {
  const state = useData(async () => {
    const [manifest, summary, recon, models, coverage, backtest, validation] = await Promise.all([
      getJson<Manifest>('manifest.json'), getJson<Summary>('summary.json'), getJson<ReconRow[]>('reconciliation_metrics.json'),
      getJson<ModelRow[]>('model_metrics.json'), getJson<CoverageRow[]>('interval_coverage.json'),
      getJson<AllocationBacktestRow[]>('allocation_backtest.json'), getOptional<TwinValidationRow[]>('twin_validation.json'),
    ])
    return { manifest, summary, recon, models, coverage, backtest, validation }
  })
  return (
    <Async state={state}>
      {({ manifest, summary, recon, models, coverage, backtest, validation }) => {
        const served = summary.served_method
        const categories = manifest.products_by_category
        const products = sum(Object.values(categories))
        const servedCat = holdoutWmape(recon, served, 'category')
        const bottomUpCat = holdoutWmape(recon, 'bottom_up', 'category')
        const itemCoverage = mean(coverage.filter((row) => row.level === 'item' && row.nominal === 0.95).map((row) => row.coverage))
        const weeks = weeklyUplift(backtest)
        const lp = backtest.filter((row) => row.policy === 'lp_scenario')
        const pro = backtest.filter((row) => row.policy === 'pro_rata')
        const revenueGain = sum(lp.map((r) => r.revenue_fulfilled)) / sum(pro.map((r) => r.revenue_fulfilled)) - 1
        const lostChange = sum(lp.map((r) => r.lost_revenue ?? 0)) / sum(pro.map((r) => r.lost_revenue ?? 1)) - 1
        const itemModels = Object.fromEntries(models.filter((m) => m.fold === 'holdout' && m.level === 'item').map((m) => [m.model, m.wmape]))
        const head = validation ? twinHeadline(validation, 'forecast_reorder') : undefined
        const naive = validation && head ? twinHeadline(validation, 'last_week_reorder', head.fold) : undefined
        const m = (name: string) => head?.metrics.get(name)
        const n = (name: string) => naive?.metrics.get(name)
        return (
          <>
            <h2 style={{ marginTop: 18 }}>The problem</h2>
            <p>
              A retailer has to decide how much of each product to stock and which stores get it. Forecasts made separately for items, stores and
              the whole region disagree, and when the warehouse is short someone has to choose which stores go without. Too little stock loses sales;
              too much ties up cash. This platform turns Walmart store sales into forecasts that add up at every level, a confidence range for each,
              a weekly allocation for scarce supply, and an <strong>inventory digital twin</strong> that tests ordering policies before anyone commits to them.
            </p>
            <p>
              <strong>What’s forecast.</strong> Daily unit sales of {num(products)} Walmart products ({num(categories.FOODS ?? 0)} food, {num(categories.HOUSEHOLD ?? 0)} household
              and {num(categories.HOBBIES ?? 0)} hobby items) in each of {Math.round(manifest.n_series / products)} California stores, 28 days ahead, rolled up to departments,
              categories, stores and the state. Product names are anonymised in the public data, so items appear as codes like FOODS_3_090.{' '}
              <a href={REPO_URL}>Code and methodology</a>.
            </p>

            {head && m('in_stock_pct') && (
              <>
                <h2>Store operations: forecast-based reordering vs current practice</h2>
                <p className="muted small">
                  Inventory twin, {head.fold === 'holdout' ? 'holdout' : (head.fold ?? '').replace('_', ' ')} window (28 days, all four stores). “Current practice” reorders from last week’s sales.
                </p>
                <div className="grid kpis">
                  <Kpi label="In-stock rate" value={pct(m('in_stock_pct')!.realised)}
                    delta={n('in_stock_pct') ? `${points(m('in_stock_pct')!.realised - n('in_stock_pct')!.realised)} vs current practice` : undefined}
                    tone={n('in_stock_pct') && m('in_stock_pct')!.realised >= n('in_stock_pct')!.realised ? 'good' : 'bad'} note="Share of product-days with stock on the shelf" />
                  <Kpi label="Fill rate" value={pct(m('fill_rate')!.realised)}
                    delta={n('fill_rate') ? `${points(m('fill_rate')!.realised - n('fill_rate')!.realised)} vs current practice` : undefined}
                    tone={n('fill_rate') && m('fill_rate')!.realised >= n('fill_rate')!.realised ? 'good' : 'bad'} note="Share of demanded units sold" />
                  <Kpi label="Lost sales" value={usd(m('lost_sales_value')!.realised)}
                    delta={n('lost_sales_value') ? `${usd(m('lost_sales_value')!.realised - n('lost_sales_value')!.realised)} vs current practice` : undefined}
                    tone={n('lost_sales_value') && m('lost_sales_value')!.realised <= n('lost_sales_value')!.realised ? 'good' : 'bad'} note="Demand that found an empty shelf, at shelf price" />
                  <Kpi label="Inventory on hand" value={usd(m('inventory_value')!.realised)}
                    delta={n('inventory_value') ? `${usd(m('inventory_value')!.realised - n('inventory_value')!.realised)} vs current practice` : undefined}
                    note="Average stock value: the working capital the policy ties up" />
                </div>
              </>
            )}

            <h2>Model health</h2>
            <div className="grid kpis">
              <Kpi label="Holdout WRMSSE (served method)" value={summary.wrmsse_holdout[served].toFixed(3)} note="Below 1 beats repeating last period’s sales. Over 6 CA levels, so not comparable to the 12-level M5 leaderboard." />
              <Kpi label="Category WMAPE, holdout" value={pct(servedCat)} delta={`${points(servedCat - bottomUpCat)} vs bottom-up`} tone={servedCat <= bottomUpCat ? 'good' : 'bad'} note="Average % miss on category sales per store, where buyers plan" />
              <Kpi label="95% interval coverage, items" value={pct(itemCoverage)} note="How often actual sales landed inside the range. Out-of-sample." />
              <Kpi label="Allocation revenue vs pro-rata" value={signedPct(revenueGain)}
                delta={`${weeks.filter((w) => w.uplift > 0).length} of ${weeks.length} weeks won`} tone={revenueGain >= 0 ? 'good' : 'bad'}
                note={summary.allocation_vs_pro_rata_ci?.revenue_fulfilled ? `95% CI on weekly gain: ${usd(summary.allocation_vs_pro_rata_ci.revenue_fulfilled.lower)} to ${usd(summary.allocation_vs_pro_rata_ci.revenue_fulfilled.upper)}` : undefined} />
            </div>

            <h2>What each step solves for the business</h2>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Step</th><th className="text">Business question</th><th className="text">Result</th></tr></thead>
                <tbody>
                  {[
                    ['Forecast (LightGBM)', 'How much will each item sell in each store over the next 28 days?', `${pct(1 - itemModels.lgbm_global / itemModels.seasonal_naive, 0)} lower item error than repeating last week’s sales (${MODEL_LABELS[summary.selected_bottom_model]} served)`],
                    ['Reconcile (MinT)', 'Do the item, store and state plans agree, so every team works from the same numbers?', `Forecasts add up at every level; category error ${pct((bottomUpCat - servedCat) / bottomUpCat, 0)} lower than adding up item forecasts`],
                    ['Intervals (conformal)', 'How sure are we, and how bad could it get?', `95% ranges contain ${pct(itemCoverage)} of actual item sales, calibrated by product speed and category`],
                    ['Safety stock', 'How much buffer does each item need over its replenishment window?', `Order-up-to levels for all ${num(manifest.n_series)} items, with service targets by value and variability class`],
                    ['Allocate (scenario LP)', 'When stock is short, which stores and departments get it?', `${signedPct(revenueGain)} revenue and ${signedPct(lostChange)} lost revenue vs a proportional split, with a minimum fill for every department`],
                    ['Simulate (inventory twin)', 'What happens to stock, service and cash under a policy, a demand spike or a supply problem?', head ? `Predicted service levels land within about a point of realised; its 90% bands are too narrow (${pct(head.inBand / Math.max(head.checks, 1), 0)} of checks inside), so dollar losses are best read as policy-vs-policy` : 'See the Inventory twin view'],
                    ['Monitor (drift)', 'Has demand shifted enough that the model needs retraining?', 'PSI and accuracy decay checks, plus a twin-divergence check'],
                  ].map(([step, question, result]) => (
                    <tr key={step}><td><strong>{step}</strong></td><td className="text" style={{ whiteSpace: 'normal' }}>{question}</td><td className="text" style={{ whiteSpace: 'normal' }}>{result}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )
      }}
    </Async>
  )
}
