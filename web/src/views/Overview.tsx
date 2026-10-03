import { getJson, getOptional } from '../data/load'
import type {
  AllocationBacktestRow, CoverageRow, Manifest, ModelRow, ReconRow, Summary, TwinExceptionRow, TwinFrontierRow, TwinPolicyCurveRow, TwinStressRow,
} from '../data/types'
import { Async, Kpi } from '../components/ui'
import { useData } from '../components/data'
import { MODEL_LABELS, num, pct, points, signedPct, usd } from '../lib/format'
import { holdoutWmape, mean, sum, weeklyUplift } from '../lib/metrics'

const REPO_URL = 'https://github.com/advaithvemurisai/demand-forecast-platform'
const LAUNCH: { key: string; title: string }[] = [
  { key: 'supplier_delay', title: 'Supplier slows down' },
  { key: 'late_shipment', title: 'One late delivery' },
  { key: 'dc_cut', title: 'Warehouse shortfall' },
  { key: 'event_spike', title: 'Holiday-style spike' },
]

export default function Overview() {
  const state = useData(async () => {
    const [manifest, summary, recon, models, coverage, backtest, curve, frontier, exceptions, stress] = await Promise.all([
      getJson<Manifest>('manifest.json'), getJson<Summary>('summary.json'), getJson<ReconRow[]>('reconciliation_metrics.json'),
      getJson<ModelRow[]>('model_metrics.json'), getJson<CoverageRow[]>('interval_coverage.json'),
      getJson<AllocationBacktestRow[]>('allocation_backtest.json'), getOptional<TwinPolicyCurveRow[]>('twin_policy_curve.json'),
      getOptional<TwinFrontierRow[]>('twin_frontier.json'), getOptional<TwinExceptionRow[]>('twin_exceptions.json'), getOptional<TwinStressRow[]>('twin_stress.json'),
    ])
    return { manifest, summary, recon, models, coverage, backtest, curve, frontier, exceptions, stress }
  })
  return (
    <Async state={state}>
      {({ manifest, summary, recon, models, coverage, backtest, curve, frontier, exceptions, stress }) => {
        const served = summary.served_method
        const categories = manifest.products_by_category
        const products = sum(Object.values(categories))
        const servedCat = holdoutWmape(recon, served, 'category')
        const bottomUpCat = holdoutWmape(recon, 'bottom_up', 'category')
        const itemCoverage = mean(coverage.filter((row) => row.level === 'item' && row.nominal === 0.95).map((row) => row.coverage))
        const weeks = weeklyUplift(backtest)
        const lp = backtest.filter((row) => row.policy === 'lp_scenario')
        const pro = backtest.filter((row) => row.policy === 'pro_rata')
        const lostChange = sum(lp.map((r) => r.lost_revenue ?? 0)) / sum(pro.map((r) => r.lost_revenue ?? 1)) - 1
        const allocationCi = summary.allocation_vs_pro_rata_ci?.revenue_fulfilled
        const itemModels = Object.fromEntries(models.filter((m) => m.fold === 'holdout' && m.level === 'item').map((m) => [m.model, m.wmape]))
        const twin = summary.twin
        const serviceSaving = sum(Object.values(twin?.service_saving ?? {}).filter((row) => row.clear).map((row) => row.saving))
        return (
          <>
            {curve && curve.length > 0 && <Value curve={curve} summary={summary} />}

            {twin && (
              <section>
                <h2>What to do this week</h2>
                <div className="actions">
                  <ServiceAction frontier={frontier} summary={summary} />
                  {exceptions && exceptions.length > 0 && <ExpediteAction rows={exceptions} />}
                  {stress && <RiskAction rows={stress} />}
                  {allocationCi && (
                    <div className="card action">
                      <span className="kicker">If the warehouse runs short</span>
                      <p><strong>Split scarce stock by expected margin, not pro-rata.</strong> Worth about {usd(allocationCi.mean * 52)} a year across four stores
                        (95% CI {usd(allocationCi.lower * 52)} to {usd(allocationCi.upper * 52)}), {signedPct(lostChange)} lost revenue, no department starved.</p>
                      <p className="muted small"><a href="#/allocation">Next week’s split</a></p>
                    </div>
                  )}
                </div>
              </section>
            )}

            {stress && (
              <section>
                <h2>Try a scenario</h2>
                <p className="muted small">Each opens the simulator with that disruption applied to all four stores and the warehouse.</p>
                <div className="launch">
                  {LAUNCH.filter(({ key }) => stress.some((row) => row.scenario === key)).map(({ key, title }) => {
                    const row = stress.find((r) => r.scenario === key && r.policy === 'forecast_reorder' && r.metric === 'lost_sales_value' && (r.rationing ?? 'days_of_cover') === 'days_of_cover')
                    return (
                      <a key={key} className="preset" href={`#/twin?scenario=${key}`}>
                        <strong>{title}</strong>
                        <span>{row?.label.replace(/^[^:]*:\s*/, '')}</span>
                        {row && <span className="num"> · {row.delta >= 0 ? '+' : ''}{usd(row.delta)} lost sales</span>}
                      </a>
                    )
                  })}
                </div>
              </section>
            )}

            <h2>The business case, step by step</h2>
            <div className="grid kpis">
              <Kpi label="Allocation when supply is short" value={allocationCi ? `${usd(allocationCi.mean * 52)}/yr` : '–'}
                delta={`${weeks.filter((w) => w.uplift > 0).length} of ${weeks.length} test weeks won`} tone="good"
                note="More revenue than a proportional split, four stores, with supply at 90% of forecast" />
              {serviceSaving > 0 && <Kpi label="Right-sized service targets" value={`${usd(serviceSaving * 13)}/yr`} tone="good"
                note="Lower lost margin + holding cost than today’s segment targets (4-week saving × 13), where the gain is clear of simulation noise" />}
              <Kpi label="Plans that agree" value={`${pct((bottomUpCat - servedCat) / bottomUpCat, 0)} lower`} delta={`category error ${pct(servedCat)} vs ${pct(bottomUpCat)}`} tone={servedCat <= bottomUpCat ? 'good' : 'bad'}
                note="Category forecasts that add up to the item plan, vs adding up item forecasts" />
              <Kpi label="Product forecasts" value={`${pct(1 - itemModels.lgbm_global / itemModels.seasonal_naive, 0)} lower error`} tone="flat"
                note={`Than repeating last week’s sales (holdout WRMSSE ${summary.wrmsse_holdout[served].toFixed(3)} over 6 levels)`} />
              <Kpi label="Forecast ranges you can trust" value={pct(itemCoverage)} tone="flat" note="of item sales landed inside the 95% range, out of sample" />
            </div>

            <h2>What each step solves</h2>
            <div className="table-wrap stack">
              <table className="stack">
                <thead><tr><th>Step</th><th className="text">Business question</th><th className="text">Result</th></tr></thead>
                <tbody>
                  {[
                    ['Forecast (LightGBM)', 'How much will each product sell in each store over the next 28 days?', `${pct(1 - itemModels.lgbm_global / itemModels.seasonal_naive, 0)} lower product error than repeating last week’s sales (${MODEL_LABELS[summary.selected_bottom_model]} served)`],
                    ['Reconcile (MinT)', 'Do the product, store and state plans agree, so every team works from the same numbers?', `Forecasts add up at every level; category error ${pct((bottomUpCat - servedCat) / bottomUpCat, 0)} lower than adding up product forecasts`],
                    ['Intervals (conformal)', 'How sure are we, and how bad could it get?', `95% ranges contain ${pct(itemCoverage)} of actual product sales, calibrated by product speed and category`],
                    ['Safety stock', 'How much buffer does each product need over its replenishment window?', `Order-up-to levels for all ${num(manifest.n_series)} product-store pairs, with service targets by value and variability`],
                    ['Allocate (scenario LP)', 'When stock is short, which stores and departments get it?', `${allocationCi ? `${usd(allocationCi.mean * 52)} a year more revenue` : 'More revenue'} and ${signedPct(lostChange)} lost revenue vs a proportional split, counting stock already on the shelf`],
                    ['Simulate (inventory twin)', 'What do a policy, a demand spike or a supply problem do to stock, service and cash?', twin ? `Predicts the network fill rate within ${points(twin.typical_miss?.fill_rate ?? twin.fill_rate_gap).replace('+', '')} on average; its ranges are still too narrow (${pct(twin.share_realised_in_band, 0)} of network checks inside), so compare policies rather than trust absolute dollars` : 'See Stress-test'],
                    ['Monitor (drift)', 'Has demand shifted enough that the model needs retraining?', 'PSI and accuracy-decay checks, plus a check that the simulator still matches reality'],
                  ].map(([step, question, result]) => (
                    <tr key={step}>
                      <td data-label="Step"><strong>{step}</strong></td>
                      <td className="text" data-label="Business question" style={{ whiteSpace: 'normal' }}>{question}</td>
                      <td className="text" data-label="Result" style={{ whiteSpace: 'normal' }}>{result}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted small">
              Data: Walmart M5, {num(products)} products ({num(categories.FOODS ?? 0)} food, {num(categories.HOUSEHOLD ?? 0)} household, {num(categories.HOBBIES ?? 0)} hobby) in
              {' '}{Math.round(manifest.n_series / products)} California stores. Product names are anonymised, so items appear as codes like FOODS_3_090. Costs, lead times and starting
              stock are not in the data and are stated assumptions. <a href={REPO_URL}>Code and methodology</a>.
            </p>
          </>
        )
      }}
    </Async>
  )
}

/** The headline: what the calibrated safety stock buys, and what forecast-based reordering adds at equal stock. */
function Value({ curve, summary }: { curve: TwinPolicyCurveRow[]; summary: Summary }) {
  const at = (policy: string, multiplier: number) => curve.find((row) => row.policy === policy && row.safety_multiplier === multiplier)
  const bare = at('forecast_reorder', 0), standard = at('forecast_reorder', 1)
  const equal = summary.twin?.equal_inventory
  if (!bare || !standard) return null
  const marginBack = bare.lost_margin - standard.lost_margin
  const extraHolding = standard.holding_cost - bare.holding_cost
  const naiveStock = equal?.current_practice_inventory_for_same
  const stockGap = naiveStock && Number.isFinite(naiveStock) ? naiveStock - standard.inventory_value : NaN
  const fillGap = equal?.current_practice != null && Number.isFinite(equal.current_practice) ? standard.fill_rate - equal.current_practice : NaN
  return (
    <section>
      <h2 style={{ marginTop: 18 }}>Where the value is</h2>
      <p className="answer">
        Calibrated safety stock does the heavy lifting: it lifts the fill rate from {pct(bare.fill_rate)} to {pct(standard.fill_rate)} and cuts lost sales by {usd(bare.lost_sales_value - standard.lost_sales_value)} over
        four weeks, for {usd(extraHolding)} of extra holding cost. Reordering from the forecast instead of last week’s sales adds only a little on top
        {Number.isFinite(stockGap) && stockGap > 0 ? <>: the same service with {usd(stockGap)} ({pct(stockGap / naiveStock!, 0)}) less stock</> : null}.
      </p>
      <div className="grid kpis">
        <Kpi label="Fill rate" value={pct(standard.fill_rate)} delta={`${points(standard.fill_rate - bare.fill_rate)} vs no safety stock`} tone="good" note="Share of demanded units sold, replayed on the holdout weeks, all four stores" />
        <Kpi label="Lost sales, 4 weeks" value={usd(standard.lost_sales_value)} delta={`${usd(standard.lost_sales_value - bare.lost_sales_value)} vs no safety stock`} tone="good" note="Demand that found an empty shelf, at shelf price" />
        <Kpi label="Cost of the buffer" value={usd(extraHolding)} delta={`vs ${usd(marginBack)} margin recovered`} tone={marginBack > extraHolding ? 'good' : 'bad'}
          note={`Four weeks of holding cost on ${usd(standard.inventory_value - bare.inventory_value)} of extra stock (at cost; assumed rates by category)`} />
        <Kpi label="Forecast vs last week’s sales" value={Number.isFinite(stockGap) ? `${usd(stockGap)} less stock` : '–'}
          delta={Number.isFinite(fillGap) ? `${points(fillGap)} fill rate at equal stock` : undefined} tone={Number.isFinite(fillGap) && fillGap >= 0 ? 'good' : 'flat'}
          note="Both policies carry the same safety stock; the forecast mostly saves working capital" />
      </div>
      <p className="muted small">
        Replayed on what actually sold in the four holdout weeks, with the safety stock switched off and on. Costs and holding rates are assumptions.{' '}
        <a href="#/twin?section=equal-inventory">See the curves</a>.
      </p>
    </section>
  )
}

function ServiceAction({ frontier, summary }: { frontier?: TwinFrontierRow[]; summary: Summary }) {
  const saving = summary.twin?.service_saving ?? {}
  const recommended = (frontier ?? []).filter((row) => row.recommended)
  const clear = recommended.filter((row) => saving[row.category]?.clear)
  const edge = summary.twin?.service_at_grid_edge ?? []
  return (
    <div className="card action">
      <span className="kicker">Service targets</span>
      {clear.length > 0 ? (
        <p><strong>Move {clear.map((row) => `${titleCase(row.category)} to ${row.label}`).join(', ')}.</strong>{' '}
          Saves {usd(sum(clear.map((row) => saving[row.category].saving)))} of lost margin + holding cost over four weeks
          {clear.length === 1 ? ` (90% range ${usd(saving[clear[0].category].lower)} to ${usd(saving[clear[0].category].upper)})` : ''}.</p>
      ) : (
        <p><strong>Keep today’s targets.</strong> No category’s cheapest target beats them by more than simulation noise.</p>
      )}
      {recommended.filter((row) => !saving[row.category]?.clear).length > 0 && clear.length > 0 && (
        <p className="muted small">{recommended.filter((row) => !saving[row.category]?.clear).map((row) => titleCase(row.category)).join(', ')}: no clear gain, keep as is.</p>
      )}
      {edge.length > 0 && <p className="muted small">{edge.map(titleCase).join(', ')} sits at the top of the tested range, so an even higher target may be cheaper still.</p>}
      <p className="muted small"><a href="#/twin">Cost curves</a></p>
    </div>
  )
}

function ExpediteAction({ rows }: { rows: TwinExceptionRow[] }) {
  const top = rows.slice(0, 5)
  return (
    <div className="card action">
      <span className="kicker">Expedite</span>
      <p><strong>{top.length} product-store pairs carry {usd(sum(top.map((r) => r.expected_lost_value)))} of sales at risk this week.</strong>{' '}
        {top.slice(0, 3).map((r) => `${r.item_id} at ${r.store_id}`).join(', ')}…</p>
      <p className="muted small"><a href="#/twin?section=exceptions">Full watch-list</a> ({rows.length} products, forecast-based reordering)</p>
    </div>
  )
}

function RiskAction({ rows }: { rows: TwinStressRow[] }) {
  const lost = rows.filter((r) => r.policy === 'forecast_reorder' && r.metric === 'lost_sales_value' && (r.rationing ?? 'days_of_cover') === 'days_of_cover')
  const worst = [...lost].sort((a, b) => b.delta - a.delta)[0]
  if (!worst) return null
  return (
    <div className="card action">
      <span className="kicker">Biggest risk</span>
      <p><strong>{worst.label}:</strong> about {usd(worst.delta)} more lost sales over four weeks
        {Number.isFinite(worst.delta_lower) ? ` (90% of futures ${usd(worst.delta_lower)} to ${usd(worst.delta_upper)})` : ''}.</p>
      <p className="muted small"><a href={`#/twin?scenario=${worst.scenario}`}>Run it in the simulator</a></p>
    </div>
  )
}

const titleCase = (value: string) => value.charAt(0) + value.slice(1).toLowerCase()
