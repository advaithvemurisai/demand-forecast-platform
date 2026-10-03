import { getJson, getOptional } from '../data/load'
import type {
  AllocationBacktestRow, CoverageRow, InventoryHealthRow, Manifest, ModelRow, ReconRow, Summary, TwinExceptionRow, TwinPolicyCurveRow, TwinResponseRow, TwinStressRow,
} from '../data/types'
import { Async, CsvButton, Kpi } from '../components/ui'
import { useData } from '../components/data'
import { MODEL_LABELS, num, pct, points, signedPct, signedUsd, usd } from '../lib/format'
import { bestResponse, holdoutWmape, mean, sum, weeklyUplift } from '../lib/metrics'

const MATERIAL_USD = 50 // a watch-list row matters if this much is at risk in a week, or a stockout is more likely than not

const REPO_URL = 'https://github.com/advaithvemurisai/demand-forecast-platform'
const LAUNCH: { key: string; title: string }[] = [
  { key: 'supplier_delay', title: 'Supplier slows down' },
  { key: 'late_shipment', title: 'One late delivery' },
  { key: 'dc_cut', title: 'Warehouse shortfall' },
  { key: 'event_spike', title: 'Holiday-style spike' },
]

export default function Overview() {
  const state = useData(async () => {
    const [manifest, summary, recon, models, coverage, backtest, curve, exceptions, stress, responses, health] = await Promise.all([
      getJson<Manifest>('manifest.json'), getJson<Summary>('summary.json'), getJson<ReconRow[]>('reconciliation_metrics.json'),
      getJson<ModelRow[]>('model_metrics.json'), getJson<CoverageRow[]>('interval_coverage.json'),
      getJson<AllocationBacktestRow[]>('allocation_backtest.json'), getOptional<TwinPolicyCurveRow[]>('twin_policy_curve.json'),
      getOptional<TwinExceptionRow[]>('twin_exceptions.json'), getOptional<TwinStressRow[]>('twin_stress.json'),
      getOptional<TwinResponseRow[]>('twin_responses.json'), getOptional<InventoryHealthRow[]>('inventory_health.json'),
    ])
    return { manifest, summary, recon, models, coverage, backtest, curve, exceptions, stress, responses, health }
  })
  return (
    <Async state={state}>
      {({ manifest, summary, recon, models, coverage, backtest, curve, exceptions, stress, responses, health }) => {
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
                  <ServiceAction summary={summary} />
                  {stress && <RiskAction rows={stress} responses={responses} />}
                  {allocationCi && (
                    <div className="card action">
                      <span className="kicker">If the warehouse runs short</span>
                      <p><strong>Split scarce stock by expected margin, not pro-rata.</strong> In a simulated shortage every week, worth about {usd(allocationCi.mean * 52)} a year across four stores
                        (95% CI {usd(allocationCi.lower * 52)} to {usd(allocationCi.upper * 52)}), {signedPct(lostChange)} lost revenue, no department starved.</p>
                      <p className="muted small"><a href="#/allocation">Next week’s split</a></p>
                    </div>
                  )}
                </div>
                {exceptions && exceptions.length > 0 && <WatchList rows={exceptions} health={health} />}
              </section>
            )}

            {health && health.length > 0 && <Health rows={health} />}

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
                note="More revenue than a proportional split across four stores, in a simulated shortage: supply held at 90% of forecast every week" />
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
        <Kpi label="Lost sales, 4 weeks" value={usd(standard.lost_sales_value)} delta={`${usd(standard.lost_sales_value - bare.lost_sales_value)} vs no safety stock`} tone="good"
          note={standard.sales_value ? `${pct(standard.lost_sales_value / (standard.sales_value + standard.lost_sales_value))} of demand (${pct(bare.lost_sales_value / ((bare.sales_value ?? 0) + bare.lost_sales_value))} without safety stock), at shelf price` : 'Demand that found an empty shelf, at shelf price'} />
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

function ServiceAction({ summary }: { summary: Summary }) {
  const twin = summary.twin
  const by = twin?.service_by_assumption
  const chosen = twin?.service_assumption ?? 'bias_corrected'
  const recs = by?.[chosen] ?? {}
  const clear = Object.entries(recs).filter(([, r]) => r.clear)
  const unclear = Object.entries(recs).filter(([, r]) => !r.clear).map(([c]) => titleCase(c))
  // Categories the uncorrected curve would push up: the advice an experienced reviewer would challenge.
  const raw = by?.every_unit ?? {}
  // ...but only where the corrected curve doesn't itself raise the target, so the two pieces of advice never conflict.
  const raises = (c: string) => (recs[c]?.clear ?? false) && (recs[c]?.inventory_change ?? 0) > 0
  const overstated = Object.entries(raw).filter(([c, r]) => r.clear && r.inventory_change > 0 && !raises(c))
  const stock = sum(clear.map(([, r]) => r.inventory_change))
  const label = (value: number) => `${(value * 100).toFixed(1).replace('.0', '')}%`
  return (
    <div className="card action">
      <span className="kicker">Service targets</span>
      {clear.length > 0 ? (
        <p><strong>Move {clear.map(([c, r]) => `${titleCase(c)} to ${label(r.service)}`).join(', ')}.</strong>{' '}
          Saves {usd(sum(clear.map(([, r]) => r.saving)))} of lost margin + holding over four weeks and {stock <= 0 ? `frees ${usd(-stock)}` : `ties up ${usd(stock)} more`} of stock.
          {unclear.length > 0 && <> {unclear.join(', ')}: keep as is.</>}</p>
      ) : (
        <p><strong>Keep today’s targets.</strong> Once stockouts are priced realistically, no category’s cheapest target beats them by more than simulation noise.</p>
      )}
      {overstated.length > 0 && (
        <p className="muted small">Don’t raise {overstated.map(([c, r]) => `${titleCase(c)} to ${label(r.service)} (+${usd(r.inventory_change)} stock)`).join(' or ')}: that only pays if every unmet unit is a lost sale,
          and the simulator overstates lost sales {twin?.lost_sales_bias ? `${twin.lost_sales_bias.toFixed(1)}×` : ''}.</p>
      )}
      <p className="muted small"><a href="#/twin?section=policy">Cost curves under each assumption</a></p>
    </div>
  )
}

function WatchList({ rows, health }: { rows: TwinExceptionRow[]; health?: InventoryHealthRow[] }) {
  const material = rows.filter((r) => r.expected_lost_value >= MATERIAL_USD || r.stockout_probability >= 0.5)
  const atRisk = sum(material.map((r) => r.expected_lost_value))
  const network = health?.find((h) => h.group_type === 'category' && h.group === 'all')
  const weekly = network ? network.sales_value / 4 : NaN
  const depts = new Map<string, number>()
  for (const r of material) depts.set(r.dept_id, (depts.get(r.dept_id) ?? 0) + r.expected_lost_value)
  const top = [...depts.entries()].sort((a, b) => b[1] - a[1])[0]
  return (
    <p className="muted small">
      <strong>Watch-list:</strong> {material.length > 0
        ? <>{material.length} product-store pairs have at least {usd(MATERIAL_USD)} at risk this week or are more likely than not to run out, {usd(atRisk)} in all
          {Number.isFinite(weekly) ? <> ({pct(atRisk / weekly, 2)} of weekly sales)</> : null}{top ? <>, most in {top[0]}</> : null}. Too small to expedite item by item.</>
        : <>nothing above {usd(MATERIAL_USD)} at risk this week.</>}{' '}
      <a href="#/twin?section=exceptions">Full list and download</a>
    </p>
  )
}

function RiskAction({ rows, responses }: { rows: TwinStressRow[]; responses?: TwinResponseRow[] }) {
  const lost = rows.filter((r) => r.policy === 'forecast_reorder' && r.metric === 'lost_sales_value' && (r.rationing ?? 'days_of_cover') === 'days_of_cover')
  const worst = [...lost].sort((a, b) => b.delta - a.delta)[0]
  if (!worst) return null
  const clear = responses ? bestResponse(responses, worst.scenario) : undefined
  // With no response clearly paying, still show the cheapest one, and say so.
  const response = clear ?? responses?.filter((r) => r.scenario === worst.scenario && r.response !== 'none').sort((a, b) => b.net_benefit - a.net_benefit)[0]
  const none = responses?.find((r) => r.scenario === worst.scenario && r.response === 'none')
  const others = (responses ?? []).filter((r) => r.scenario === worst.scenario && r.response !== 'none' && r !== response && r.net_benefit_upper < 0)
  return (
    <div className="card action">
      <span className="kicker">Biggest risk</span>
      <p><strong>{worst.label}:</strong> about {usd(worst.delta)} more lost sales over four weeks
        {Number.isFinite(worst.delta_lower) ? ` (90% of futures ${usd(worst.delta_lower)} to ${usd(worst.delta_upper)})` : ''}.</p>
      {response && none && (
        <p><strong>Best response:</strong> {response.label.toLowerCase()}. The extra loss falls to {usd(response.added_lost_sales)} at a cost of {usd(response.response_cost)}
          {clear ? `, net ${signedUsd(response.net_benefit)} of margin.` : `, about break-even on margin (${signedUsd(response.net_benefit)}, within noise).${others.length ? ` ${others.map((r) => r.label.split(' ')[0]).join(' and ')} cost${others.length === 1 ? 's' : ''} more than ${others.length === 1 ? 'it saves' : 'they save'}.` : ''}`}</p>
      )}
      <p className="muted small"><a href="#/twin?section=responses">Every response compared</a> · <a href={`#/twin?scenario=${worst.scenario}`}>Run it in the simulator</a></p>
    </div>
  )
}

/** The KPIs operations look at first: weeks of supply, turns, GMROI, losses as a share of sales, and the warehouse. */
function Health({ rows }: { rows: InventoryHealthRow[] }) {
  const categories = rows.filter((r) => r.group_type === 'category')
  const total = categories.find((r) => r.group === 'all')
  if (!total) return null
  const ordered = [...categories.filter((r) => r.group !== 'all').sort((a, b) => a.group.localeCompare(b.group)), total]
  const name = (r: InventoryHealthRow) => (r.group === 'all' ? 'All categories' : titleCase(r.group))
  return (
    <section>
      <h2>Stock health</h2>
      <div className="grid kpis">
        <Kpi label="Weeks of supply" value={total.weeks_of_supply.toFixed(1)} tone="flat" note={`${total.store_weeks_of_supply.toFixed(1)} in stores + ${(total.weeks_of_supply - total.store_weeks_of_supply).toFixed(1)} in the warehouse, at cost`} />
        <Kpi label="Inventory turns" value={`${total.turns.toFixed(1)}× a year`} tone="flat" note={`GMROI ${total.gmroi.toFixed(2)}: ${usd(total.gmroi)} of gross margin a year per $1 of stock (assumed margins)`} />
        <Kpi label="Lost sales" value={`${pct(total.lost_share)} of demand`} tone="flat" note={`${usd(total.lost_sales_value)} over 4 weeks against ${usd(total.sales_value)} sold`} />
        <Kpi label="Warehouse" value={`${pct(total.dc_fill_rate ?? NaN)} of store orders filled`} tone="flat"
          note={`${usd(total.dc_inventory_value ?? NaN)} on hand, ${usd(total.dc_on_order_value ?? NaN)} on order from the supplier`} />
      </div>
      <div className="table-wrap stack">
        <table className="stack health">
          <thead><tr><th className="text">Category</th><th>Weeks of supply</th><th>Turns / yr</th><th>GMROI</th><th>Fill rate</th><th>Lost, % of demand</th><th>Store stock</th><th>Warehouse stock</th></tr></thead>
          <tbody>
            {ordered.map((r) => (
              <tr key={r.group}>
                <td className="text" data-label="Category">{r.group === 'all' ? <strong>{name(r)}</strong> : name(r)}</td>
                <td className="num" data-label="Weeks of supply">{r.weeks_of_supply.toFixed(1)}</td><td className="num" data-label="Turns / yr">{r.turns.toFixed(1)}</td>
                <td className="num" data-label="GMROI">{r.gmroi.toFixed(2)}</td><td className="num" data-label="Fill rate">{pct(r.fill_rate)}</td>
                <td className="num" data-label="Lost, % of demand">{pct(r.lost_share)}</td><td className="num" data-label="Store stock">{usd(r.store_inventory_value)}</td>
                <td className="num" data-label="Warehouse stock">{r.dc_inventory_value != null ? usd(r.dc_inventory_value) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-head">
        <p className="muted small" style={{ margin: 0 }}>Replay of the four holdout weeks on what actually sold, forecast-based reordering, stores and warehouse together. Weeks of supply = stock at cost over weekly cost of goods sold; stock-to-sales is the same ratio. Margins are assumptions.</p>
        <CsvButton filename="inventory_health.csv" rows={rows} columns={[
          { header: 'group_type', value: (r) => r.group_type }, { header: 'group', value: (r) => r.group }, { header: 'weeks_of_supply', value: (r) => r.weeks_of_supply },
          { header: 'store_weeks_of_supply', value: (r) => r.store_weeks_of_supply }, { header: 'turns_per_year', value: (r) => r.turns }, { header: 'gmroi', value: (r) => r.gmroi },
          { header: 'fill_rate', value: (r) => r.fill_rate }, { header: 'in_stock_rate', value: (r) => r.in_stock_pct }, { header: 'lost_share_of_demand', value: (r) => r.lost_share },
          { header: 'sales_4wk', value: (r) => r.sales_value }, { header: 'lost_sales_4wk', value: (r) => r.lost_sales_value },
          { header: 'store_inventory_at_cost', value: (r) => r.store_inventory_value }, { header: 'warehouse_inventory_at_cost', value: (r) => r.dc_inventory_value },
        ]} />
      </div>
    </section>
  )
}

const titleCase = (value: string) => value.charAt(0) + value.slice(1).toLowerCase()
