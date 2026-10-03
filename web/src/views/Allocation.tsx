import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson, getOptional } from '../data/load'
import type { AllocationBacktestRow, AllocationRow, SafetyRow, Summary } from '../data/types'
import { Async, ChartBox, Kpi, Question, Segmented } from '../components/ui'
import { useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { num, pct, signedPct, usd } from '../lib/format'
import { sum, weeklyUplift } from '../lib/metrics'

export default function Allocation() {
  const [store, setStore] = useState('CA_1')
  const [query, setQuery] = useState('')
  const state = useData(async () => ({
    allocation: await getJson<AllocationRow[]>('allocation.json'),
    backtest: await getJson<AllocationBacktestRow[]>('allocation_backtest.json'),
    summary: await getJson<Summary>('summary.json'),
  }))
  const safety = useData(() => getOptional<SafetyRow[]>('safety_stock.json'), [])
  return (
    <Async state={state}>
      {({ allocation, backtest, summary }) => {
        const stores = [...new Set(allocation.map((r) => r.store_id))].sort()
        const rows = allocation.filter((r) => r.store_id === store).sort((a, b) => a.dept_id.localeCompare(b.dept_id)).map((r) => ({ ...r, label: `${r.dept_id} ($${r.unit_value.toFixed(2)}/unit)` }))
        const weeks = weeklyUplift(backtest)
        const lp = backtest.filter((r) => r.policy === 'lp_scenario')
        const pro = backtest.filter((r) => r.policy === 'pro_rata')
        const hasLoss = lp.every((r) => r.lost_revenue !== undefined)
        const ci = summary.allocation_vs_pro_rata_ci
        const fair = summary.allocation_fairness
        const filtered = (safety.data ?? []).filter((r) => !query || `${r.item_id}${r.store_id}`.toLowerCase().includes(query.toLowerCase())).slice(0, 300)
        return (
          <>
            <Question>when the warehouse can’t cover every store’s demand, who gets the stock?</Question>
            <p>
              <strong>Week of {allocation[0].week_start}.</strong> The DC holds {num(allocation[0].supply)} units (90% of forecast demand). The scenario LP maximises expected
              <em> margin</em> over conformal demand scenarios, with every department held to at least half of its forecast. “Pro-rata” splits supply in proportion to the point forecast.
            </p>
            <div className="controls"><Segmented label="Store" value={store} options={stores.map((s) => ({ value: s, label: s }))} onChange={setStore} /></div>
            <ChartBox size="tall">
              <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="label" {...axisProps} interval={0} tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <YAxis {...axisProps} width={56} label={{ value: 'Units for the week', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
                <Tooltip {...tooltipStyle} formatter={(v: unknown) => `${num(Number(v))} units`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="allocated_quantity" name="Scenario LP" fill={SERIES[0]} isAnimationActive={false} />
                <Bar dataKey="pro_rata_quantity" name="Pro-rata" fill={SERIES[1]} isAnimationActive={false} />
                <Line dataKey="forecast" name="Forecast demand" stroke="var(--ink)" strokeWidth={0} dot={{ r: 5, fill: 'var(--ink)' }} isAnimationActive={false} />
              </ComposedChart>
            </ChartBox>
            <p className="legend-note">Stock is allocated to 28 targets (7 departments × 4 stores) drawing on the same DC supply; this chart shows one store. Labels give each department’s average selling price.</p>

            <h2>Backtest: optimiser vs pro-rata</h2>
            <div className="grid kpis">
              <Kpi label="Revenue fulfilled" value={signedPct(sum(lp.map((r) => r.revenue_fulfilled)) / sum(pro.map((r) => r.revenue_fulfilled)) - 1)} tone="flat"
                note={ci?.revenue_fulfilled ? `Weekly difference ${usd(ci.revenue_fulfilled.mean)} (95% CI ${usd(ci.revenue_fulfilled.lower)} to ${usd(ci.revenue_fulfilled.upper)}); optimiser won ${weeks.filter((w) => w.uplift > 0).length} of ${weeks.length} weeks` : `${weeks.filter((w) => w.uplift > 0).length} of ${weeks.length} weeks won`} />
              {hasLoss && <Kpi label="Lost revenue" value={signedPct(sum(lp.map((r) => r.lost_revenue ?? 0)) / sum(pro.map((r) => r.lost_revenue ?? 1)) - 1)} tone="flat" note="Revenue on demand that went unfilled, optimiser vs pro-rata" />}
              {hasLoss && <Kpi label="Units lost" value={signedPct(sum(lp.map((r) => r.units_lost ?? 0)) / sum(pro.map((r) => r.units_lost ?? 1)) - 1)} tone="flat" note="The optimiser can ship fewer units while earning more: it favours margin over unit count" />}
              {fair && <Kpi label="Starved departments" value={String(fair.lp_scenario?.length ?? 0)} tone={(fair.lp_scenario?.length ?? 0) === 0 ? 'good' : 'bad'} note="Nodes under half fill for more than two weeks running" />}
            </div>
            <ChartBox size="short">
              <BarChart data={weeks} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="label" {...axisProps} interval={0} angle={-30} textAnchor="end" height={60} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                <YAxis {...axisProps} width={52} tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                <Tooltip {...tooltipStyle} formatter={(v: unknown) => signedPct(Number(v), 2)} />
                <Bar dataKey="uplift" name="Revenue vs pro-rata" fill={SERIES[0]} isAnimationActive={false} />
              </BarChart>
            </ChartBox>
            <p className="legend-note">Realised revenue fulfilled by the optimiser vs pro-rata, per week (scored on what actually sold).</p>
            {hasLoss && (
              <>
                <ChartBox size="short">
                  <BarChart data={weeks} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="label" {...axisProps} interval={0} angle={-30} textAnchor="end" height={60} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                    <YAxis {...axisProps} width={56} tickFormatter={(v: number) => usd(v)} />
                    <Tooltip {...tooltipStyle} formatter={(v: unknown) => usd(Number(v))} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="lpLost" name="Scenario LP" fill={SERIES[0]} isAnimationActive={false} />
                    <Bar dataKey="proLost" name="Pro-rata" fill={SERIES[1]} isAnimationActive={false} />
                  </BarChart>
                </ChartBox>
                <p className="legend-note">Revenue lost to unfilled demand each week. With supply at 90% of forecast some loss is unavoidable; the LP puts it where it costs least.</p>
              </>
            )}

            <details>
              <summary>Item-level safety stock (week 1, service targets by value and variability class)</summary>
              <p className="muted small">
                Each item’s target covers its replenishment window (lead time + days between orders). Staples (class A, steady) are held to 98% service; the erratic long tail (C, Z) to 85%.
                Order-up-to includes a two-unit shelf minimum.
              </p>
              <div className="controls"><label className="field"><span>Filter items</span><input type="text" placeholder="e.g. FOODS_3_090" value={query} onChange={(e) => setQuery(e.target.value)} /></label></div>
              <div className="table-wrap" style={{ maxHeight: 420 }}>
                <table>
                  <thead><tr><th>Item</th><th className="text">Store</th><th>Window forecast</th><th>Safety stock</th><th>Order-up-to</th><th>Service target</th><th>Expected fill</th><th className="text">Class</th></tr></thead>
                  <tbody>
                    {filtered.map((r) => (
                      <tr key={`${r.item_id}${r.store_id}`}>
                        <td>{r.item_id}</td><td className="text">{r.store_id}</td><td className="num">{r.forecast.toFixed(1)}</td><td className="num">{r.safety_stock.toFixed(1)}</td><td className="num">{r.order_up_to.toFixed(1)}</td>
                        <td className="num">{r.service_level ? pct(r.service_level, 0) : '–'}</td><td className="num">{r.expected_fill_rate ? pct(r.expected_fill_rate) : '–'}</td><td className="text">{r.abc && r.xyz ? `${r.abc}${r.xyz}` : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="legend-note">Showing the 300 highest-volume matches of {num(safety.data?.length ?? 0)} items.</p>
            </details>
          </>
        )
      }}
    </Async>
  )
}
