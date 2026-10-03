import { useEffect, useMemo, useRef, useState } from 'react'
import { Area, Bar, CartesianGrid, ComposedChart, Legend, Line, LineChart, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson, getOptional } from '../data/load'
import type {
  KpiSet, Summary, TwinExceptionRow, TwinFrontierRow, TwinPolicyCurveRow, TwinPresetIndex, TwinRequest, TwinResult, TwinScenario, TwinStressRow, TwinTimelineRow,
  TwinValidationRow,
} from '../data/types'
import { Async, ChartBox, Kpi, Problem, Segmented, Select, type Tone } from '../components/ui'
import { useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { POLICY_LABELS, RATIONING_LABELS, num, pct, points, signedPct, signedUsd, usd } from '../lib/format'
import { mean } from '../lib/metrics'
import { runSimulation, STAGE_LABELS, warmSimulator, type Stage } from '../twin/client'

type Kind = 'pct' | 'usd' | 'units'
const METRIC_ROWS: { key: string; label: string; kind: Kind }[] = [
  { key: 'fill_rate', label: 'Fill rate', kind: 'pct' },
  { key: 'in_stock_pct', label: 'In-stock rate', kind: 'pct' },
  { key: 'lost_sales_value', label: 'Lost sales', kind: 'usd' },
  { key: 'inventory_value', label: 'Inventory (at cost)', kind: 'usd' },
  { key: 'units_demanded', label: 'Demand (units)', kind: 'units' },
]
const CATEGORY_COLORS: Record<string, string> = { FOODS: SERIES[0], HOUSEHOLD: SERIES[1], HOBBIES: SERIES[2] }
const fmt = (kind: Kind, value: number) => (kind === 'pct' ? pct(value) : kind === 'usd' ? usd(value) : num(value))
const hashParams = () => new URLSearchParams(window.location.hash.split('?')[1] ?? '')

export default function Twin() {
  const state = useData(async () => {
    const [validation, timeline, frontier, exceptions, presetIndex, summary, curve, stress] = await Promise.all([
      getOptional<TwinValidationRow[]>('twin_validation.json'), getOptional<TwinTimelineRow[]>('twin_timeline.json'),
      getOptional<TwinFrontierRow[]>('twin_frontier.json'), getOptional<TwinExceptionRow[]>('twin_exceptions.json'),
      getOptional<TwinPresetIndex>('twin/presets.json'), getJson<Summary>('summary.json'), getOptional<TwinPolicyCurveRow[]>('twin_policy_curve.json'),
      getOptional<TwinStressRow[]>('twin_stress.json'),
    ])
    return { validation, timeline, frontier, exceptions, presetIndex, summary, curve, stress }
  })
  useEffect(() => {
    const section = hashParams().get('section')
    if (section && state.data) document.getElementById(section)?.scrollIntoView({ block: 'start' })
  }, [state.data])
  return (
    <Async state={state}>
      {({ validation, timeline, frontier, exceptions, presetIndex, summary, curve, stress }) =>
        !validation || !presetIndex ? (
          <Problem message="The inventory twin tables are not in this build. Run the pipeline and `npm run data`." />
        ) : (
          <>
            <div className="question" style={{ marginTop: 18 }}>
              The simulator replays the chain (supplier → one warehouse → four stores) day by day for every product. The warehouse holds stock per product and shares it between the
              stores, so you can try a policy, a demand spike or a supply problem on screen first. Costs, lead times and starting stock are assumptions: the M5 data has none of them.
            </div>
            <WhatIf index={presetIndex} />
            {frontier && <Frontier rows={frontier} summary={summary} />}
            {curve && curve.length > 0 && <PolicyCurve rows={curve} summary={summary} />}
            {stress && <Rationing rows={stress} />}
            {timeline && exceptions && <Exceptions timeline={timeline} exceptions={exceptions} />}
            <Validation rows={validation} summary={summary} />
          </>
        )
      }
    </Async>
  )
}

/* ------------------------------------------------------------------ 1. what if */

type Shown = { label: string; detail: string; result: TwinResult }

function WhatIf({ index }: { index: TwinPresetIndex }) {
  const requested = hashParams().get('scenario')
  const presetKeys = Object.keys(index.presets)
  const [scope, setScope] = useState('all')
  const [policy, setPolicy] = useState(index.policies[0])
  const [preset, setPreset] = useState(requested && presetKeys.includes(requested) ? requested : presetKeys[0])
  const [custom, setCustom] = useState<Shown | undefined>()
  const [useCustom, setUseCustom] = useState(false)
  useEffect(() => {
    const warm = () => warmSimulator()
    const idle = (window as unknown as { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback
    const handle = idle ? idle(warm) : window.setTimeout(warm, 2500)
    return () => { if (!idle) window.clearTimeout(handle) }
  }, [])
  const shown: Shown | undefined = useMemo(() => {
    if (useCustom && custom) return custom
    const result = index.results[`${preset}|${policy}`]
    return result ? { label: index.presets[preset].label, detail: index.presets[preset].detail, result } : undefined
  }, [useCustom, custom, preset, policy, index])
  const where = scope === 'all' ? 'across the four stores' : `in ${scope}`
  const kpis = (outcome: TwinResult['baseline']) => outcome.kpis[scope] ?? outcome.kpis.all

  return (
    <section id="what-if">
      <h2>1 · What if…?</h2>
      <p className="answer">
        {shown?.result.scenario
          ? <>{shown.label}: lost sales {signedUsd(kpis(shown.result.scenario).lost_sales_value.mean - kpis(shown.result.baseline).lost_sales_value.mean)} over the 4 weeks {where},
            and fill rate moves {points(kpis(shown.result.scenario).fill_rate.mean - kpis(shown.result.baseline).fill_rate.mean)}.</>
          : 'Pick a scenario to compare it with business as usual.'}
      </p>
      <div className="presets">
        {presetKeys.map((key) => (
          <button key={key} type="button" className="preset" aria-pressed={!useCustom && preset === key} onClick={() => { setPreset(key); setUseCustom(false) }}>
            <strong>{index.presets[key].label}</strong><span>{index.presets[key].detail}</span>
          </button>
        ))}
        <button type="button" className="preset" aria-pressed={useCustom} onClick={() => custom ? setUseCustom(true) : document.getElementById('custom')?.scrollIntoView({ block: 'center' })}>
          <strong>Your scenario</strong><span>{custom ? custom.detail : 'Build one below and press Run'}</span>
        </button>
      </div>
      <div className="controls">
        <Select label="Show" value={scope} options={[{ value: 'all', label: 'All four stores' }, ...index.stores.map((s) => ({ value: s, label: s }))]} onChange={setScope} />
        <Segmented label="Reordering policy" value={policy} options={index.policies.map((p) => ({ value: p, label: p === 'forecast_reorder' ? 'Forecast-based' : 'Last-week (current)' }))} onChange={(v) => { setPolicy(v); setUseCustom(false) }} />
      </div>
      {shown && <ScenarioResult shown={shown} dates={index.dates} scope={scope} />}
      <CustomForm index={index} policy={policy} onResult={(result) => { setCustom(result); setUseCustom(true) }} />
    </section>
  )
}

function Delta({ label, kpis, base, metric, kind, goodWhen }: { label: string; kpis: KpiSet; base: KpiSet; metric: string; kind: 'pct' | 'usd'; goodWhen: 'up' | 'down' | 'neutral' }) {
  const now = kpis[metric], was = base[metric]
  const diff = now.mean - was.mean
  const tiny = Math.abs(diff) < (kind === 'pct' ? 0.0005 : 0.5)
  const tone: Tone = tiny || goodWhen === 'neutral' ? 'flat' : (diff > 0) === (goodWhen === 'up') ? 'good' : 'bad'
  const deltaText = tiny ? 'no change' : kind === 'pct' ? points(diff) : signedUsd(diff)
  return (
    <Kpi label={label} value={fmt(kind, now.mean)} delta={`${deltaText} vs baseline`} tone={tone}
      note={`90% of futures: ${fmt(kind, now.lower)} – ${fmt(kind, now.upper)} · baseline ${fmt(kind, was.mean)}`} />
  )
}

/** Department columns for one store, or each department summed over the stores. */
function departmentSeries(timeline: TwinResult['baseline']['timeline'], scope: string) {
  const parts = timeline.group.map((group) => group.split('|'))
  const depts = [...new Set(parts.map(([, dept]) => dept ?? parts[0][0]))].sort()
  const columns = (dept: string) => parts.flatMap(([store, d], i) => ((d ?? store) === dept && (scope === 'all' || store === scope || d === undefined) ? [i] : []))
  const pick = (matrix: number[][], dept: string) => matrix.map((row) => columns(dept).reduce((total, i) => total + (row[i] ?? 0), 0))
  return { depts, pick }
}

function ScenarioResult({ shown, dates, scope }: { shown: Shown; dates: string[]; scope: string }) {
  const { result } = shown
  const outcome = result.scenario ?? result.baseline
  const { depts, pick } = departmentSeries(outcome.timeline, scope)
  const [dept, setDept] = useState(depts[0])
  const active = depts.includes(dept) ? dept : depts[0]
  const onHand = pick(outcome.timeline.on_hand, active), demand = pick(outcome.timeline.demand, active), lost = pick(outcome.timeline.lost, active)
  const baseline = result.scenario ? pick(result.baseline.timeline.on_hand, active) : undefined
  const data = dates.map((date, day) => ({ date, onHand: onHand[day] ?? 0, demand: demand[day] ?? 0, lost: lost[day] ?? 0, baseline: baseline?.[day] }))
  const now = outcome.kpis[scope] ?? outcome.kpis.all
  const base = result.baseline.kpis[scope] ?? result.baseline.kpis.all
  return (
    <>
      <div className="grid kpis" style={{ marginTop: 12 }}>
        <Delta label="Fill rate" kpis={now} base={base} metric="fill_rate" kind="pct" goodWhen="up" />
        <Delta label="In-stock rate" kpis={now} base={base} metric="in_stock_pct" kind="pct" goodWhen="up" />
        <Delta label="Lost sales" kpis={now} base={base} metric="lost_sales_value" kind="usd" goodWhen="down" />
        <Delta label="Inventory (at cost)" kpis={now} base={base} metric="inventory_value" kind="usd" goodWhen="neutral" />
      </div>
      <div className="controls"><Select label="Department" value={active} options={depts.map((d) => ({ value: d, label: d }))} onChange={setDept} /></div>
      <TimelineChart data={data} />
      <p className="legend-note">
        {scope === 'all' ? 'All four stores' : scope}, average of {result.reps} simulated futures sharing the same random demand, so differences come from the scenario alone. {shown.detail}.
        Policy: {POLICY_LABELS[result.policy]}.
      </p>
    </>
  )
}

function TimelineChart({ data }: { data: { date: string; onHand: number; demand: number; lost: number; baseline?: number }[] }) {
  return (
    <ChartBox>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid {...gridProps} />
        <XAxis dataKey="date" {...axisProps} tickFormatter={(d: string) => d.slice(5)} minTickGap={28} />
        <YAxis {...axisProps} width={56} label={{ value: 'Units', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
        <Tooltip {...tooltipStyle} formatter={(v: unknown) => num(Number(v), 0)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="demand" name="Demand" fill="var(--grid)" isAnimationActive={false} />
        <Area dataKey="lost" name="Lost sales" stroke="none" fill={SERIES[1]} fillOpacity={0.55} isAnimationActive={false} />
        {data.some((d) => d.baseline !== undefined) && <Line dataKey="baseline" name="On hand (baseline)" stroke="var(--muted)" strokeDasharray="4 3" strokeWidth={1.8} dot={false} isAnimationActive={false} />}
        <Line dataKey="onHand" name="On hand" stroke={SERIES[0]} strokeWidth={2.4} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ChartBox>
  )
}

function CustomForm({ index, policy, onResult }: { index: TwinPresetIndex; policy: string; onResult: (shown: Shown) => void }) {
  const [demandScale, setDemandScale] = useState(1.3)
  const [category, setCategory] = useState('FOODS')
  const [start, setStart] = useState(7)
  const [length, setLength] = useState(7)
  const [dcFactor, setDcFactor] = useState(1)
  const [delay, setDelay] = useState(0)
  const [delayKind, setDelayKind] = useState<'once' | 'replan' | 'lasting'>('replan')
  const [rationing, setRationing] = useState('days_of_cover')
  const [service, setService] = useState('current')
  const [reps, setReps] = useState(20)
  const [stage, setStage] = useState<Stage | undefined>()
  const [error, setError] = useState<string | undefined>()
  const busy = useRef(false)
  const run = async () => {
    if (busy.current) return
    busy.current = true
    setError(undefined)
    const days: [number, number] = [start, Math.min(start + length, 28)]
    const scenario: TwinScenario = {
      demand_scale: demandScale, category: category === 'ALL' ? null : category, days, dc_factor: dcFactor, dc_days: dcFactor < 1 ? [0, 28] : null, delay,
      delay_days: delay > 0 && delayKind === 'once' ? [0, 7] : null, replan_after: delay > 0 && delayKind === 'replan' ? 7 : null,
    }
    const request: TwinRequest = { policy, service: service === 'current' ? 'current' : Number(service), rationing, scenario, reps, seed: 1 }
    const delayText = { once: `this week’s supplier delivery ${delay} days late`, replan: `supplier ${delay} days slower, planner adjusts after a week`, lasting: `supplier ${delay} days slower, never planned for` }[delayKind]
    const parts = [
      demandScale !== 1 ? `${category === 'ALL' ? 'All' : category} demand ×${demandScale.toFixed(2)} for days ${days[0] + 1}–${days[1]}` : '',
      dcFactor < 1 ? `warehouse receives ${Math.round((1 - dcFactor) * 100)}% less` : '', delay > 0 ? delayText : '',
      rationing !== 'days_of_cover' ? RATIONING_LABELS[rationing].toLowerCase() : '', service !== 'current' ? `${pct(Number(service), 1).replace('.0%', '%')} service target` : '',
    ].filter(Boolean)
    try {
      const result = await runSimulation(request, setStage)
      onResult({ label: 'Your scenario', detail: parts.join(', ') || 'no change from baseline', result })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStage(undefined)
      busy.current = false
    }
  }
  return (
    <details id="custom" open>
      <summary>Build your own scenario</summary>
      <div className="card">
        <div className="controls">
          <label className="field"><span>Demand multiplier: ×{demandScale.toFixed(2)}</span><input type="range" min="0.5" max="2" step="0.05" value={demandScale} onChange={(e) => setDemandScale(Number(e.target.value))} /></label>
          <Select label="Applies to" value={category} options={[{ value: 'ALL', label: 'All categories' }, { value: 'FOODS', label: 'Foods' }, { value: 'HOUSEHOLD', label: 'Household' }, { value: 'HOBBIES', label: 'Hobbies' }]} onChange={setCategory} />
          <label className="field"><span>Starts on day {start + 1}</span><input type="range" min="0" max="27" value={start} onChange={(e) => setStart(Number(e.target.value))} /></label>
          <label className="field"><span>Lasts {length} days</span><input type="range" min="1" max="28" value={length} onChange={(e) => setLength(Number(e.target.value))} /></label>
        </div>
        <div className="controls">
          <label className="field"><span>Warehouse receives {Math.round(dcFactor * 100)}% of ordered stock</span><input type="range" min="0.4" max="1" step="0.05" value={dcFactor} onChange={(e) => setDcFactor(Number(e.target.value))} /></label>
          <label className="field"><span>Supplier delay: {delay} extra days</span><input type="range" min="0" max="14" value={delay} onChange={(e) => setDelay(Number(e.target.value))} /></label>
          <Select label="Kind of delay" value={delayKind} options={[{ value: 'once', label: 'One late delivery' }, { value: 'replan', label: 'Lasting; planner adjusts after a week' }, { value: 'lasting', label: 'Lasting; never planned for' }]} onChange={setDelayKind} />
        </div>
        <div className="controls">
          <Select label="When the warehouse is short" value={rationing} options={Object.entries(RATIONING_LABELS).map(([value, label]) => ({ value, label }))} onChange={setRationing} />
          <Select label="Service target" value={service} options={[{ value: 'current', label: 'Segment targets (current)' }, ...index.service_grid.map((s) => ({ value: String(s), label: pct(s, 1).replace('.0%', '%') }))]} onChange={setService} />
          <Select label="Simulated futures" value={String(reps)} options={[10, 20, 50].map((n) => ({ value: String(n), label: String(n) }))} onChange={(v) => setReps(Number(v))} />
        </div>
        <button type="button" className="primary" onClick={run} disabled={!!stage}>{stage ? 'Running…' : 'Run simulation'}</button>
        {stage && <p className="muted small" role="status">{STAGE_LABELS[stage]}</p>}
        {error && <div className="callout warn" role="alert">{error}</div>}
        <p className="legend-note">Runs on your device: the Python simulator loads into the page the first time (about 12 MB, then cached) and simulates all four stores and the warehouse together. Nothing is sent to a server.</p>
      </div>
    </details>
  )
}

/* ------------------------------------------------------------------ 2. which policy */

function Frontier({ rows, summary }: { rows: TwinFrontierRow[]; summary: Summary }) {
  const categories = [...new Set(rows.map((r) => r.category))].sort()
  const grid = rows.filter((r) => r.service_level !== null)
  const recommended = rows.filter((r) => r.recommended)
  const current = rows.filter((r) => r.service_level === null)
  const clear = recommended.filter((r) => r.clear_saving)
  const edge = summary.twin?.service_at_grid_edge ?? recommended.filter((r) => r.at_grid_edge).map((r) => r.category)
  return (
    <section id="policy">
      <h2>2 · Which service targets should we run?</h2>
      <p className="answer">
        {clear.length > 0
          ? <>Move {clear.map((r) => `${r.category} to ${r.label}`).join(', ')}: that saves {usd(clear.reduce((a, r) => a + r.saving_vs_current, 0))} of lost margin + holding cost over 4 weeks.</>
          : <>Today’s segment targets are already about as cheap as any single target: no category saves more than simulation noise.</>}
        {recommended.filter((r) => !r.clear_saving).length > 0 && clear.length > 0 && <> {recommended.filter((r) => !r.clear_saving).map((r) => r.category).join(', ')}: no clear gain.</>}
      </p>
      <p className="muted small">
        Each line sweeps the service target from 80% to 99.5% for one category. More stock means fewer lost sales but more cash tied up. Cost = lost margin + holding cost, with stock valued at
        cost (assumed margin and holding rate by category). Every option sees the same simulated futures, so savings are measured future by future; a saving whose 90% range includes zero is not a clear win.
        {edge.length > 0 && <> {edge.join(', ')} is cheapest at the edge of the tested range.</>}
      </p>
      <ChartBox size="tall">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 40, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis type="number" dataKey="inventory_value" name="Inventory" {...axisProps} tickFormatter={(v: number) => usd(v)} domain={['auto', 'auto']} label={{ value: 'Average inventory (at cost)', position: 'insideBottom', offset: -12, fill: 'var(--muted)', fontSize: 12 }} />
          <YAxis type="number" dataKey="lost_sales_value" name="Lost sales" {...axisProps} width={60} tickFormatter={(v: number) => usd(v)} label={{ value: 'Lost sales (4 weeks)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
          <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: '3 3' }} formatter={(v: unknown, name: string) => (name === 'Inventory' || name === 'Lost sales' ? usd(Number(v)) : String(v))} labelFormatter={() => ''} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} verticalAlign="bottom" />
          {categories.map((category) => (
            <Scatter key={category} name={category} data={grid.filter((r) => r.category === category)} fill={CATEGORY_COLORS[category]} line={{ stroke: CATEGORY_COLORS[category], strokeWidth: 1.6 }} isAnimationActive={false} />
          ))}
          <Scatter name="Cheapest" data={recommended} fill="var(--ink)" shape="star" legendType="star" isAnimationActive={false} />
          <Scatter name="Current segment targets" data={current} fill="var(--muted)" shape="diamond" legendType="diamond" isAnimationActive={false} />
        </ScatterChart>
      </ChartBox>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th>Category</th><th>Cheapest target</th><th>Fill rate</th><th>Inventory</th><th>Total cost</th><th>Current cost</th><th>Saving (90% range)</th><th className="text">Verdict</th></tr></thead>
          <tbody>
            {recommended.map((r) => {
              const now = current.find((c) => c.category === r.category)
              return (
                <tr key={r.category}>
                  <td data-label="Category">{r.category}</td><td className="num" data-label="Cheapest target">{r.label}</td><td className="num" data-label="Fill rate">{pct(r.fill_rate)}</td>
                  <td className="num" data-label="Inventory">{usd(r.inventory_value)}</td><td className="num" data-label="Total cost">{usd(r.total_cost)}</td>
                  <td className="num" data-label="Current cost">{now ? usd(now.total_cost) : '–'}</td>
                  <td className="num" data-label="Saving (90% range)">{usd(r.saving_vs_current)} ({usd(r.saving_lower)} to {usd(r.saving_upper)})</td>
                  <td className={`text ${r.clear_saving ? 'good' : 'flat'}`} data-label="Verdict">{r.clear_saving ? 'Change' : 'Keep current'}{r.at_grid_edge ? ' · at range edge' : ''}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function PolicyCurve({ rows, summary }: { rows: TwinPolicyCurveRow[]; summary: Summary }) {
  const equal = summary.twin?.equal_inventory
  const series = (policy: string) => rows.filter((r) => r.policy === policy).sort((a, b) => a.inventory_value - b.inventory_value)
  const forecast = series('forecast_reorder'), naive = series('last_week_reorder')
  const data = [...forecast.map((r) => ({ inventory: r.inventory_value, forecast: r.fill_rate, x: r.safety_multiplier })), ...naive.map((r) => ({ inventory: r.inventory_value, naive: r.fill_rate, x: r.safety_multiplier }))]
    .sort((a, b) => a.inventory - b.inventory)
  return (
    <section id="equal-inventory">
      <h3>Is the forecast better, or does it just hold more stock?</h3>
      <p>
        {verdict(equal)}
      </p>
      <ChartBox size="short">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 0 }}>
          <CartesianGrid {...gridProps} />
          <XAxis type="number" dataKey="inventory" {...axisProps} domain={['auto', 'auto']} tickFormatter={(v: number) => usd(v)} label={{ value: 'Average inventory (at cost)', position: 'insideBottom', offset: -14, fill: 'var(--muted)', fontSize: 12 }} />
          <YAxis {...axisProps} width={56} domain={['auto', 1]} tickFormatter={(v: number) => pct(v, 0)} />
          <Tooltip {...tooltipStyle} formatter={(v: unknown) => pct(Number(v))} labelFormatter={(v: unknown) => `Inventory ${usd(Number(v))}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" />
          <Line dataKey="forecast" name="Forecast-based reorder" stroke={SERIES[0]} strokeWidth={2.2} connectNulls isAnimationActive={false} />
          <Line dataKey="naive" name="Last-week reorder (current practice)" stroke={SERIES[1]} strokeWidth={2.2} connectNulls isAnimationActive={false} />
        </LineChart>
      </ChartBox>
      <p className="legend-note">Fill rate against inventory as the safety stock is scaled from 0× to 4×, both policies carrying the same safety stock, replayed on what actually sold in the holdout weeks.</p>
    </section>
  )
}

function verdict(equal: NonNullable<Summary['twin']>['equal_inventory']) {
  if (!equal) return null
  const fillGap = equal.current_practice != null && Number.isFinite(equal.current_practice) ? equal.forecast - equal.current_practice : NaN
  const naiveStock = equal.current_practice_inventory_for_same
  const extra = naiveStock != null && Number.isFinite(naiveStock) ? naiveStock - equal.inventory_value : NaN
  const lead = !Number.isFinite(fillGap) ? 'Better.' : fillGap >= 0.01 ? 'Clearly better.' : fillGap > 0 ? 'Only slightly better.' : 'Not better.'
  return (
    <>
      <strong>{lead}</strong>{' '}
      {Number.isFinite(fillGap)
        ? <>At the same {usd(equal.inventory_value)} of stock the forecast fills {pct(equal.forecast)} of demand and last-week reordering {pct(equal.current_practice!)}.</>
        : <>Last-week reordering does not reach the forecast’s fill rate anywhere in the range tested.</>}
      {Number.isFinite(extra) && extra > 0 && <> Matching the forecast with last-week reordering takes about {usd(extra)} ({pct(extra / naiveStock!, 0)}) more stock.</>}
      {' '}Most of the service comes from the safety stock both policies carry: over a 3–8 day replenishment window, last week’s sales rate is already a fair estimate for most products.
    </>
  )
}

function Rationing({ rows }: { rows: TwinStressRow[] }) {
  const cut = rows.filter((r) => r.scenario === 'dc_cut' && r.policy === 'forecast_reorder' && r.metric === 'lost_sales_value' && r.rationing)
  if (cut.length < 2) return null
  const best = [...cut].sort((a, b) => a.delta - b.delta)[0]
  const spread = Math.max(...cut.map((r) => r.delta)) - best.delta
  return (
    <section id="rationing">
      <h3>When the warehouse is short, how should it share each product?</h3>
      <p>
        {spread < 0.05 * Math.abs(best.delta)
          ? <>With 30% less stock arriving for two weeks, <strong>the rule barely matters</strong>: every rule loses about {usd(best.delta)} of extra sales. Each product’s shortfall is shared between
            just four stores with similar cover, so the stores that miss out are much the same whichever rule decides.</>
          : <>With 30% less stock arriving for two weeks, <strong>{RATIONING_LABELS[best.rationing].toLowerCase()}</strong> loses the least: {usd(best.delta)} of extra lost sales.</>}
        {' '}This is the product-by-product version of the department split on the Allocate page.
      </p>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th>Rule</th><th>Extra lost sales</th><th>90% of futures</th></tr></thead>
          <tbody>
            {cut.sort((a, b) => a.delta - b.delta).map((r) => (
              <tr key={r.rationing}><td data-label="Rule">{RATIONING_LABELS[r.rationing]}</td><td className="num" data-label="Extra lost sales">{usd(r.delta)}</td><td className="num" data-label="90% of futures">{usd(r.delta_lower)} to {usd(r.delta_upper)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend-note">Days of cover serves the store closest to running out first; by value serves the highest-margin row first; proportional gives each store the same share of its order.</p>
    </section>
  )
}

/* ------------------------------------------------------------------ 3. where will we run out */

function Exceptions({ timeline, exceptions }: { timeline: TwinTimelineRow[]; exceptions: TwinExceptionRow[] }) {
  const stores = [...new Set(timeline.map((r) => r.store_id))].sort()
  const [store, setStore] = useState(stores[0])
  const depts = [...new Set(timeline.filter((r) => r.store_id === store).map((r) => r.dept_id))].sort()
  const [dept, setDept] = useState(depts[0])
  const activeDept = depts.includes(dept) ? dept : depts[0]
  const [onlyStore, setOnlyStore] = useState(false)
  const [order, setOrder] = useState<'value' | 'chance'>('value')
  const series = timeline.filter((r) => r.store_id === store && r.dept_id === activeDept)
  const sorted = [...exceptions].sort((a, b) => order === 'value' ? b.expected_lost_value - a.expected_lost_value : b.stockout_probability - a.stockout_probability || b.expected_lost_value - a.expected_lost_value)
  const list = sorted.filter((r) => !onlyStore || r.store_id === store).slice(0, 20)
  const atRisk = exceptions.filter((r) => r.stockout_probability >= 0.5).length
  const total = exceptions.reduce((a, r) => a + r.expected_lost_value, 0)
  return (
    <section id="exceptions">
      <h2>3 · Where will we run out this week?</h2>
      <p className="answer">
        Even on forecast-based reordering, the {num(exceptions.length)} products with the most at stake put {usd(total)} of sales at risk over the next 7 days; {num(atRisk)} of them are more likely than not to run out.
      </p>
      <div className="controls">
        <Segmented label="Sort by" value={order} options={[{ value: 'value', label: 'Sales at risk' }, { value: 'chance', label: 'Chance of stockout' }]} onChange={setOrder} />
        <Segmented label="Show" value={onlyStore ? 'store' : 'all'} options={[{ value: 'all', label: 'All stores' }, { value: 'store', label: `Only ${store}` }]} onChange={(v) => setOnlyStore(v === 'store')} />
      </div>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th>Item</th><th className="text">Store</th><th className="text">Dept</th><th>Chance of stockout (7 days)</th><th>Expected lost units</th><th>Sales at risk</th></tr></thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.series_id}><td data-label="Item">{r.item_id}</td><td className="text" data-label="Store">{r.store_id}</td><td className="text" data-label="Dept">{r.dept_id}</td>
                <td className={`num ${r.stockout_probability >= 0.5 ? 'bad' : ''}`} data-label="Chance of stockout (7 days)">{pct(r.stockout_probability, 0)}</td>
                <td className="num" data-label="Expected lost units">{r.expected_lost_units.toFixed(1)}</td><td className="num" data-label="Sales at risk">{usd(r.expected_lost_value)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend-note">Forecast-based reordering, all four stores and the warehouse simulated together. Starting stock comes from replaying the last two weeks.</p>
      <h3>Stock on hand vs demand: replay of the holdout window</h3>
      <div className="controls">
        <Select label="Store" value={store} options={stores.map((s) => ({ value: s, label: s }))} onChange={setStore} />
        <Select label="Department" value={activeDept} options={depts.map((d) => ({ value: d, label: d }))} onChange={setDept} />
      </div>
      <TimelineChart data={series.map((r) => ({ date: r.date, onHand: r.on_hand, demand: r.demand, lost: r.lost }))} />
      <p className="legend-note">Forecast-based reorder replayed on actual sales. Orange shading = demand that found an empty shelf.</p>
    </section>
  )
}

/* ------------------------------------------------------------------ 4. can we trust it */

function Validation({ rows, summary }: { rows: TwinValidationRow[]; summary: Summary }) {
  const network = rows.filter((r) => r.store_id === 'all' && r.policy === 'forecast_reorder')
  const stores = rows.filter((r) => r.store_id !== 'all' && r.policy === 'forecast_reorder')
  const folds = [...new Set(network.map((r) => r.fold))].sort()
  const [fold, setFold] = useState(folds.at(-1) ?? '')
  const shown = network.filter((r) => r.fold === fold)
  const allInBand = network.filter((r) => r.in_band).length
  const storeInBand = stores.filter((r) => r.in_band).length
  const miss = (metric: string) => mean(network.filter((r) => r.metric === metric).map((r) => Math.abs(r.predicted - r.realised)))
  const ratio = (metric: string) => mean(network.filter((r) => r.metric === metric).map((r) => r.predicted)) / mean(network.filter((r) => r.metric === metric).map((r) => r.realised))
  const gap = summary.twin?.fill_rate_gap
  return (
    <section id="trust">
      <h2>4 · How far can we trust it?</h2>
      <p className="answer">
        For service levels, closely: across {folds.length} windows the network fill-rate prediction is off by {points(miss('fill_rate')).replace('+', '')} on average and in-stock by {points(miss('in_stock_pct')).replace('+', '')}.
        Its ranges are still too narrow: the realised result fell inside the 90% range in {allInBand} of {network.length} network checks and {storeInBand} of {stores.length} store checks.
      </p>
      <p className="muted">
        Predicted dollar losses run {ratio('lost_sales_value').toFixed(1)}× realised: predicted demand is {signedPct(ratio('units_demanded') - 1, 0)} vs what sold, and what sold is itself held down by real
        stockouts in the M5 data. Lost sales is a tail quantity, so a small demand overshoot moves it a lot. Read dollar figures as policy-vs-policy comparisons, not forecasts.
        {gap !== undefined && <> Latest fill-rate gap {points(gap).replace('+', '')}; the retraining check alerts above 5.0 pts.</>}
      </p>
      <p className="muted small">
        Each row: the simulator predicts the window from forecast and calibration data only (50 random futures, all four stores and the warehouse together), then the actual sales are replayed
        through the same policy. The bar is the 90% range of the simulated futures; the line is what actually happened.
      </p>
      <div className="controls"><Select label="Window" value={fold} options={folds.map((f) => ({ value: f, label: f.replace('_', ' ') }))} onChange={setFold} /></div>
      <div className="card">
        {METRIC_ROWS.map(({ key, label, kind }) => {
          const row = shown.find((r) => r.metric === key)
          return row ? <RangeRow key={key} label={label} kind={kind} row={row} /> : null
        })}
      </div>
      <details>
        <summary>All validation checks</summary>
        <div className="table-wrap" style={{ maxHeight: 360 }}>
          <table>
            <thead><tr><th>Window</th><th className="text">Store</th><th className="text">Policy</th><th className="text">Metric</th><th>Predicted</th><th>Range</th><th>Realised</th><th className="text">Inside</th></tr></thead>
            <tbody>
              {rows.map((r, i) => {
                const kind = r.metric.includes('value') || r.metric.includes('margin') || r.metric.includes('cost') ? 'usd' : r.metric.includes('units') ? 'units' : 'pct'
                const f = (v: number) => (kind === 'pct' ? pct(v) : kind === 'usd' ? usd(v) : num(v))
                return (
                  <tr key={i}><td>{r.fold.replace('_', ' ')}</td><td className="text">{r.store_id}</td><td className="text">{POLICY_LABELS[r.policy]}</td><td className="text">{r.metric}</td>
                    <td className="num">{f(r.predicted)}</td><td className="num">{f(r.lower)} – {f(r.upper)}</td><td className="num">{f(r.realised)}</td><td className="text">{r.in_band ? '✓' : '✗'}</td></tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  )
}

function RangeRow({ label, kind, row }: { label: string; kind: Kind; row: TwinValidationRow }) {
  const lo = Math.min(row.lower, row.realised)
  const hi = Math.max(row.upper, row.realised)
  const pad = (hi - lo || Math.abs(hi) * 0.1 || 1) * 0.25
  const scale = (v: number) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100
  return (
    <div className="range">
      <div><strong>{label}</strong><div className="muted small">predicted {fmt(kind, row.predicted)}</div></div>
      <svg viewBox="0 0 100 30" preserveAspectRatio="none" role="img" aria-label={`${label}: range ${fmt(kind, row.lower)} to ${fmt(kind, row.upper)}, realised ${fmt(kind, row.realised)}`}>
        <line x1="0" x2="100" y1="15" y2="15" stroke="var(--grid)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        <rect x={scale(row.lower)} width={Math.max(scale(row.upper) - scale(row.lower), 0.6)} y="9" height="12" rx="2" fill="var(--c1)" opacity="0.35" />
        <line x1={scale(row.realised)} x2={scale(row.realised)} y1="3" y2="27" stroke={row.in_band ? 'var(--good)' : 'var(--bad)'} strokeWidth="3" vectorEffect="non-scaling-stroke" />
      </svg>
      <div>
        <span className={`pill ${row.in_band ? 'in' : 'out'}`}>{row.in_band ? 'inside range' : 'outside range'}</span>{' '}
        <span className="num">realised {fmt(kind, row.realised)}</span>
      </div>
    </div>
  )
}
