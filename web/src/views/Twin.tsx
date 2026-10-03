import { useEffect, useMemo, useRef, useState } from 'react'
import { Area, Bar, CartesianGrid, ComposedChart, Legend, Line, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson, getOptional } from '../data/load'
import type {
  KpiSet, Summary, TwinExceptionRow, TwinFrontierRow, TwinPresetIndex, TwinPresets, TwinRequest, TwinResult, TwinScenario, TwinTimelineRow, TwinValidationRow,
} from '../data/types'
import { Async, ChartBox, Kpi, Loading, Problem, Segmented, Select, type Tone } from '../components/ui'
import { useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { POLICY_LABELS, num, pct, points, signedPct, signedUsd, usd } from '../lib/format'
import { mean } from '../lib/metrics'
import { runSimulation, STAGE_LABELS, warmSimulator, type Stage } from '../twin/client'

type Kind = 'pct' | 'usd' | 'units'
const METRIC_ROWS: { key: string; label: string; kind: Kind }[] = [
  { key: 'fill_rate', label: 'Fill rate', kind: 'pct' },
  { key: 'in_stock_pct', label: 'In-stock rate', kind: 'pct' },
  { key: 'lost_sales_value', label: 'Lost sales', kind: 'usd' },
  { key: 'inventory_value', label: 'Inventory on hand', kind: 'usd' },
  { key: 'units_demanded', label: 'Demand (units)', kind: 'units' },
]
const CATEGORY_COLORS: Record<string, string> = { FOODS: SERIES[0], HOUSEHOLD: SERIES[1], HOBBIES: SERIES[2] }
const fmt = (kind: Kind, value: number) => (kind === 'pct' ? pct(value) : kind === 'usd' ? usd(value) : num(value))

export default function Twin() {
  const state = useData(async () => {
    const [validation, timeline, frontier, exceptions, presetIndex, summary] = await Promise.all([
      getOptional<TwinValidationRow[]>('twin_validation.json'), getOptional<TwinTimelineRow[]>('twin_timeline.json'),
      getOptional<TwinFrontierRow[]>('twin_frontier.json'), getOptional<TwinExceptionRow[]>('twin_exceptions.json'),
      getOptional<TwinPresetIndex>('twin/presets.json'), getJson<Summary>('summary.json'),
    ])
    return { validation, timeline, frontier, exceptions, presetIndex, summary }
  })
  return (
    <Async state={state}>
      {({ validation, timeline, frontier, exceptions, presetIndex, summary }) =>
        !validation || !presetIndex ? (
          <Problem message="The inventory twin tables are not in this build. Run the pipeline and `npm run data`." />
        ) : (
          <>
            <div className="question" style={{ marginTop: 18 }}>
              The twin replays the whole chain (supplier → warehouse → stores) day by day for every product, so you can try a policy, a demand spike or a supply problem
              on screen first. Starting stock, lead times, margins and holding costs are assumptions: the M5 data has none of them.
            </div>
            <Validation rows={validation} summary={summary} />
            {timeline && exceptions && <Exceptions timeline={timeline} exceptions={exceptions} />}
            <WhatIf index={presetIndex} />
            {frontier && <Frontier rows={frontier} />}
          </>
        )
      }
    </Async>
  )
}

/* ------------------------------------------------------------------ 1. can we trust it */

function Validation({ rows, summary }: { rows: TwinValidationRow[]; summary: Summary }) {
  const network = rows.filter((r) => r.store_id === 'all' && r.policy === 'forecast_reorder')
  const folds = [...new Set(network.map((r) => r.fold))].sort()
  const [fold, setFold] = useState(folds.at(-1) ?? '')
  const shown = network.filter((r) => r.fold === fold)
  const allInBand = network.filter((r) => r.in_band).length
  const miss = (metric: string) => mean(network.filter((r) => r.metric === metric).map((r) => Math.abs(r.predicted - r.realised)))
  const ratio = (metric: string) => mean(network.filter((r) => r.metric === metric).map((r) => r.predicted)) / mean(network.filter((r) => r.metric === metric).map((r) => r.realised))
  const gap = summary.twin?.fill_rate_gap
  return (
    <section>
      <h2>1 · Can we trust it?</h2>
      <p className="answer">
        Service levels: yes, roughly. Across {folds.length} windows the twin’s fill-rate prediction is off by {points(miss('fill_rate'))} on average and in-stock by {points(miss('in_stock_pct'))}.
        Its 90% bands, though, are too narrow: the realised result fell inside in only {allInBand} of {network.length} checks.
      </p>
      <p className="muted">
        Dollar losses are overstated about {ratio('lost_sales_value').toFixed(1)}×: predicted demand runs {signedPct(ratio('units_demanded') - 1, 0)} above what sold, and what sold is itself held down by real
        stockouts in the M5 data. Lost sales is a tail quantity, so a small demand overshoot moves it a lot. Treat dollar figures as relative (policy vs policy), not absolute.
        {gap !== undefined && <> Latest fill-rate gap {points(gap)}; the retraining check alerts above 5.0 pts.</>}
      </p>
      <p className="muted small">
        Each row: the twin simulates the window from forecast and calibration data only (50 random futures), then the actual sales are replayed through the same policy.
        The bar is the 90% band of the simulated futures; the dot is what actually happened.
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
            <thead><tr><th>Window</th><th className="text">Store</th><th className="text">Policy</th><th className="text">Metric</th><th>Predicted</th><th>Band</th><th>Realised</th><th className="text">Inside</th></tr></thead>
            <tbody>
              {rows.map((r, i) => {
                const kind = r.metric.includes('value') ? 'usd' : r.metric.includes('units') ? 'units' : 'pct'
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
      <svg viewBox="0 0 100 30" preserveAspectRatio="none" role="img" aria-label={`${label}: band ${fmt(kind, row.lower)} to ${fmt(kind, row.upper)}, realised ${fmt(kind, row.realised)}`}>
        <line x1="0" x2="100" y1="15" y2="15" stroke="var(--grid)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        <rect x={scale(row.lower)} width={Math.max(scale(row.upper) - scale(row.lower), 0.6)} y="9" height="12" rx="2" fill="var(--c1)" opacity="0.35" />
        <line x1={scale(row.realised)} x2={scale(row.realised)} y1="3" y2="27" stroke={row.in_band ? 'var(--good)' : 'var(--bad)'} strokeWidth="3" vectorEffect="non-scaling-stroke" />
      </svg>
      <div>
        <span className={`pill ${row.in_band ? 'in' : 'out'}`}>{row.in_band ? 'inside band' : 'outside band'}</span>{' '}
        <span className="num">realised {fmt(kind, row.realised)}</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 2. where will we run out */

function Exceptions({ timeline, exceptions }: { timeline: TwinTimelineRow[]; exceptions: TwinExceptionRow[] }) {
  const stores = [...new Set(timeline.map((r) => r.store_id))].sort()
  const [store, setStore] = useState(stores[0])
  const depts = [...new Set(timeline.filter((r) => r.store_id === store).map((r) => r.dept_id))].sort()
  const [dept, setDept] = useState(depts[0])
  const activeDept = depts.includes(dept) ? dept : depts[0]
  const [onlyStore, setOnlyStore] = useState(false)
  const series = timeline.filter((r) => r.store_id === store && r.dept_id === activeDept)
  const list = exceptions.filter((r) => !onlyStore || r.store_id === store).slice(0, 20)
  const atRisk = exceptions.filter((r) => r.stockout_probability >= 0.5).length
  return (
    <section>
      <h2>2 · Where will we run out?</h2>
      <p className="answer">
        {atRisk} of the {exceptions.length} highest-impact products are more likely than not to stock out within a week if reordering stays on last week’s sales.
        Together they put about {usd(exceptions.reduce((a, r) => a + r.expected_lost_value, 0))} of sales at risk.
      </p>
      <div className="controls"><Segmented label="Show" value={onlyStore ? 'store' : 'all'} options={[{ value: 'all', label: 'All stores' }, { value: 'store', label: `Only ${store}` }]} onChange={(v) => setOnlyStore(v === 'store')} /></div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Item</th><th className="text">Store</th><th className="text">Dept</th><th>Chance of stockout (7 days)</th><th>Expected lost units</th><th>Expected lost sales</th></tr></thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.series_id}><td>{r.item_id}</td><td className="text">{r.store_id}</td><td className="text">{r.dept_id}</td>
                <td className={`num ${r.stockout_probability >= 0.5 ? 'bad' : ''}`}>{pct(r.stockout_probability, 0)}</td><td className="num">{r.expected_lost_units.toFixed(1)}</td><td className="num">{usd(r.expected_lost_value)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend-note">Current practice (last-week reorder), 20 random futures per store, ranked by expected lost sales. Starting stock comes from replaying the last two weeks.</p>
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

/* ------------------------------------------------------------------ 3. what if */

type Shown = { label: string; detail: string; result: TwinResult }

function WhatIf({ index }: { index: TwinPresetIndex }) {
  const [store, setStore] = useState(index.stores[0])
  const [policy, setPolicy] = useState(index.policies[0])
  const [preset, setPreset] = useState(Object.keys(index.presets)[0])
  const [custom, setCustom] = useState<Shown | undefined>()
  const [useCustom, setUseCustom] = useState(false)
  const presets = useData(() => getJson<TwinPresets>(`twin/presets/${store}.json`), [store])
  useEffect(() => {
    const warm = () => warmSimulator()
    const idle = (window as unknown as { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback
    const handle = idle ? idle(warm) : window.setTimeout(warm, 2500)
    return () => { if (!idle) window.clearTimeout(handle) }
  }, [])
  const shown: Shown | undefined = useMemo(() => {
    if (useCustom && custom) return custom
    const result = presets.data?.results[`${preset}|${policy}`]
    return result ? { label: index.presets[preset].label, detail: index.presets[preset].detail, result } : undefined
  }, [useCustom, custom, presets.data, preset, policy, index])

  return (
    <section>
      <h2>3 · What if…?</h2>
      <p className="answer">
        {shown?.result.scenario
          ? <>{shown.label}: lost sales {signedUsd(shown.result.scenario.kpis.lost_sales_value.mean - shown.result.baseline.kpis.lost_sales_value.mean)} over the 4 weeks in {store}, and fill rate moves {points(shown.result.scenario.kpis.fill_rate.mean - shown.result.baseline.kpis.fill_rate.mean)}.</>
          : 'Pick a scenario to compare it with business as usual.'}
      </p>
      <div className="controls">
        <Select label="Store" value={store} options={index.stores.map((s) => ({ value: s, label: s }))} onChange={(v) => { setStore(v); setUseCustom(false) }} />
        <Segmented label="Reordering policy" value={policy} options={index.policies.map((p) => ({ value: p, label: p === 'forecast_reorder' ? 'Forecast-based' : 'Last-week (current)' }))} onChange={(v) => { setPolicy(v); setUseCustom(false) }} />
      </div>
      <div className="presets">
        {Object.entries(index.presets).map(([key, value]) => (
          <button key={key} type="button" className="preset" aria-pressed={!useCustom && preset === key} onClick={() => { setPreset(key); setUseCustom(false) }}>
            <strong>{value.label}</strong><span>{value.detail}</span>
          </button>
        ))}
        <button type="button" className="preset" aria-pressed={useCustom} onClick={() => custom && setUseCustom(true)} disabled={!custom} style={{ opacity: custom ? 1 : 0.6 }}>
          <strong>Your scenario</strong><span>{custom ? custom.detail : 'Set one below and press Run'}</span>
        </button>
      </div>
      <Async state={presets}>{(data) => (shown ? <ScenarioResult shown={shown} dates={data.dates} /> : <Loading />)}</Async>
      <CustomForm index={index} store={store} policy={policy} onResult={(result) => { setCustom(result); setUseCustom(true) }} />
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

function ScenarioResult({ shown, dates }: { shown: Shown; dates: string[] }) {
  const { result } = shown
  const outcome = result.scenario ?? result.baseline
  const depts = outcome.timeline.dept
  const [dept, setDept] = useState(depts[0])
  const column = Math.max(depts.indexOf(dept), 0)
  const data = dates.map((date, day) => ({
    date, onHand: outcome.timeline.on_hand[day]?.[column] ?? 0, demand: outcome.timeline.demand[day]?.[column] ?? 0, lost: outcome.timeline.lost[day]?.[column] ?? 0,
    baseline: result.scenario ? result.baseline.timeline.on_hand[day]?.[column] : undefined,
  }))
  return (
    <>
      <div className="grid kpis" style={{ marginTop: 12 }}>
        <Delta label="Fill rate" kpis={outcome.kpis} base={result.baseline.kpis} metric="fill_rate" kind="pct" goodWhen="up" />
        <Delta label="In-stock rate" kpis={outcome.kpis} base={result.baseline.kpis} metric="in_stock_pct" kind="pct" goodWhen="up" />
        <Delta label="Lost sales" kpis={outcome.kpis} base={result.baseline.kpis} metric="lost_sales_value" kind="usd" goodWhen="down" />
        <Delta label="Inventory on hand" kpis={outcome.kpis} base={result.baseline.kpis} metric="inventory_value" kind="usd" goodWhen="neutral" />
      </div>
      <div className="controls"><Select label="Department" value={depts[column]} options={depts.map((d) => ({ value: d, label: d }))} onChange={setDept} /></div>
      <TimelineChart data={data} />
      <p className="legend-note">
        Average of {result.reps} simulated futures sharing the same random demand, so differences come from the scenario alone. {shown.detail}.
        Policy: {POLICY_LABELS[result.policy]}.
      </p>
    </>
  )
}

function CustomForm({ index, store, policy, onResult }: { index: TwinPresetIndex; store: string; policy: string; onResult: (shown: Shown) => void }) {
  const [demandScale, setDemandScale] = useState(1.3)
  const [category, setCategory] = useState('FOODS')
  const [start, setStart] = useState(7)
  const [length, setLength] = useState(7)
  const [dcFactor, setDcFactor] = useState(1)
  const [delay, setDelay] = useState(0)
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
    }
    const request: TwinRequest = { policy, service: service === 'current' ? 'current' : Number(service), rationing: 'days_of_cover', scenario, reps, seed: 1 }
    const parts = [
      demandScale !== 1 ? `${category === 'ALL' ? 'All' : category} demand ×${demandScale.toFixed(2)} for days ${days[0] + 1}–${days[1]}` : '',
      dcFactor < 1 ? `warehouse receives ${Math.round((1 - dcFactor) * 100)}% less` : '', delay > 0 ? `supplier ${delay} days late` : '',
    ].filter(Boolean)
    try {
      const result = await runSimulation(store, request, setStage)
      onResult({ label: 'Your scenario', detail: parts.join(', ') || 'no change from baseline', result })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStage(undefined)
      busy.current = false
    }
  }
  return (
    <details>
      <summary>Adjust: build your own scenario</summary>
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
          <Select label="Service target" value={service} options={[{ value: 'current', label: 'Segment targets (current)' }, ...index.service_grid.map((s) => ({ value: String(s), label: pct(s, 0) }))]} onChange={setService} />
          <Select label="Simulated futures" value={String(reps)} options={[10, 20, 50].map((n) => ({ value: String(n), label: String(n) }))} onChange={(v) => setReps(Number(v))} />
        </div>
        <button type="button" className="primary" onClick={run} disabled={!!stage}>{stage ? 'Running…' : 'Run simulation'}</button>
        {stage && <p className="muted small" role="status">{STAGE_LABELS[stage]}</p>}
        {error && <div className="callout warn" role="alert">{error}</div>}
        <p className="legend-note">Runs on your device: the Python simulator is loaded into the page the first time (about 12 MB, then cached). Nothing is sent to a server.</p>
      </div>
    </details>
  )
}

/* ------------------------------------------------------------------ 4. which policy */

function Frontier({ rows }: { rows: TwinFrontierRow[] }) {
  const categories = [...new Set(rows.map((r) => r.category))].sort()
  const grid = rows.filter((r) => r.service_level !== null)
  const recommended = rows.filter((r) => r.recommended)
  const total = (r: TwinFrontierRow[]) => r.reduce((a, row) => a + row.total_cost, 0)
  const current = rows.filter((r) => r.service_level === null)
  const saving = total(current) - total(recommended)
  return (
    <section>
      <h2>4 · Which policy should we run?</h2>
      <p className="answer">
        {recommended.map((r) => `${r.category} ${pct(r.service_level ?? 0, 0)}`).join(' · ')} is the cheapest service target per category
        {current.length > 0 && <>; versus today’s segment targets that changes lost margin + holding cost by {signedUsd(-saving)} over 4 weeks across the four stores</>}.
      </p>
      <p className="muted small">
        Each line sweeps the service target from 80% to 99% for one category. More stock means fewer lost sales but more cash tied up. Cost = lost margin + holding cost
        (assumed margin by category, 0.5% of value per week to hold stock).
      </p>
      <ChartBox size="tall">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 40, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis type="number" dataKey="inventory_value" name="Inventory" {...axisProps} tickFormatter={(v: number) => usd(v)} domain={['auto', 'auto']} label={{ value: 'Average inventory on hand', position: 'insideBottom', offset: -12, fill: 'var(--muted)', fontSize: 12 }} />
          <YAxis type="number" dataKey="lost_sales_value" name="Lost sales" {...axisProps} width={60} tickFormatter={(v: number) => usd(v)} label={{ value: 'Lost sales (4 weeks)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
          <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: '3 3' }} formatter={(v: unknown, name: string) => (name === 'Inventory' || name === 'Lost sales' ? usd(Number(v)) : String(v))} labelFormatter={() => ''} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} verticalAlign="bottom" />
          {categories.map((category) => (
            <Scatter key={category} name={category} data={grid.filter((r) => r.category === category)} fill={CATEGORY_COLORS[category]} line={{ stroke: CATEGORY_COLORS[category], strokeWidth: 1.6 }} isAnimationActive={false} />
          ))}
          <Scatter name="Recommended" data={recommended} fill="var(--ink)" shape="star" legendType="star" isAnimationActive={false} />
          <Scatter name="Current segment targets" data={current} fill="var(--muted)" shape="diamond" legendType="diamond" isAnimationActive={false} />
        </ScatterChart>
      </ChartBox>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Category</th><th>Recommended service</th><th>Fill rate</th><th>Inventory</th><th>Lost sales</th><th>Total cost</th><th>Current total cost</th></tr></thead>
          <tbody>
            {recommended.map((r) => {
              const now = current.find((c) => c.category === r.category)
              return (
                <tr key={r.category}><td>{r.category}</td><td className="num">{pct(r.service_level ?? 0, 0)}</td><td className="num">{pct(r.fill_rate)}</td><td className="num">{usd(r.inventory_value)}</td><td className="num">{usd(r.lost_sales_value)}</td>
                  <td className="num">{usd(r.total_cost)}</td><td className="num">{now ? usd(now.total_cost) : '–'}</td></tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
