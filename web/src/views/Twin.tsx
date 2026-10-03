import { useEffect, useMemo, useRef, useState } from 'react'
import { Area, Bar, CartesianGrid, ComposedChart, Legend, Line, LineChart, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson, getOptional } from '../data/load'
import type {
  InventoryHealthRow, KpiSet, Summary, TwinExceptionRow, TwinResponseRow, TwinFrontierRow, TwinPolicyCurveRow, TwinPresetIndex, TwinRequest, TwinResult, TwinScenario, TwinStressRow, TwinTimelineRow,
  TwinValidationRow,
} from '../data/types'
import { Async, ChartBox, CsvButton, Kpi, Problem, Segmented, Select, type Tone } from '../components/ui'
import { useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { POLICY_LABELS, RATIONING_LABELS, num, pct, points, signedPct, signedUsd, usd } from '../lib/format'
import { bestResponse, mean } from '../lib/metrics'
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
    const [validation, timeline, frontier, exceptions, presetIndex, summary, curve, stress, responses, health] = await Promise.all([
      getOptional<TwinValidationRow[]>('twin_validation.json'), getOptional<TwinTimelineRow[]>('twin_timeline.json'),
      getOptional<TwinFrontierRow[]>('twin_frontier.json'), getOptional<TwinExceptionRow[]>('twin_exceptions.json'),
      getOptional<TwinPresetIndex>('twin/presets.json'), getJson<Summary>('summary.json'), getOptional<TwinPolicyCurveRow[]>('twin_policy_curve.json'),
      getOptional<TwinStressRow[]>('twin_stress.json'), getOptional<TwinResponseRow[]>('twin_responses.json'), getOptional<InventoryHealthRow[]>('inventory_health.json'),
    ])
    return { validation, timeline, frontier, exceptions, presetIndex, summary, curve, stress, responses, health }
  })
  useEffect(() => {
    const section = hashParams().get('section')
    if (section && state.data) document.getElementById(section)?.scrollIntoView({ block: 'start' })
  }, [state.data])
  return (
    <Async state={state}>
      {({ validation, timeline, frontier, exceptions, presetIndex, summary, curve, stress, responses, health }) =>
        !validation || !presetIndex ? (
          <Problem message="The inventory twin tables are not in this build. Run the pipeline and `npm run data`." />
        ) : (
          <>
            <div className="question" style={{ marginTop: 18 }}>
              The simulator replays the chain (supplier → one warehouse → four stores) day by day for every product. The warehouse holds stock per product and shares it between the
              stores, so you can try a policy, a demand spike or a supply problem on screen first. Costs, lead times and starting stock are assumptions: the M5 data has none of them.
            </div>
            <WhatIf index={presetIndex} />
            {responses && responses.length > 0 && <Responses rows={responses} summary={summary} />}
            {frontier && <Frontier rows={frontier} summary={summary} />}
            {curve && curve.length > 0 && <PolicyCurve rows={curve} summary={summary} />}
            {stress && <Rationing rows={stress} />}
            {timeline && exceptions && <Exceptions timeline={timeline} exceptions={exceptions} health={health} />}
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
      <Warehouse result={result} dates={dates} scope={scope} />
      <div className="controls"><Select label="Department" value={active} options={depts.map((d) => ({ value: d, label: d }))} onChange={setDept} /></div>
      <TimelineChart data={data} />
      <p className="legend-note">
        {scope === 'all' ? 'All four stores' : scope}, average of {result.reps} simulated futures sharing the same random demand, so differences come from the scenario alone. {shown.detail}.
        Policy: {POLICY_LABELS[result.policy]}.
      </p>
    </>
  )
}

/** The warehouse is where supplier problems land: its stock, what is on order, and how much of the stores' orders it filled. */
function Warehouse({ result, dates, scope }: { result: TwinResult; dates: string[]; scope: string }) {
  const outcome = result.scenario ?? result.baseline
  const now = outcome.kpis.all, base = result.baseline.kpis.all
  if (!outcome.dc || !now.dc_fill_rate) return null
  const data = dates.map((date, day) => ({ date, stock: outcome.dc!.on_hand[day], onOrder: outcome.dc!.on_order[day], baseline: result.scenario ? result.baseline.dc?.on_hand[day] : undefined }))
  const premium = now.response_cost?.mean ?? 0
  return (
    <details open={scope === 'all'}>
      <summary>Warehouse: stock, inbound orders and fill rate</summary>
      <div className="grid kpis" style={{ marginTop: 8 }}>
        <Delta label="Warehouse fill rate" kpis={now} base={base} metric="dc_fill_rate" kind="pct" goodWhen="up" />
        <Delta label="Warehouse stock (at cost)" kpis={now} base={base} metric="dc_inventory_value" kind="usd" goodWhen="neutral" />
        <Delta label="On order from supplier" kpis={now} base={base} metric="dc_on_order_value" kind="usd" goodWhen="neutral" />
        {premium > 0 && <Kpi label="Response premium" value={usd(premium)} tone="flat" note="Expedite or second-supplier cost over the 4 weeks" />}
      </div>
      <ChartBox size="short">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="date" {...axisProps} tickFormatter={(d: string) => d.slice(5)} minTickGap={28} />
          <YAxis {...axisProps} width={60} tickFormatter={(v: number) => usd(v)} />
          <Tooltip {...tooltipStyle} formatter={(v: unknown) => usd(Number(v))} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {data.some((d) => d.baseline !== undefined) && <Line dataKey="baseline" name="Warehouse stock (baseline)" stroke="var(--muted)" strokeDasharray="4 3" strokeWidth={1.8} dot={false} isAnimationActive={false} />}
          <Line dataKey="stock" name="Warehouse stock" stroke={SERIES[0]} strokeWidth={2.4} dot={false} isAnimationActive={false} />
          <Line dataKey="onOrder" name="On order" stroke={SERIES[2]} strokeWidth={1.8} dot={false} isAnimationActive={false} />
        </LineChart>
      </ChartBox>
      <p className="legend-note">One warehouse serves all four stores, so these figures are network-wide whichever store is shown. Fill rate = share of store orders the warehouse shipped.</p>
    </details>
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

const RESPONSE_LABELS = {
  none: 'None', planned: 'Spike is in the forecast', prebuild: 'Pre-build a week of warehouse stock', expedite: 'Expedite half of late orders (+20%)', backup: 'Second supplier covers half a cut (+10%)',
} as const
const RESPONSE_SPECS: Record<keyof typeof RESPONSE_LABELS, TwinScenario> = {
  none: {}, planned: { planned: true }, prebuild: { prebuild_days: 7 }, expedite: { expedite_share: 0.5, premium: 0.2 }, backup: { backup_share: 0.5, premium: 0.1 },
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
  const [response, setResponse] = useState<'none' | 'planned' | 'prebuild' | 'expedite' | 'backup'>('none')
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
      ...RESPONSE_SPECS[response],
    }
    const request: TwinRequest = { policy, service: service === 'current' ? 'current' : Number(service), rationing, scenario, reps, seed: 1 }
    const delayText = { once: `this week’s supplier delivery ${delay} days late`, replan: `supplier ${delay} days slower, planner adjusts after a week`, lasting: `supplier ${delay} days slower, never planned for` }[delayKind]
    const parts = [
      demandScale !== 1 ? `${category === 'ALL' ? 'All' : category} demand ×${demandScale.toFixed(2)} for days ${days[0] + 1}–${days[1]}` : '',
      dcFactor < 1 ? `warehouse receives ${Math.round((1 - dcFactor) * 100)}% less` : '', delay > 0 ? delayText : '',
      rationing !== 'days_of_cover' ? RATIONING_LABELS[rationing].toLowerCase() : '', response !== 'none' ? `response: ${RESPONSE_LABELS[response].toLowerCase()}` : '', service !== 'current' ? `${pct(Number(service), 1).replace('.0%', '%')} service target` : '',
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
          <Select label="Planner response" value={response} options={Object.entries(RESPONSE_LABELS).map(([value, label]) => ({ value: value as typeof response, label }))} onChange={setResponse} />
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

/* ------------------------------------------------------------------ 1b. what would we do about it */

const SCENARIO_TITLES: Record<string, string> = {
  supplier_delay: 'Supplier slows down (7 → 14 days)', late_shipment: 'One delivery 7 days late', dc_cut: 'Warehouse gets 30% less for two weeks', event_spike: 'Food demand +50% for a week',
}

/** One sentence for the whole table: which responses pay, and if none clearly does, the closest call. */
function responseVerdict(rows: TwinResponseRow[], scenarios: string[]): string {
  const options = rows.filter((r) => r.response !== 'none' && scenarios.includes(r.scenario))
  const paying = scenarios.map((key) => bestResponse(rows, key)).filter((r): r is TwinResponseRow => !!r)
  if (paying.length) return paying.map((r) => `${SCENARIO_TITLES[r.scenario]}: ${r.label.toLowerCase()} pays (${signedUsd(r.net_benefit)} net).`).join(' ')
  const closest = [...options].sort((a, b) => b.net_benefit - a.net_benefit)[0]
  const none = rows.find((r) => r.scenario === closest?.scenario && r.response === 'none')
  return `Every response cuts the loss, but with grocery margins none clearly pays for its premiums and extra holding. The closest is to ${closest.label.charAt(0).toLowerCase()}${closest.label.slice(1)} (${SCENARIO_TITLES[closest.scenario].toLowerCase()}): the extra loss falls from ${usd(none?.added_lost_sales ?? NaN)} to ${usd(closest.added_lost_sales)}, roughly break-even (${signedUsd(closest.net_benefit)}).`
}

function Responses({ rows, summary }: { rows: TwinResponseRow[]; summary: Summary }) {
  const scenarios = [...new Set(rows.map((r) => r.scenario))]
  const ordered = Object.keys(SCENARIO_TITLES).filter((k) => scenarios.includes(k))
  const bias = summary.twin?.lost_sales_bias
  return (
    <section id="responses">
      <h2>What would we do about it?</h2>
      <p className="answer">{responseVerdict(rows, ordered)}</p>
      <p className="muted small">
        Each response is simulated against the same futures as the disruption. Cost = expedite or second-supplier premium (assumed) + extra holding at the stores and warehouse.
        Net benefit = lost margin recovered (corrected for the simulator’s {bias ? `${bias.toFixed(1)}× ` : ''}overstatement of lost sales) minus that cost; a response is worth it when its 90% range is above zero.
      </p>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th className="text">Response</th><th>Extra lost sales, 4 wk</th><th>Cost of response</th><th>Net benefit (90% range)</th><th>Warehouse fill</th></tr></thead>
          <tbody>
            {ordered.flatMap((key) => [
              <tr key={key} className="group-row"><td className="text" colSpan={5}><strong>{SCENARIO_TITLES[key]}</strong></td></tr>,
              ...rows.filter((r) => r.scenario === key).map((r) => {
                const worth = r.response !== 'none' && r.net_benefit_lower > 0
                return (
                  <tr key={`${key}${r.response}`}>
                    <td className="text" data-label="Response" style={{ whiteSpace: 'normal', minWidth: 280 }}>{r.label}{r.detail ? <div className="muted small">{r.detail}</div> : null}</td>
                    <td className="num" data-label="Extra lost sales, 4 wk">{usd(r.added_lost_sales)}</td>
                    <td className="num" data-label="Cost of response">{r.response === 'none' ? '–' : usd(r.response_cost)}</td>
                    <td className={`num ${worth ? 'good' : r.response === 'none' ? '' : 'flat'}`} data-label="Net benefit (90% range)">{r.response === 'none' ? '–' : `${signedUsd(r.net_benefit)} (${usd(r.net_benefit_lower)} to ${usd(r.net_benefit_upper)})`}</td>
                    <td className="num" data-label="Warehouse fill">{pct(r.dc_fill_rate)}</td>
                  </tr>
                )
              }),
            ])}
          </tbody>
        </table>
      </div>
      <p className="legend-note">Pre-building needs notice: it starts two weeks before the disruption. Premiums (expedite +20%, second supplier +10% of unit cost) are assumptions; change them in <code>twin_runs.RESPONSES</code>.</p>
    </section>
  )
}

/* ------------------------------------------------------------------ 2. which policy */

const ASSUMPTION_SHORT: Record<string, string> = { every_unit: 'Every unmet unit lost', bias_corrected: 'Bias-corrected', shopper_response: 'Bias-corrected + shopper response' }

function Frontier({ rows, summary }: { rows: TwinFrontierRow[]; summary: Summary }) {
  const assumptions = [...new Set(rows.map((r) => r.assumption))]
  const [assumption, setAssumption] = useState(summary.twin?.service_assumption && assumptions.includes(summary.twin.service_assumption) ? summary.twin.service_assumption : assumptions[0])
  const shown = rows.filter((r) => r.assumption === assumption)
  const categories = [...new Set(shown.map((r) => r.category))].sort()
  const grid = shown.filter((r) => r.service_level !== null)
  const recommended = shown.filter((r) => r.recommended)
  const current = shown.filter((r) => r.service_level === null)
  const budget = shown.filter((r) => r.within_budget)
  const clear = recommended.filter((r) => r.clear_saving)
  const label = shown[0]?.assumption_label ?? ''
  const bias = summary.twin?.lost_sales_bias
  const best = (name: string, category: string) => rows.find((r) => r.assumption === name && r.category === category && r.recommended)
  const exportRows = rows.filter((r) => r.recommended || r.within_budget || r.service_level === null)
  return (
    <section id="policy">
      <h2>2 · Which service targets should we run?</h2>
      <p className="answer">
        {clear.length > 0
          ? <>{clear.map((r) => `${titleCase(r.category)} to ${r.label} (${r.inventory_change >= 0 ? '+' : '−'}${usd(Math.abs(r.inventory_change))} stock)`).join(', ')}: saves {usd(clear.reduce((a, r) => a + r.saving_vs_current, 0))} of lost margin + holding cost over 4 weeks.</>
          : <>Keep today’s segment targets: no single target is cheaper by more than simulation noise.</>}
        {recommended.filter((r) => !r.clear_saving).length > 0 && clear.length > 0 && <> {recommended.filter((r) => !r.clear_saving).map((r) => titleCase(r.category)).join(', ')}: no clear gain, keep as is.</>}
      </p>
      <p className="muted small">
        The answer depends on what a stockout really costs. The simulator counts every unmet unit as a lost sale, but its validation shows it overstates lost sales
        {bias ? <> about {bias.toFixed(1)}×</> : null} against what actually happened, and shoppers often substitute or come back: in the largest study of retail stockouts (Gruen &amp; Corsten, 2002)
        31% bought elsewhere and 9% didn’t buy, so only about 40% of stockouts lose the sale. Overstating that cost pushes the cheapest target up, because the optimal service level is the
        shortage cost over shortage + overstock cost. The default here corrects for the measured bias.
      </p>
      <div className="controls">
        <Segmented label="What a stockout costs" value={assumption} options={assumptions.map((a) => ({ value: a, label: ASSUMPTION_SHORT[a] ?? a }))} onChange={setAssumption} />
      </div>
      <ChartBox size="tall">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 40, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis type="number" dataKey="inventory_value" name="Inventory" {...axisProps} tickFormatter={(v: number) => usd(v)} domain={['auto', 'auto']} label={{ value: 'Average inventory (at cost)', position: 'insideBottom', offset: -12, fill: 'var(--muted)', fontSize: 12 }} />
          <YAxis type="number" dataKey="total_cost" name="Cost" {...axisProps} width={60} tickFormatter={(v: number) => usd(v)} domain={['auto', 'auto']} label={{ value: 'Lost margin + holding (4 wk)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
          <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: '3 3' }} formatter={(v: unknown, name: string) => (name === 'Inventory' || name === 'Cost' ? usd(Number(v)) : String(v))} labelFormatter={() => ''} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} verticalAlign="bottom" />
          {categories.map((category) => (
            <Scatter key={category} name={category} data={grid.filter((r) => r.category === category)} fill={CATEGORY_COLORS[category]} line={{ stroke: CATEGORY_COLORS[category], strokeWidth: 1.6 }} isAnimationActive={false} />
          ))}
          <Scatter name="Cheapest" data={recommended} fill="var(--ink)" shape="star" legendType="star" isAnimationActive={false} />
          <Scatter name="Current segment targets" data={current} fill="var(--muted)" shape="diamond" legendType="diamond" isAnimationActive={false} />
        </ScatterChart>
      </ChartBox>
      <p className="legend-note">{label}. Each line sweeps one category’s target from 80% to 99.8%; the low point of each curve is its cheapest target. Stock valued at cost; margins and holding rates are assumptions.</p>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th>Category</th><th>Cheapest target</th><th>Fill rate</th><th>Stock change</th><th>Saving, 4 wk (90% range)</th><th className="text">Verdict</th><th>If stock can’t rise</th></tr></thead>
          <tbody>
            {recommended.map((r) => {
              const capped = budget.find((b) => b.category === r.category)
              return (
                <tr key={r.category}>
                  <td data-label="Category">{r.category}</td><td className="num" data-label="Cheapest target">{r.label}</td><td className="num" data-label="Fill rate">{pct(r.fill_rate)}</td>
                  <td className="num" data-label="Stock change">{signedUsd(r.inventory_change)} ({signedPct(r.inventory_change / Math.max(r.inventory_value - r.inventory_change, 1), 0)})</td>
                  <td className="num" data-label="Saving, 4 wk (90% range)">{usd(r.saving_vs_current)} ({usd(r.saving_lower)} to {usd(r.saving_upper)})</td>
                  <td className={`text ${r.clear_saving ? 'good' : 'flat'}`} data-label="Verdict">{r.clear_saving ? 'Change' : 'Keep current'}{r.at_grid_edge ? ' · at range edge' : ''}</td>
                  <td className="num" data-label="If stock can’t rise">{capped ? `${capped.label}${capped.clear_saving ? ` · saves ${usd(capped.saving_vs_current)}` : ' · no clear gain'}` : 'today’s'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="table-head">
        <h3>Cheapest target under each assumption</h3>
        <CsvButton filename="service_targets.csv" rows={exportRows} columns={[
          { header: 'assumption', value: (r) => r.assumption }, { header: 'category', value: (r) => r.category },
          { header: 'option', value: (r) => (r.service_level === null ? 'current' : r.recommended && r.within_budget ? 'cheapest (within budget)' : r.recommended ? 'cheapest' : 'cheapest within budget') },
          { header: 'service_target', value: (r) => r.service_level ?? 'segment targets' }, { header: 'fill_rate', value: (r) => r.fill_rate },
          { header: 'inventory_at_cost', value: (r) => r.inventory_value }, { header: 'inventory_change', value: (r) => r.inventory_change },
          { header: 'lost_margin_4wk', value: (r) => r.lost_margin }, { header: 'holding_cost_4wk', value: (r) => r.holding_cost }, { header: 'total_cost_4wk', value: (r) => r.total_cost },
          { header: 'saving_4wk', value: (r) => r.saving_vs_current }, { header: 'saving_lower', value: (r) => r.saving_lower }, { header: 'saving_upper', value: (r) => r.saving_upper },
          { header: 'clear_saving', value: (r) => r.clear_saving },
        ]} />
      </div>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th>Category</th>{assumptions.map((a) => <th key={a}>{ASSUMPTION_SHORT[a] ?? a}</th>)}</tr></thead>
          <tbody>
            {categories.map((category) => (
              <tr key={category}>
                <td data-label="Category">{category}</td>
                {assumptions.map((a) => {
                  const row = best(a, category)
                  return <td key={a} className={`num ${row?.clear_saving ? '' : 'flat'}`} data-label={ASSUMPTION_SHORT[a] ?? a}>{row ? `${row.label} · ${signedUsd(row.inventory_change)} stock${row.clear_saving ? '' : ' · noise'}` : '–'}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend-note">“noise” = the saving’s 90% range includes zero, so keep today’s target. Raising a target only pays if nearly every stockout is a lost sale.</p>
    </section>
  )
}

const titleCase = (value: string) => value.charAt(0) + value.slice(1).toLowerCase()

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

const THRESHOLDS = [0, 25, 50, 100]

function Exceptions({ timeline, exceptions, health }: { timeline: TwinTimelineRow[]; exceptions: TwinExceptionRow[]; health?: InventoryHealthRow[] }) {
  const stores = [...new Set(timeline.map((r) => r.store_id))].sort()
  const [store, setStore] = useState(stores[0])
  const depts = [...new Set(timeline.filter((r) => r.store_id === store).map((r) => r.dept_id))].sort()
  const [dept, setDept] = useState(depts[0])
  const activeDept = depts.includes(dept) ? dept : depts[0]
  const [threshold, setThreshold] = useState(50)
  const [order, setOrder] = useState<'value' | 'chance'>('value')
  const series = timeline.filter((r) => r.store_id === store && r.dept_id === activeDept)
  const material = exceptions.filter((r) => r.expected_lost_value >= threshold || r.stockout_probability >= 0.5)
  const sorted = [...material].sort((a, b) => order === 'value' ? b.expected_lost_value - a.expected_lost_value : b.stockout_probability - a.stockout_probability || b.expected_lost_value - a.expected_lost_value)
  const total = exceptions.reduce((a, r) => a + r.expected_lost_value, 0)
  const materialTotal = material.reduce((a, r) => a + r.expected_lost_value, 0)
  const network = health?.find((h) => h.group_type === 'category' && h.group === 'all')
  const weeklySales = network ? network.sales_value / 4 : NaN
  const byDept = [...new Set(exceptions.map((r) => r.dept_id))].map((d) => {
    const rows = exceptions.filter((r) => r.dept_id === d)
    return { dept: d, products: rows.length, material: rows.filter((r) => material.includes(r)).length, value: rows.reduce((a, r) => a + r.expected_lost_value, 0) }
  }).sort((a, b) => b.value - a.value)
  return (
    <section id="exceptions">
      <h2>3 · Where will we run out this week?</h2>
      <p className="answer">
        {num(material.length)} of the {num(exceptions.length)} riskiest product-store pairs are material (at least {usd(threshold)} of sales at risk, or more likely than not to run out); they carry {usd(materialTotal)}
        {Number.isFinite(weeklySales) ? <>, {pct(materialTotal / weeklySales, 2)} of a week’s sales</> : null}.
      </p>
      <p className="muted small">
        Exception lists are only useful above a materiality threshold. The whole list of {num(exceptions.length)} carries {usd(total)}{Number.isFinite(weeklySales) ? ` (${pct(total / weeklySales, 1)} of weekly sales)` : ''}, mostly
        a few dollars per product, which is better handled by the department’s service target than item by item.
      </p>
      <div className="table-wrap stack">
        <table className="stack">
          <thead><tr><th className="text">Department</th><th>Products on the list</th><th>Material</th><th>Sales at risk (7 days)</th><th>Share of list</th></tr></thead>
          <tbody>
            {byDept.map((r) => (
              <tr key={r.dept}><td className="text" data-label="Department">{r.dept}</td><td className="num" data-label="Products on the list">{r.products}</td><td className="num" data-label="Material">{r.material}</td>
                <td className="num" data-label="Sales at risk (7 days)">{usd(r.value)}</td><td className="num" data-label="Share of list">{pct(r.value / Math.max(total, 1e-9), 0)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="controls">
        <Segmented label="Materiality threshold" value={String(threshold)} options={THRESHOLDS.map((t) => ({ value: String(t), label: t === 0 ? 'Show all' : `≥ ${usd(t)}` }))} onChange={(v) => setThreshold(Number(v))} />
        <Segmented label="Sort by" value={order} options={[{ value: 'value', label: 'Sales at risk' }, { value: 'chance', label: 'Chance of stockout' }]} onChange={setOrder} />
        <CsvButton filename="stockout_watch_list.csv" label="Download watch-list" rows={sorted} columns={[
          { header: 'item_id', value: (r) => r.item_id }, { header: 'store_id', value: (r) => r.store_id }, { header: 'dept_id', value: (r) => r.dept_id },
          { header: 'stockout_probability_7d', value: (r) => r.stockout_probability }, { header: 'expected_lost_units_7d', value: (r) => r.expected_lost_units },
          { header: 'sales_at_risk_7d', value: (r) => r.expected_lost_value }, { header: 'policy', value: (r) => r.policy },
        ]} />
      </div>
      <div className="table-wrap stack" style={{ maxHeight: 460 }}>
        <table className="stack">
          <thead><tr><th>Item</th><th className="text">Store</th><th className="text">Dept</th><th>Chance of stockout (7 days)</th><th>Expected lost units</th><th>Sales at risk</th></tr></thead>
          <tbody>
            {sorted.slice(0, 50).map((r) => (
              <tr key={r.series_id}><td data-label="Item">{r.item_id}</td><td className="text" data-label="Store">{r.store_id}</td><td className="text" data-label="Dept">{r.dept_id}</td>
                <td className={`num ${r.stockout_probability >= 0.5 ? 'bad' : ''}`} data-label="Chance of stockout (7 days)">{pct(r.stockout_probability, 0)}</td>
                <td className="num" data-label="Expected lost units">{r.expected_lost_units.toFixed(1)}</td><td className="num" data-label="Sales at risk">{usd(r.expected_lost_value)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend-note">Showing {Math.min(sorted.length, 50)} of {num(sorted.length)} above the threshold (the download has them all). Forecast-based reordering, all four stores and the warehouse simulated together; starting stock from replaying the last two weeks.</p>
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
      <p className="muted">
        <strong>It assumes perfect store execution.</strong> Every unit that reaches a store is on the shelf. In practice two-thirds to three-quarters of retail stockouts start in the store
        (ordering, forecasting and shelf replenishment are the largest causes), and studies across 29 countries put the average out-of-stock rate near 8% (Gruen &amp; Corsten, 2002).
        The simulated in-stock rate of about {pct(mean(network.filter((r) => r.metric === 'in_stock_pct').map((r) => r.realised)), 1)} is therefore a ceiling for what shoppers would see, not a prediction of it.
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
