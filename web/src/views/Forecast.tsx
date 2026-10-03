import { useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson } from '../data/load'
import type { ColumnarForecast, ForecastIndex, LevelForecast, Manifest, Summary } from '../data/types'
import { Async, ChartBox, Question, Segmented, Select } from '../components/ui'
import { useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { LEVEL_LABELS, LEVELS, METHOD_LABELS, nodeLabel } from '../lib/format'

interface Row { date: string; actual?: number; [key: string]: number | string | [number, number] | undefined }

export default function Forecast() {
  const [level, setLevel] = useState<string>('store')
  const [store, setStore] = useState('CA_1')
  const [dept, setDept] = useState('FOODS_1')
  const [node, setNode] = useState('')
  const [methods, setMethods] = useState<string[] | null>(null)
  const [show80, setShow80] = useState(true)
  const meta = useData(async () => ({ index: await getJson<ForecastIndex>('forecast/index.json'), summary: await getJson<Summary>('summary.json'), manifest: await getJson<Manifest>('manifest.json') }))
  const itemFile = meta.data?.index.item_files.find((file) => file.store === store && file.dept === dept)?.file
  const body = useData<{ level?: LevelForecast; items?: ColumnarForecast }>(
    () => (level === 'item' ? (itemFile ? getJson<ColumnarForecast>(`forecast/${itemFile}`).then((items) => ({ items })) : Promise.resolve({})) : getJson<LevelForecast>(`forecast/${level}.json`).then((lv) => ({ level: lv }))),
    [level, itemFile],
  )
  return (
    <Async state={meta}>
      {({ index, summary }) => {
        const served = summary.served_method
        const palette = Object.fromEntries([served, ...Object.keys(METHOD_LABELS).filter((m) => m !== served)].map((method, i) => [method, SERIES[i]]))
        const activeMethods = methods ?? [served, 'base']
        const stores = [...new Set(index.item_files.map((f) => f.store))].sort()
        const depts = [...new Set(index.item_files.filter((f) => f.store === store).map((f) => f.dept))].sort()
        const nodes = level === 'item' ? (body.data?.items?.ids ?? []) : (index.levels[level] ?? [])
        const current = nodes.includes(node) ? node : nodes[0] ?? ''
        return (
          <>
            <Question>how much will sell, anywhere in the hierarchy, and how wide is the range of outcomes?</Question>
            <div className="controls">
              <Select label="Level" value={level} options={LEVELS.map((l) => ({ value: l, label: LEVEL_LABELS[l] }))} onChange={(v) => { setLevel(v); setNode('') }} />
              {level === 'item' && <Select label="Store" value={store} options={stores.map((s) => ({ value: s, label: s }))} onChange={(v) => { setStore(v); setNode('') }} />}
              {level === 'item' && <Select label="Department" value={dept} options={depts.map((d) => ({ value: d, label: d }))} onChange={(v) => { setDept(v); setNode('') }} />}
              <Select label="Node" value={current} options={nodes.map((n) => ({ value: n, label: level === 'item' ? n.match(/item_id=([^|]+)/)?.[1] ?? n : nodeLabel(n) }))} onChange={setNode} />
              <Segmented label="80% interval" value={show80 ? 'on' : 'off'} options={[{ value: 'on', label: 'Show' }, { value: 'off', label: 'Hide' }]} onChange={(v) => setShow80(v === 'on')} />
            </div>
            {level !== 'item' && (
              <div className="controls">
                <div className="field">
                  <span>Backtest methods (forecasts made at each fold origin, compared with what sold)</span>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px' }}>
                    {Object.keys(METHOD_LABELS).map((method) => (
                      <label key={method} style={{ color: 'var(--ink)', display: 'flex', gap: 5, alignItems: 'center' }}>
                        <input type="checkbox" checked={activeMethods.includes(method)} onChange={(e) => setMethods(e.target.checked ? [...activeMethods, method] : activeMethods.filter((x) => x !== method))} />
                        <span style={{ borderBottom: `3px solid ${palette[method]}` }}>{METHOD_LABELS[method]}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            )}
            <Async state={body}>
              {(data) => {
                const prod = data.level?.production ?? data.items
                if (!prod || !current) return <p className="muted">No forecast for this selection.</p>
                const idx = prod.ids.indexOf(current)
                const rows: Row[] = []
                if (data.level) {
                  const byDate = new Map<string, Row>()
                  for (const point of data.level.backtest.filter((p) => p.series_id === current)) {
                    const row = byDate.get(point.date) ?? { date: point.date }
                    if (point.method === 'base') row.actual = point.actual
                    if (activeMethods.includes(point.method)) row[point.method] = point.forecast
                    byDate.set(point.date, row)
                  }
                  rows.push(...[...byDate.values()].sort((a, b) => a.date.localeCompare(b.date)))
                }
                prod.dates.forEach((date, d) => {
                  const row: Row = { date, production: prod.forecast[idx][d], band95: [prod.lower_95[idx][d], prod.upper_95[idx][d]] }
                  if (prod.lower_80 && prod.upper_80 && show80) row.band80 = [prod.lower_80[idx][d], prod.upper_80[idx][d]]
                  rows.push(row)
                })
                return (
                  <>
                    <h3>{level === 'item' ? current.match(/item_id=([^|]+)/)?.[1] : nodeLabel(current)}</h3>
                    <ChartBox size="tall">
                      <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                        <CartesianGrid {...gridProps} />
                        <XAxis dataKey="date" {...axisProps} tickFormatter={(d: string) => d.slice(5)} minTickGap={36} />
                        <YAxis {...axisProps} width={56} label={{ value: 'Units per day', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 12 }} />
                        <Tooltip {...tooltipStyle} formatter={(v: unknown) => (Array.isArray(v) ? `${Number(v[0]).toFixed(1)} – ${Number(v[1]).toFixed(1)}` : Number(v).toFixed(1))} />
                        <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: 12 }} />
                        <Area dataKey="band95" name="95% interval" stroke="none" fill={palette[served]} fillOpacity={0.14} isAnimationActive={false} />
                        {show80 && <Area dataKey="band80" name="80% interval" stroke="none" fill={palette[served]} fillOpacity={0.3} isAnimationActive={false} />}
                        {data.level && <Line dataKey="actual" name="Actual" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={false} />}
                        {data.level && activeMethods.map((method) => <Line key={method} dataKey={method} name={METHOD_LABELS[method]} stroke={palette[method]} strokeWidth={1.6} dot={false} isAnimationActive={false} />)}
                        <Line dataKey="production" name={`Production forecast (${METHOD_LABELS[served]})`} stroke={palette[served]} strokeWidth={2.4} dot={false} isAnimationActive={false} />
                        <ReferenceLine x={summary.origins.production} stroke="var(--muted)" strokeDasharray="3 3" />
                      </ComposedChart>
                    </ChartBox>
                    <p className="legend-note">
                      {level === 'item'
                        ? 'Item-level backtest lines are omitted to keep the site small; the production forecast and intervals are shown for every item. '
                        : 'Left of the dotted line: forecasts made at each fold origin vs actual sales. Right: the served 28-day production forecast. '}
                      The production forecast is LightGBM’s item × store forecasts, reconciled with MinT so every level adds up.
                    </p>
                  </>
                )
              }}
            </Async>
          </>
        )
      }}
    </Async>
  )
}

