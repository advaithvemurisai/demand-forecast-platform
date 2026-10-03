import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson, getOptional } from '../data/load'
import type { DecisionAccuracyRow, EventAccuracyRow, FvaRow, ModelRow, ReconRow, Summary, WeekdayBiasRow } from '../data/types'
import { Async, ChartBox, Question, Segmented, Select } from '../components/ui'
import { shade, useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { LEVEL_LABELS, LEVELS, METHOD_LABELS, MODEL_LABELS, pct, points, relativeGap } from '../lib/format'
import { mean } from '../lib/metrics'

type Metric = 'wmape' | 'wrmsse'

export default function Accuracy() {
  const [window_, setWindow] = useState<'holdout' | 'backtest mean'>('holdout')
  const [metric, setMetric] = useState<Metric>('wmape')
  const [modelLevel, setModelLevel] = useState('item')
  const state = useData(async () => {
    const [recon, models, summary, decision, weekday, events, fva] = await Promise.all([
      getJson<ReconRow[]>('reconciliation_metrics.json'), getJson<ModelRow[]>('model_metrics.json'), getJson<Summary>('summary.json'),
      getOptional<DecisionAccuracyRow[]>('decision_accuracy.json'), getOptional<WeekdayBiasRow[]>('weekday_bias.json'),
      getOptional<EventAccuracyRow[]>('event_accuracy.json'), getOptional<FvaRow[]>('override_fva.json'),
    ])
    return { recon, models, summary, decision, weekday, events, fva }
  })
  return (
    <Async state={state}>
      {({ recon, models, summary, decision, weekday, events, fva }) => {
        const served = summary.served_method
        const inWindow = <T extends { fold: string }>(rows: T[]) => rows.filter((row) => (window_ === 'holdout' ? row.fold === 'holdout' : row.fold.startsWith('backtest')))
        const cell = (level: string, method: string) => mean(inWindow(recon).filter((r) => r.level === level && r.method === method).map((r) => r[metric]))
        const methods = Object.keys(METHOD_LABELS)
        const bars = (() => {
          const groups = new Map<string, { model: string; bottom_up?: number; direct?: number }>()
          for (const row of inWindow(models).filter((r) => r.level === modelLevel)) {
            const entry = groups.get(row.model) ?? { model: MODEL_LABELS[row.model] ?? row.model }
            const list = inWindow(models).filter((r) => r.level === modelLevel && r.model === row.model && r.approach === row.approach).map((r) => r[metric])
            if (row.approach === 'bottom_up') entry.bottom_up = mean(list)
            else entry.direct = mean(list)
            groups.set(row.model, entry)
          }
          return [...groups.values()].sort((a, b) => (b.bottom_up ?? b.direct ?? 0) - (a.bottom_up ?? a.direct ?? 0))
        })()
        const decisionHoldout = decision?.filter((row) => row.fold === 'holdout') ?? []
        return (
          <>
            <Question>which way of making the forecasts add up is the most accurate, and how good is the underlying model?</Question>
            <div className="controls">
              <Segmented label="Evaluation window" value={window_} options={[{ value: 'holdout', label: 'Holdout' }, { value: 'backtest mean', label: 'Backtest mean' }]} onChange={setWindow} />
              <Segmented label="Metric" value={metric} options={[{ value: 'wmape', label: 'WMAPE' }, { value: 'wrmsse', label: 'WRMSSE' }]} onChange={setMetric} />
            </div>
            <p className="legend-note">The holdout was never used for any modelling decision.</p>
            <div className="table-wrap heat">
              <table>
                <thead><tr><th>Level</th>{methods.map((m) => <th key={m}>{METHOD_LABELS[m]}</th>)}</tr></thead>
                <tbody>
                  {LEVELS.map((level) => {
                    const values = methods.map((m) => cell(level, m))
                    const gaps = relativeGap(values)
                    return <tr key={level}><td>{LEVEL_LABELS[level]}</td>{values.map((v, i) => <td key={methods[i]} style={{ background: shade(gaps[i]) }}>{v.toFixed(3)}</td>)}</tr>
                  })}
                </tbody>
              </table>
            </div>
            <p className="legend-note">Reconciliation {metric.toUpperCase()} by level (lower is better; darker = further behind the best in its row). Base forecast = each level forecast on its own: SARIMA from department × store upward, LightGBM at item × store.</p>
            <p>
              <strong>Served:</strong> {METHOD_LABELS[served]}, chosen by the lowest backtest WRMSSE among coherent methods. The unreconciled forecasts can be more accurate at the
              aggregate levels but are <em>incoherent</em>: stores don’t add up to the state, so a planner can’t order against them. See the Method view.
            </p>

            <h2>Base models</h2>
            <div className="controls"><Select label="Level" value={modelLevel} options={LEVELS.map((l) => ({ value: l, label: LEVEL_LABELS[l] }))} onChange={setModelLevel} /></div>
            <ChartBox size="short">
              <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 30, bottom: 4, left: 40 }}>
                <CartesianGrid {...gridProps} horizontal={false} vertical />
                <XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="model" {...axisProps} width={130} />
                <Tooltip {...tooltipStyle} formatter={(v: unknown) => Number(v).toFixed(4)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="bottom_up" name="Item forecasts added up" fill={SERIES[0]} isAnimationActive={false} />
                <Bar dataKey="direct" name="Forecast directly at this level" fill={SERIES[1]} isAnimationActive={false} />
              </BarChart>
            </ChartBox>

            {decisionHoldout.length > 0 && (
              <>
                <h2>Accuracy where planners decide</h2>
                <p className="muted">
                  Daily item WMAPE is not how replenishment is judged. These numbers sum forecasts over each product’s <strong>replenishment window</strong>
                  (lead time + days between orders: 3 days for food, 8 for the rest) at item × store level, holdout window.
                </p>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Category</th><th>Window</th><th className="text">Product speed</th><th>WMAPE</th><th>Bias</th><th>Units</th></tr></thead>
                    <tbody>
                      {[...decisionHoldout].sort((a, b) => a.cat_id.localeCompare(b.cat_id) || b.units - a.units).map((row) => (
                        <tr key={`${row.cat_id}${row.velocity}`}>
                          <td>{row.cat_id}</td><td>{row.window_days} days</td><td className="text">{row.velocity}</td>
                          <td className="num">{pct(row.wmape)}</td>
                          <td className={`num ${Math.abs(row.bias) > 0.1 ? 'bad' : ''}`}>{row.bias >= 0 ? '+' : '−'}{pct(Math.abs(row.bias))}</td>
                          <td className="num">{Math.round(row.units).toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="legend-note">Bias is forecast minus actual over actual: positive = over-forecast. Sporadic sellers are inherently noisy at any window.</p>
              </>
            )}

            {weekday && weekday.length > 0 && (
              <div className="grid two">
                <div>
                  <h3>Bias by day of week (holdout)</h3>
                  <ChartBox size="short">
                    <BarChart data={weekday.filter((w) => w.fold === 'holdout')} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis dataKey="weekday" {...axisProps} />
                      <YAxis {...axisProps} width={48} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip {...tooltipStyle} formatter={(v: unknown) => pct(Number(v))} />
                      <Bar dataKey="bias" name="Forecast − actual" fill={SERIES[0]} isAnimationActive={false} />
                    </BarChart>
                  </ChartBox>
                  <p className="legend-note">Negative = under-forecast. Weekend peaks are where the item model has historically fallen short.</p>
                </div>
                {events && events.length > 0 && (
                  <div>
                    <h3>Event days vs normal days</h3>
                    <div className="table-wrap">
                      <table>
                        <thead><tr><th>Window</th><th className="text">Days</th><th>Item WMAPE</th><th>Total WMAPE</th><th>Total bias</th></tr></thead>
                        <tbody>
                          {events.map((row) => (
                            <tr key={`${row.fold}${row.day_type}`}><td>{row.fold.replace('_', ' ')}</td><td className="text">{row.day_type} ({row.days})</td><td className="num">{pct(row.item_wmape)}</td><td className="num">{pct(row.total_wmape)}</td><td className="num">{pct(row.total_bias)}</td></tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="legend-note">Events carry distance-to-event features, and stores are forecast closed on Christmas Day.</p>
                  </div>
                )}
              </div>
            )}

            {fva && fva.length > 0 && (
              <>
                <h3>Forecast value added by a bias-correction override</h3>
                <p className="muted">
                  A planner-style rule: products whose forecast has missed in one direction for weeks get nudged halfway toward what actually sold. FVA is base WMAPE minus adjusted
                  WMAPE (positive = the override helped). This is a demo of the override layer with an automatic rule, not real planner input.
                </p>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Window</th><th className="text">Scope</th><th>Products</th><th>Base WMAPE</th><th>Adjusted WMAPE</th><th>FVA</th></tr></thead>
                    <tbody>
                      {fva.map((row) => (
                        <tr key={`${row.fold}${row.scope}`}>
                          <td>{row.fold.replace('_', ' ')}</td><td className="text">{row.scope}</td><td className="num">{row.n.toLocaleString()}</td>
                          <td className="num">{pct(row.base_wmape)}</td><td className="num">{pct(row.adjusted_wmape)}</td>
                          <td className={`num ${row.fva > 0 ? 'good' : 'bad'}`}>{points(row.fva)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </>
        )
      }}
    </Async>
  )
}
