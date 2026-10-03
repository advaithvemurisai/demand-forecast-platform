import { CartesianGrid, Legend, ReferenceLine, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { getJson, getOptional } from '../data/load'
import type { CoverageRow, SegmentCoverageRow } from '../data/types'
import { Async, ChartBox, Question } from '../components/ui'
import { useData } from '../components/data'
import { axisProps, gridProps, SERIES, tooltipStyle } from '../lib/chart'
import { LEVEL_LABELS, LEVELS, num, pct } from '../lib/format'
import { coverageByLevel, mean } from '../lib/metrics'

export default function Intervals() {
  const state = useData(async () => ({
    coverage: await getJson<CoverageRow[]>('interval_coverage.json'),
    segments: await getOptional<SegmentCoverageRow[]>('interval_coverage_segment.json'),
  }))
  return (
    <Async state={state}>
      {({ coverage, segments }) => {
        const summary = coverageByLevel(coverage)
        const series = [0.8, 0.95].map((nominal) => ({
          nominal,
          points: LEVELS.map((level) => ({ level: LEVEL_LABELS[level], coverage: summary.find((s) => s.level === level && s.nominal === nominal)?.coverage ?? NaN })),
        }))
        const segmentTable = (() => {
          if (!segments) return []
          const groups = new Map<string, number[]>()
          for (const row of segments.filter((r) => r.nominal === 0.95)) {
            const key = `${row.level}|${row.segment}`
            groups.set(key, [...(groups.get(key) ?? []), row.coverage])
          }
          return [...groups.entries()].map(([key, values]) => {
            const [level, segment] = key.split('|')
            return { level, segment, coverage: mean(values), n: segments.filter((r) => r.level === level && r.segment === segment && r.nominal === 0.95).reduce((a, r) => a + r.n, 0) }
          }).sort((a, b) => a.level.localeCompare(b.level) || a.segment.localeCompare(b.segment))
        })()
        return (
          <>
            <Question>can the forecast ranges be trusted to set safety stock? If a 95% range misses more than 5% of the time, stores run out more often than planned.</Question>
            <ChartBox>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis type="category" dataKey="level" allowDuplicatedCategory={false} {...axisProps} />
                <YAxis type="number" dataKey="coverage" domain={[0.6, 1.02]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} {...axisProps} width={48} />
                <Tooltip {...tooltipStyle} formatter={(v: unknown) => pct(Number(v))} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {series.map((s, i) => <ReferenceLine key={`ref${s.nominal}`} y={s.nominal} stroke={SERIES[i]} strokeDasharray="3 3" />)}
                {series.map((s, i) => <Scatter key={s.nominal} name={`${pct(s.nominal, 0)} interval`} data={s.points} fill={SERIES[i]} isAnimationActive={false} />)}
              </ScatterChart>
            </ChartBox>
            <p className="legend-note">Empirical coverage vs nominal (dotted lines). Each fold is calibrated only on earlier folds, so this coverage is out-of-sample.</p>
            <p>
              Split-conformal intervals with signed, scale-normalised scores, calibrated separately inside each segment (<strong>Mondrian conformal</strong>): items by speed class and category,
              departments by category. Aggregate levels run a few points narrow because they calibrate on few nodes × 28 days. Days a product was probably out of stock are left out of calibration.
            </p>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Level</th><th>Coverage 80%</th><th>Coverage 95%</th><th>Mean width 80%</th><th>Mean width 95%</th></tr></thead>
                <tbody>
                  {LEVELS.map((level) => {
                    const at = (nominal: number) => summary.find((s) => s.level === level && s.nominal === nominal)
                    return <tr key={level}><td>{LEVEL_LABELS[level]}</td><td className="num">{pct(at(0.8)?.coverage ?? NaN)}</td><td className="num">{pct(at(0.95)?.coverage ?? NaN)}</td><td className="num">{num(at(0.8)?.width ?? NaN, 1)}</td><td className="num">{num(at(0.95)?.width ?? NaN, 1)}</td></tr>
                  })}
                </tbody>
              </table>
            </div>
            {segmentTable.length > 0 && (
              <>
                <h3>95% coverage by segment</h3>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Level</th><th className="text">Segment</th><th>Coverage</th><th>Observations</th></tr></thead>
                    <tbody>
                      {segmentTable.map((row) => (
                        <tr key={`${row.level}${row.segment}`}><td>{LEVEL_LABELS[row.level]}</td><td className="text">{row.segment.replace('|', ' · ')}</td><td className={`num ${Math.abs(row.coverage - 0.95) > 0.03 ? 'bad' : ''}`}>{pct(row.coverage)}</td><td className="num">{num(row.n)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="legend-note">A pooled interval can look right overall while being wrong for each segment. Red = more than 3 points from the 95% target.</p>
              </>
            )}
          </>
        )
      }}
    </Async>
  )
}
