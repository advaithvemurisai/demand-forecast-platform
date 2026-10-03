import { getJson } from '../data/load'
import type { DriftRow } from '../data/types'
import { Async, Question } from '../components/ui'
import { useData } from '../components/data'
import { num } from '../lib/format'

export default function Drift() {
  const state = useData(() => getJson<DriftRow[]>('drift.json'))
  return (
    <Async state={state}>
      {(drift) => {
        const flag = drift[0]?.retrain
        return (
          <>
            <Question>is the model still accurate, or has customer demand shifted enough to retrain it?</Question>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Store</th><th className="text">Status</th><th>Demand-mix PSI</th><th>Residual PSI</th><th>Prior-year daily units</th><th>Last 28 days daily units</th></tr></thead>
                <tbody>
                  {drift.map((row) => (
                    <tr key={row.node_id}>
                      <td>{row.node_id.replace('store:store_id=', '')}</td>
                      <td className="text"><span className={`status ${row.drift ? 'bad' : 'good'}`}>{row.drift ? '⚠ Drift' : '✓ Stable'}</span></td>
                      <td className="num">{row.demand_psi.toFixed(3)}</td><td className="num">{row.residual_psi.toFixed(3)}</td>
                      <td className="num">{num(row.reference_mean_daily)}</td><td className="num">{num(row.current_mean_daily)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className={`callout ${flag ? 'warn' : 'ok'}`}>
              Retraining flag: <strong>{flag ? 'on' : 'off'}</strong> ({drift[0]?.retrain_reasons}). Alerts fire at PSI ≥ 0.2, a &gt;10% rise in store-level WMAPE, or when the inventory twin’s
              predicted fill rate drifts more than 5 points from what was realised (a sign the twin needs recalibrating before its what-ifs are trusted).
            </div>
          </>
        )
      }}
    </Async>
  )
}
