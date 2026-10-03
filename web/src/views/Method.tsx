import { getJson } from '../data/load'
import type { Summary } from '../data/types'
import { Async } from '../components/ui'
import { useData } from '../components/data'
import { METHOD_LABELS, MODEL_LABELS, num } from '../lib/format'

const REPO_URL = 'https://github.com/advaithvemurisai/demand-forecast-platform'

export default function Method() {
  const state = useData(() => getJson<Summary>('summary.json'))
  return (
    <Async state={state}>
      {(run) => (
        <div style={{ maxWidth: '78ch' }}>
          <h2 style={{ marginTop: 18 }}>The business problem</h2>
          <p>A retailer plans stock at several levels at once: buyers plan categories, store managers plan their store, and the supply chain plans the region. If each level is forecast separately, the numbers disagree, and nobody knows which plan to order against. On top of that, supply is often short, so someone has to decide which stores go without. Getting it wrong means empty shelves in one store and excess stock in another.</p>
          <h2>What MinT reconciliation does, in plain terms</h2>
          <p>It takes the forecasts from every level and finds the closest set of numbers that add up exactly. Forecasts that have been reliable barely move; noisy ones absorb most of the correction. The name comes from the maths: of all the ways to make the numbers add up, it picks the one with the smallest total forecast-error variance (the <em>trace</em> of the error covariance matrix). Pulling aggregate forecasts toward weekly ones first (<strong>temporal reconciliation</strong>) was tested and did not help, so it is off by default.</p>
          <h2>What the allocation adds</h2>
          <p>A forecast says what <em>would</em> sell; the allocation decides what each store <em>gets</em>. It works on net need (forecast plus safety stock, minus what the store already holds), and the LP uses the forecast ranges to weigh the chance each unit sells against its margin, with a minimum fill so no department is starved. In the backtest stores carry stock from week to week; the gain over a proportional split is reported with a confidence interval. The weekly split is by department; inside the simulator the warehouse shares each product between stores day by day.</p>
          <h2>What the inventory twin adds</h2>
          <p>The twin replays the chain day by day (supplier → one DC → all four stores), tracking on-hand stock, in-transit orders and lost sales. The DC holds stock per product and rations it between the stores when it is short; its safety stock pools the stores’ demand risk (square-root law) and covers a variable supplier lead time. Inventory and holding cost are valued at cost, not shelf price. It is <em>validated</em> by predicting a window’s service levels from forecast and calibration data alone, then replaying what actually sold. It also runs what-ifs (demand spikes, supplier delays, warehouse shortfalls) with confidence bands, and sweeps service levels to find the cheapest per category. The browser runs the same Python simulator through WebAssembly.</p>
          <h2>Pipeline</h2>
          <p>M5 raw → silver Parquet → leakage-safe features (every lag ≥ the 28-day horizon, plus event-distance features) → naive floors, LightGBM global/local, SARIMA and Prophet → bottom-up, top-down and MinT reconciliation → Mondrian conformal intervals and ABC × XYZ safety stock → scenario LP allocation → inventory twin. Runs are tracked in MLflow.</p>
          <h2>Evaluation</h2>
          <p>Three 28-day rolling-origin backtests, then a holdout on the M5 validation window scored against real actuals. The bottom-level model (<strong>{MODEL_LABELS[run.selected_bottom_model]}</strong>) and the served reconciliation method (<strong>{METHOD_LABELS[run.served_method]}</strong>) were chosen on backtest folds only.</p>
          <h2>MinT at SKU scale</h2>
          <p>A diagonal-covariance MinT is solved with the Woodbury identity, which inverts a small matrix instead of {num(run.n_series)} × {num(run.n_series)}. The out-of-sample variant weights each node by its base model’s error in the previous 28-day window.</p>
          <h2>Honest caveats</h2>
          <ul>
            <li>M5 has no inventory, lead-time, cost or case-pack data. The twin’s starting stock, lead times, margins and holding costs are <strong>assumptions</strong> (set in the pipeline config).</li>
            <li>M5 records sales, not demand: a zero can mean no demand or no stock. Probable stockouts are flagged (steady sellers only, where a long zero run is statistically implausible) and left out of training and calibration, but some censoring remains.</li>
            <li>Twin demand scenarios replay whole past weeks of sales, shared by every product and store, so common shocks are kept, but only as far as the few available weeks allow. Its ranges are still too narrow.</li>
            <li>Cost, lead-time spread and holding rates are assumed by category; the DC→store leg has a fixed one-day lead time.</li>
            <li>The un-reconciled aggregates can beat every coherent method at the aggregate levels; coherence is what makes a plan usable. WRMSSE covers 6 California levels and is not comparable to the M5 leaderboard.</li>
          </ul>
          <p>Full write-up and code: <a href={REPO_URL}>{REPO_URL}</a></p>
        </div>
      )}
    </Async>
  )
}
