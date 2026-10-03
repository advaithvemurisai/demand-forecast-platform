import { getJson } from '../data/load'
import type { Summary } from '../data/types'
import { Async } from '../components/ui'
import { useData } from '../components/data'
import { METHOD_LABELS, MODEL_LABELS, num, pct, usd } from '../lib/format'

const REPO_URL = 'https://github.com/advaithvemurisai/demand-forecast-platform'

export default function Method() {
  const state = useData(() => getJson<Summary>('summary.json'))
  return (
    <Async state={state}>
      {(run) => {
        const twin = run.twin
        const equal = twin?.equal_inventory
        const naiveStock = equal?.current_practice_inventory_for_same
        const extraStock = equal && naiveStock != null && Number.isFinite(naiveStock) ? naiveStock / equal.inventory_value - 1 : NaN
        return (
          <div style={{ maxWidth: '78ch' }}>
            <h2 style={{ marginTop: 18 }}>The business problem</h2>
            <p>A retailer plans stock at several levels at once: buyers plan categories, store managers plan their store, and the supply chain plans the region. If each level is forecast separately, the numbers disagree, and nobody knows which plan to order against. Supply is often short, so someone has to decide which stores go without. And a new ordering policy is usually tested on the shelf. Getting it wrong means empty shelves in one store and excess stock in another.</p>

            <h2>1 · Forecast</h2>
            <p>One global <strong>{MODEL_LABELS[run.selected_bottom_model]}</strong> model forecasts daily sales for all {num(run.n_series)} product-store pairs, 28 days ahead. Every lag is at least 28 days, so nothing leaks from the forecast window. M5 records sales, not demand, so a zero can mean an empty shelf: for steady sellers, a run of zeros too long to be chance (a Poisson test) is treated as a probable stockout and left out of training and calibration. Closed days are forecast at zero, and events get distance-to-event features. Naive and seasonal-naive forecasts set the floor every model must beat.</p>

            <h2>2 · Reconcile: one plan that adds up</h2>
            <p>Classical SARIMA models forecast the department, category, store and state totals directly. <strong>MinT reconciliation</strong> then finds the closest set of numbers that add up exactly across every level. Forecasts that have been reliable barely move; noisy ones absorb most of the correction. Of all the ways to make the numbers add up, MinT picks the one with the smallest total forecast-error variance (the <em>trace</em> of the error covariance matrix); the served variant (<strong>{METHOD_LABELS[run.served_method]}</strong>) weights each series by its error in the previous 28-day window. With a diagonal covariance it is solved through the Woodbury identity, which inverts a small matrix instead of a {num(run.n_series)} × {num(run.n_series)} one. Pulling aggregate forecasts toward weekly ones first (temporal reconciliation) was tested and did not help, so it is off.</p>

            <h2>3 · Ranges and safety stock</h2>
            <p>Every forecast gets a range from <strong>split-conformal prediction</strong>: past out-of-sample errors set its width, calibrated separately for each segment (product speed × category), so slow, erratic products get wider ranges than steady staples. No distribution is assumed.</p>
            <p>Safety stock uses the same idea over each product’s <strong>replenishment window</strong> (one day of lead time plus the days until the next order: 2 for food, 7 otherwise). The target is the conformal quantile of past errors over that window at the product’s service level. Service levels follow an <strong>ABC × XYZ</strong> grid: high-revenue, steady products (A, X) get 98%, the erratic long tail (C, Z) 85%. Every product keeps at least two units on the shelf.</p>

            <h2>4 · Allocate when the warehouse is short</h2>
            <p>A forecast says what <em>would</em> sell; the allocation decides what each store <em>gets</em>. M5 has no warehouse stock, so a shortage is imposed as a stress test: in the backtest the warehouse ships 90% of each week’s forecast, and next week’s plan gets 90% of the stores’ net need. Both rules work on <strong>net need</strong>: forecast plus safety stock, minus what the store already holds. A scenario linear program maximises expected <em>margin</em> over demand scenarios drawn from the forecast ranges, with every department’s stock kept at least half its forecast so none is starved; the benchmark splits supply in proportion to net need. In the backtest stores carry unsold stock from week to week, and the gain is reported with a confidence interval. The split is by department; inside the simulator the warehouse then shares each product between stores day by day.</p>

            <h2>5 · Stress-test in the inventory twin</h2>
            <p>The twin simulates <strong>supplier → one warehouse → four stores</strong> for every product, one day at a time: receive stock, sell (a sale is lost when the shelf is empty), reorder up to target, and ration the warehouse’s stock when it is short. All four stores run in one simulation:</p>
            <ul>
              <li>The warehouse holds stock <strong>per product</strong> and shares it between the stores that ordered it: by days of cover, proportionally, or by margin.</li>
              <li>Its safety stock pools the stores’ demand risk (square-root law) and covers a supplier lead time that varies from order to order.</li>
              <li>Each simulated future replays a past week of real sales, scaled to today’s forecast and shared by every product and store, so a bad week hits the whole network at once.</li>
              <li>Inventory and holding cost are valued at <strong>cost</strong>, not shelf price, with a higher holding rate for perishable food.</li>
              <li>Supplier problems come in two forms: one late delivery, or a lasting slowdown the planner adjusts to after a week.</li>
              <li>Each disruption can be met with a response, each with its own cost: plan on the longer lead time at once, pre-build warehouse stock, expedite part of a late order or use a second supplier (at an assumed premium), or put a known spike in the forecast.</li>
              <li>It reports what operations look at first: weeks of supply, turns, GMROI, lost sales as a share of demand, and the warehouse’s stock, inbound orders and fill rate.</li>
            </ul>
            <p>The pipeline, the API and this site all run the same numpy-only Python file; the browser runs it through WebAssembly (Pyodide), and a test checks both agree to six decimal places.</p>

            <h2>6 · Comparing policies fairly</h2>
            <p>Two rules make comparisons honest. First, <strong>common random numbers</strong>: every option sees the same simulated futures, so a difference comes from the policy, not from luck, and savings are measured future by future with a 90% range. A saving whose range includes zero is reported as no clear gain.</p>
            <p>Second, <strong>compare at equal stock</strong>. Any policy looks better if it simply holds more inventory, so both reordering policies carry the same safety stock, and the stock is swept from none to four times the standard to trace fill rate against inventory for each.
              {equal && Number.isFinite(extraStock) && (
                <> On the holdout weeks that comparison changed the conclusion: reordering from the forecast fills {pct(equal.forecast)} of demand against {pct(equal.current_practice ?? NaN)} for reordering from last week’s sales at the same {usd(equal.inventory_value)} of stock, and last week’s sales need about {pct(extraStock, 0)} more stock to match. The safety stock, not the reordering rule, does most of the work.</>
              )}
            </p>

            <h2>7 · Pricing a stockout honestly</h2>
            <p>The cheapest service target balances lost margin against holding cost, and the optimum sits where shortage cost ÷ (shortage + overstock cost) says it should. Overstate what a stockout costs and every target drifts up. Two things overstate it here: the simulator predicts {twin?.lost_sales_bias ? `${twin.lost_sales_bias.toFixed(1)}×` : 'more'} the lost sales that actually happened in the validation windows, and it treats every unmet unit as a lost sale, whereas shoppers facing an empty shelf mostly switch item or brand, wait, or buy elsewhere (only about 40% of encounters lose the sale to the store; Gruen &amp; Corsten, 2002). The service-target curves are therefore shown under three assumptions, and the headline advice uses the bias-corrected one. Under the uncorrected curve every category looked worth raising, Household by $141k of stock; once corrected, Foods should go down, Hobbies up a little and Household stay, and the site shows all three answers instead of the most flattering one.</p>

            <h2>Evaluation</h2>
            <ul>
              <li><strong>Forecasts:</strong> three 28-day rolling-origin backtests, then a holdout on the M5 validation window scored against real actuals. The model and the reconciliation method were chosen on backtests only, never on the holdout. Accuracy is also scored over each product’s replenishment window, by weekday and on event days.</li>
              <li><strong>Allocation:</strong> weekly allocations are scored on what actually sold.</li>
              <li><strong>Twin:</strong> for each past window it predicts service levels from forecast and calibration data alone, then replays what actually sold through the same policy.
                {twin && <> The realised value fell inside the predicted 90% range in {pct(twin.share_realised_in_band, 0)} of network checks.</>}</li>
              <li><strong>Monitoring:</strong> PSI drift on demand and on forecast errors, accuracy decay, and an alert when the twin’s predicted fill rate drifts more than 5 points from reality.</li>
            </ul>

            <h2>Honest caveats</h2>
            <ul>
              <li><strong>Assumed costs and lead times.</strong> M5 has no inventory, lead-time, cost or case-pack data. Starting stock, lead times and their spread, margins (which set unit costs) and holding rates are stated assumptions; the warehouse → store leg has a fixed one-day lead time.</li>
              <li><strong>Censored sales.</strong> Probable stockouts are masked for steady sellers only, so some censoring remains, and accuracy is measured against sales that stockouts themselves hold down.</li>
              <li><strong>Simulator ranges are too narrow</strong> because only a few past weeks are available to replay. Predicted dollar losses run high, since predicted demand is above observed (censored) sales. Read dollar figures as comparisons between policies, not forecasts.</li>
              <li><strong>Perfect store execution.</strong> Every unit that reaches a store is assumed to be on the shelf. Most real stockouts start in the store (shelf replenishment, store ordering), so the simulated in-stock rate is a ceiling, not a prediction.</li>
              <li><strong>Product-level accuracy is modest</strong>, as it is for any daily forecast of slow-moving products; aggregate ranges cover slightly less than their nominal level.</li>
              <li><strong>Unreconciled totals can be more accurate</strong> at the aggregate levels, but they don’t add up, so they can’t be used as one plan. WRMSSE covers 6 California levels and is not comparable to the M5 leaderboard.</li>
            </ul>
            <p>Full write-up and code: <a href={REPO_URL}>{REPO_URL}</a></p>
          </div>
        )
      }}
    </Async>
  )
}
