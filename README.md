# Hierarchical Demand Forecasting, Allocation and Inventory Twin

**[▶ Live app](https://demand-forecast-platform.vercel.app/)**: forecasts, allocation and an inventory simulator that runs in your browser

## The business problem

A retailer has to decide how much of each product to stock and which stores get it. Three things make that hard:

- **Plans at different levels disagree.** Buyers plan categories, store managers plan their store, and the supply chain plans the region. Forecast each level separately and the numbers don't add up, so teams order against conflicting plans.
- **Supply is often short.** When the warehouse can't cover every store, someone has to decide who goes without. A proportional split ignores which units are most likely to sell and what they're worth.
- **Nobody can test a policy before living with it.** What does a demand spike, a late supplier or a different service target do to stock, sales and cash? The answer is usually found out on the shelf.

Too little stock loses sales; too much ties up cash. This project builds the pipeline a planning team would use for all three, on real Walmart sales data (the M5 competition dataset, California stores).

## What's being forecast

**Daily unit sales of 3,049 Walmart products in each of 4 California stores, 28 days ahead.** The products span three categories:

| Category | Products | Departments |
|---|---|---|
| Foods | 1,437 | 3 |
| Household | 1,047 | 2 |
| Hobbies | 565 | 2 |

That is 12,196 product-store combinations, rolled up into departments, categories, stores and the state total. Walmart anonymised the product names, so items appear as codes such as `FOODS_3_090`.

## What each step solves

Holdout = a final 28-day window never used for any modelling choice. Numbers are from the committed run (`data/dashboard/`).

| Step | Business question | Result |
|---|---|---|
| **Forecast** | How much will each product sell in each store, day by day, over four weeks? | 14% lower product-level error than repeating last week's sales |
| **Reconcile** | Do the item, store and regional plans agree? | Forecasts add up at every level; category error 15% lower than adding up item forecasts (WRMSSE 0.448) |
| **Quantify uncertainty** | How sure are we, and how bad could it get? | 95% ranges contain 94.1% of item sales; aggregate levels run narrower (92.4–94.0%) |
| **Safety stock** | How much buffer does each product need over its replenishment window? | A stock target per product: lead time + days between orders, service targets by value × variability class, shelf minimum, case packs |
| **Allocate** | When the warehouse is short, which departments in which stores get the stock? | About $91k a year more revenue and 9.5% less lost revenue than a proportional split of net need, no department starved |
| **Simulate** | What happens to stock, service and cash under a policy, a demand spike or a supply problem? | Day-by-day twin of one warehouse and four stores: the calibrated safety stock lifts fill rate from 87% to 97%; service targets re-tuned per category (below) |
| **Monitor** | Has demand shifted enough to retrain? | PSI and accuracy-decay checks, plus a twin-divergence check |

### Reconciliation in plain terms

The platform uses **MinT reconciliation**. It takes forecasts from every level and finds the closest set of numbers that add up exactly. Forecasts with a good track record barely move; unreliable ones absorb most of the correction. The result is one consistent plan every team can work from, more accurate than building up from items.

## The inventory digital twin

A stateful, vectorised simulation of **supplier → one California distribution centre → 4 stores × 3,049 products**, one day at a time: receive in-transit stock, sell (lost sales when the shelf is empty), order up to a policy target on review days, and ration the DC's stock when it is short. All four stores run in one simulation: the DC holds stock **per product** and shares each product between the stores that ordered it, and every simulated future draws one demand week shared by every product and store, so common shocks hit the whole network at once. The DC reorders each product weekly against its network forecast, with a pooled safety stock (square-root law across stores) that also covers a variable supplier lead time; every supplier order draws its own lead time. Inventory and holding cost are valued at cost, not shelf price. It follows supply-chain digital-twin practice: one narrow use case with KPIs defined first, validation against history, replications with confidence bands, and a maturity path from descriptive to prescriptive.

| Level | What it does | Where |
|---|---|---|
| **Descriptive** | Replays a window on actual sales: on-hand, demand and lost sales per store × department | `twin_timeline` |
| **Diagnostic** | Flags probable stockouts (steady sellers only) and tracks how far the twin drifts from reality | `probable_stockouts`, `twin_validation` |
| **Predictive** | Simulates 50 futures from forecast and calibration data alone, then compares with what happened | `twin_validation` |
| **Prescriptive** | Cost-optimal service level per category, a fair policy comparison at equal inventory, four stress tests, three DC rationing rules, and a stockout watch-list | `twin_frontier`, `twin_policy_curve`, `twin_stress`, `twin_exceptions` |

**What the twin says on the California data**

- **The safety stock is where the value is.** Replaying the holdout weeks on what actually sold, the calibrated safety stock lifts fill rate from 87.3% to 97.0% and cuts lost sales by $188k over four weeks across the four stores, for $9.9k of extra holding cost against $55k of margin recovered.
- **Reordering from the forecast instead of last week's sales adds little on top.** With both policies carrying the same safety stock, the forecast fills 97.0% vs 96.8% at the same $402k of stock; to match it, last-week reordering needs about $415k (3% more). Over a 3–8 day replenishment window, last week's sales rate is already a fair estimate for most products. (An earlier version of this project compared against a last-week policy with *no* safety stock and reported +12.8 pts of fill rate; that gap was the buffer, not the forecast.)
- **Cheapest service targets:** Foods 98%, Hobbies 99%, Household 99.5% (99.8% costs more). Against today's value × variability targets this saves about $2.4k of lost margin + holding cost per four weeks (≈ $31k a year), and each category's saving clears zero across 90% of simulated futures.
- **Supplier lead time is the dominant risk.** If lead time doubles from 7 to 14 days and the planner adjusts after a week, lost sales rise by about $44k over four weeks (90% of futures: $30k–$59k) and fill rate falls 2.6 pts. One delivery arriving a week late costs about $15k, a holiday-style +50% Foods week $21k, and 30% less stock reaching the warehouse for two weeks $8k.
- **How the DC shares a short product barely matters**: days-of-cover, proportional and by-value rationing all lose about $8.2k in the warehouse-shortfall test, because each product's shortfall is split between just four stores with similar cover.
- **Service levels are predicted well; ranges are still too narrow.** Across three windows the predicted network fill rate is off by 1.1 pts on average and in-stock by 0.3 pts (holdout: 96.8% predicted vs 97.0% realised). The realised value fell inside the 90% range in 5 of 24 network checks and 25 of 96 store checks (up from 1 of 18 when stores were simulated separately), because the demand bootstrap only has 4–16 past weeks to draw from.
- **Dollar losses are overstated about 1.7×.** Predicted demand runs ~8% above observed sales, and observed sales are themselves held down by real stockouts. Lost sales is a tail quantity, so read dollar figures as policy-vs-policy comparisons, not absolute forecasts.

The simulator is one numpy-only file, [`src/forecasting/twin.py`](src/forecasting/twin.py). The pipeline, the API and the browser all run that same file (the browser through Pyodide), and a parity test checks CPython and WebAssembly agree to 1e-6.

## How it's built

```mermaid
flowchart LR
    D[Sales data] --> F[Features]
    F --> M[Forecast models]
    M --> R[Reconciliation]
    R --> P[Forecast ranges]
    P --> S[Product safety stock]
    P --> A[Department allocation]
    S --> T[Inventory twin]
    A --> O[Web app, API, BI exports]
    T --> O
```

- **Models:** one LightGBM model forecasts every product in every store, trained without days a steady seller was probably out of stock (6% of training rows). Department, category, store and state forecasts are the product forecasts added up, then reconciled with classical SARIMA forecasts made directly at those levels. Closed days (Christmas) are forecast at zero, and events carry distance-to-event features.
- **Evaluation:** three rolling-origin backtests, then a held-out final window never used for any modelling choice.
- **Intervals:** split-conformal, calibrated separately per segment (Mondrian: items by speed class and category, departments by category), out-of-sample.
- **Allocation:** each week one warehouse can ship 90% of forecast demand to the 4 stores, a simulated shortage. Stores carry stock from week to week, and both rules allocate against **net need** (forecast plus safety stock, minus what the store already holds). An optimisation model splits supply across 28 department × store targets, maximising expected *margin* (assumed by category) over demand scenarios, with every target's stock held to at least half its forecast so none is starved; the benchmark splits supply in proportion to net need. Over 12 test weeks the optimiser earned more in 11, a mean +$1.8k per week (95% CI $0.2k to $3.4k, about $91k a year), and left 9.5% less revenue unfilled while losing 4.4% more units: it favours margin over unit count. Next week's split starts from the twin's on-hand stock at the end of the replayed holdout. The split is by department; inside the twin the DC then rations each product between stores.
- **Decision-grain accuracy:** forecasts are also scored over each product's replenishment window, with bias by weekday, event days, and a tracking-signal exceptions list of products the forecast keeps missing in one direction.
- **Delivery:** a static React app ([`web/`](web/)), a FastAPI service ([`api/`](api/)), Tableau / Looker Studio exports, and MLflow run tracking.

## Honest limitations

- **M5 has no inventory, lead-time, cost or case-pack data.** The twin's starting stock, lead times and their spread, margins (by category, which set unit costs) and holding costs (including food spoilage) are assumptions set in `pipeline.Config` and `TwinConfig`, not measurements. The DC → store leg has a fixed one-day lead time.
- **M5 records sales, not demand.** A zero can mean no demand or no stock. Only steady sellers (≥1 unit/day) are judged by a Poisson zero-run test (≈4% of days flagged); a looser rule flagged 19% and biased the model upward. Some censoring remains, so item-level WMAPE (0.743, vs 0.727 before masking) is measured against sales that are themselves censored.
- **Twin ranges are too narrow** (see above): demand scenarios replay whole past weeks shared by every product and store, so common shocks are kept, but only as far as the few available weeks allow.
- **Aggregate intervals under-cover.** At 95% nominal, store coverage is 92.6% and category 92.4%; at 80% nominal, state/total/store cover about 71%.
- **Item-level accuracy is modest** (WMAPE 0.74 daily; 0.30–0.36 for fast sellers over their replenishment window, above 1.0 for sporadic ones). The product model under-forecasts sporadic and slow sellers and is biased up on fast Household items.
- **Temporal reconciliation did not help.** Pulling aggregate daily forecasts toward weekly ARIMA forecasts made the base SARIMA slightly worse (state WMAPE 0.0502 vs 0.0462) and left reconciled accuracy unchanged, so it is off by default (`--temporal`).
- **The override demo is automatic, not planner input.** A bias-correction rule for products with a persistent tracking signal adds value on the products it flags (FVA +2 to +11 pts WMAPE) and a negligible amount overall; real planner overrides would need real data.
- A classical model forecasting the store and regional totals directly can be more accurate at those levels, but its numbers don't add up across the hierarchy, so it can't be used as a plan.
- The run covers California only; extending to all states needs more memory than a laptop.

## Run it

```bash
python -m pip install -e ".[prophet,api,data,dev]"   # macOS: brew install libomp
python scripts/download_data.py                      # downloads and verifies the M5 data
python -m forecasting --states CA                    # prepare the data
python -m forecasting.pipeline                       # train, evaluate, simulate, write results (~16 min on a laptop)
python -m pytest -q
python -m forecasting.replay --weeks 8               # optional: replay the weekly planning cycle (one fit per week)

# web app (static; the simulator runs in your browser)
python scripts/build_web_data.py                     # data/dashboard -> web/public
cd web && npm install && npm run dev                 # or: npm test, npm run build

# API
DATA_DIR=data/dashboard uvicorn api.main:app         # /docs for the OpenAPI page
```

Useful flags: `--no-twin`, `--temporal`, `--max-items 300 --no-prophet --no-mlflow` for a 3-minute smoke run.

## Deploy

- **Web app → Vercel:** import the repo and set the root directory to `web/`. [`web/vercel.json`](web/vercel.json) runs `npm run build:vercel`, which installs numpy/pandas/pyarrow into a throwaway virtualenv (Vercel's Python is uv-managed and refuses system installs), regenerates `web/public/data` and `web/public/py` from the committed extract in `data/dashboard/` (including about 90 s of preset what-if runs), then builds. It also sets cache headers and a content-security-policy that allows only the Pyodide CDN. Everything is static: no backend, no cold starts. If a future build image can't run the Python step, remove `web/public/data` and `web/public/py` from `web/.gitignore`, run `python scripts/build_web_data.py` locally, commit the output (about 17 MB), and set the build command back to `npm run build`.
- **API (optional):** [`api/requirements.txt`](api/requirements.txt) is the slim install (no modelling stack). Set `DATA_DIR=data/dashboard`, `ALLOWED_ORIGINS`, and `TRUSTED_PROXY_HOPS` when behind a proxy. `POST /twin/simulate` runs the whole network (one bundle, `twin_inputs/network.npz`, since every store shares the DC), returns KPIs for all stores and each store, and is capped at 50 futures and rate-limited.
- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the Python tests, checks the browser snapshot is current, builds the web data, then lints, tests and builds the web app.

The web app and the API read the committed extract in `data/dashboard/`. See [tableau/README.md](tableau/README.md) and [looker_studio/](looker_studio/) for the BI versions.
