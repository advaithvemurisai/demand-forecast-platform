import type {
  AllocationBacktestRow, CoverageRow, ReconRow, Summary, TwinFrontierRow, TwinResponseRow, TwinStressRow, TwinValidationRow,
} from '../data/types'

export const sum = (values: number[]): number => values.reduce((total, value) => total + value, 0)
export const mean = (values: number[]): number => (values.length ? sum(values) / values.length : NaN)

/** Weekly revenue uplift of the optimiser over pro-rata, and how many weeks it won. */
export function weeklyUplift(rows: AllocationBacktestRow[]) {
  const weeks = new Map<string, { fold: string; week: number; lp?: AllocationBacktestRow; prorata?: AllocationBacktestRow }>()
  for (const row of rows) {
    const key = `${row.fold}|${row.week}`
    const entry = weeks.get(key) ?? { fold: row.fold, week: row.week }
    if (row.policy === 'lp_scenario') entry.lp = row
    else entry.prorata = row
    weeks.set(key, entry)
  }
  return [...weeks.values()]
    .filter((entry) => entry.lp && entry.prorata)
    .map((entry) => ({
      fold: entry.fold, week: entry.week, label: `${entry.fold.replace('backtest_', 'backtest ')} · wk ${entry.week}`,
      uplift: entry.lp!.revenue_fulfilled / entry.prorata!.revenue_fulfilled - 1,
      lpLost: entry.lp!.lost_revenue, proLost: entry.prorata!.lost_revenue,
      lpShort: entry.lp!.stockout_nodes, proShort: entry.prorata!.stockout_nodes,
    }))
}

export function coverageByLevel(rows: CoverageRow[]) {
  const groups = new Map<string, CoverageRow[]>()
  for (const row of rows) {
    const key = `${row.level}|${row.nominal}`
    groups.set(key, [...(groups.get(key) ?? []), row])
  }
  return [...groups.entries()].map(([key, group]) => {
    const [level, nominal] = key.split('|')
    return { level, nominal: Number(nominal), coverage: mean(group.map((r) => r.coverage)), width: mean(group.map((r) => r.mean_width)) }
  })
}

/** Holdout WMAPE of a method at a level, or NaN when absent. */
export function holdoutWmape(rows: ReconRow[], method: string, level: string): number {
  return rows.find((row) => row.fold === 'holdout' && row.method === method && row.level === level)?.wmape ?? NaN
}

/** Network-level twin headline numbers for one policy and fold. */
export function twinHeadline(rows: TwinValidationRow[], policy = 'forecast_reorder', fold?: string) {
  const network = rows.filter((row) => row.store_id === 'all' && row.policy === policy)
  const latest = fold ?? network.map((row) => row.fold).sort().at(-1)
  const byMetric = new Map(network.filter((row) => row.fold === latest).map((row) => [row.metric, row]))
  return { fold: latest, metrics: byMetric, inBand: network.filter((row) => row.in_band).length, checks: network.length }
}

export function recommendedService(rows: TwinFrontierRow[], assumption?: string): Record<string, TwinFrontierRow> {
  return Object.fromEntries(rows.filter((row) => row.recommended && (!assumption || row.assumption === assumption)).map((row) => [row.category, row]))
}

/** The best response to a scenario: the largest net benefit whose 90% range clears zero, else none. */
export function bestResponse(rows: TwinResponseRow[], scenario: string): TwinResponseRow | undefined {
  return rows.filter((r) => r.scenario === scenario && r.response !== 'none' && r.net_benefit_lower > 0).sort((a, b) => b.net_benefit - a.net_benefit)[0]
}

/** Change in a stress metric versus baseline for one policy (positive lost sales = worse). */
export function stressDelta(rows: TwinStressRow[], scenario: string, policy: string, metric: string): number {
  return rows.find((row) => row.scenario === scenario && row.policy === policy && row.metric === metric)?.delta ?? NaN
}

export const servedWrmsse = (summary: Summary): number => summary.wrmsse_holdout[summary.served_method]
