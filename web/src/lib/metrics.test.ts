import { describe, expect, it } from 'vitest'
import { coverageByLevel, recommendedService, twinHeadline, weeklyUplift } from './metrics'
import type { AllocationBacktestRow, TwinFrontierRow, TwinValidationRow } from '../data/types'

const bt = (policy: string, revenue: number, week = 1): AllocationBacktestRow => ({
  fold: 'backtest_2', week, policy, supply: 1, units_fulfilled: 1, units_demanded: 1, fill_rate: 1, revenue_fulfilled: revenue, stockout_nodes: 5,
  units_lost: 1, lost_revenue: 10,
})

describe('metrics', () => {
  it('computes weekly uplift per paired week', () => {
    const rows = [bt('lp_scenario', 105), bt('pro_rata', 100), bt('lp_scenario', 90, 2)]
    const weeks = weeklyUplift(rows)
    expect(weeks).toHaveLength(1)
    expect(weeks[0].uplift).toBeCloseTo(0.05)
  })
  it('averages coverage across folds per level and nominal', () => {
    const out = coverageByLevel([
      { fold: 'a', level: 'item', nominal: 0.95, coverage: 0.94, mean_width: 2 },
      { fold: 'b', level: 'item', nominal: 0.95, coverage: 0.96, mean_width: 4 },
    ])
    expect(out).toEqual([{ level: 'item', nominal: 0.95, coverage: 0.95, width: 3 }])
  })
  it('finds the latest network twin fold and counts in-band checks', () => {
    const row = (fold: string, inBand: boolean): TwinValidationRow => ({
      fold, store_id: 'all', policy: 'forecast_reorder', metric: 'fill_rate', predicted: 0.97, lower: 0.95, upper: 0.99, realised: 0.96, in_band: inBand, divergence: 0,
    })
    const head = twinHeadline([row('backtest_2', true), row('holdout', false)])
    expect(head.fold).toBe('holdout')
    expect(head.inBand).toBe(1)
    expect(head.checks).toBe(2)
  })
  it('picks the recommended service level per category', () => {
    const frontier = [
      { category: 'FOODS', service_level: 0.95, recommended: true }, { category: 'FOODS', service_level: 0.9, recommended: false },
    ] as TwinFrontierRow[]
    expect(recommendedService(frontier).FOODS.service_level).toBe(0.95)
  })
})
