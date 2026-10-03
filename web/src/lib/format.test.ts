import { describe, expect, it } from 'vitest'
import { nodeLabel, num, pct, relativeGap, shortDate, signedPct, storeOf, usd } from './format'

describe('format', () => {
  it('formats percentages with explicit signs', () => {
    expect(pct(0.9534)).toBe('95.3%')
    expect(signedPct(0.016)).toBe('+1.6%')
    expect(signedPct(-0.047)).toBe('−4.7%')
  })
  it('abbreviates dollars', () => {
    expect(usd(950)).toBe('$950')
    expect(usd(44241)).toBe('$44.2k')
    expect(usd(-2_500_000)).toBe('−$2.50M')
  })
  it('groups thousands', () => expect(num(12196)).toBe('12,196'))
  it('labels hierarchy nodes', () => {
    expect(nodeLabel('department:dept_id=FOODS_3|store_id=CA_1')).toBe('dept: FOODS_3 · store: CA_1')
    expect(nodeLabel('total:total')).toBe('All CA stores')
    expect(storeOf('item:item_id=FOODS_3_090|store_id=CA_2')).toBe('CA_2')
  })
  it('measures the gap to the best in a row', () => {
    const gaps = relativeGap([0.2, 0.1, 0.15])
    expect(gaps[1]).toBe(0)
    expect(gaps[0]).toBeCloseTo(1)
    expect(gaps[2]).toBeCloseTo(0.5)
  })
})

describe('shortDate', () => {
  it('formats ISO dates without shifting the day', () => {
    expect(shortDate('2016-05-23')).toBe('23 May')
    expect(shortDate('2016-06-19T00:00:00')).toBe('19 Jun')
  })
})
