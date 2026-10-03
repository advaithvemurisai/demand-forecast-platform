import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { ColumnarForecast, ForecastIndex, KpiBand, Manifest, TwinPresetIndex, TwinPresets } from './types'

// These check the files scripts/build_web_data.py wrote. They are skipped before the data is built
// (CI builds it first), so a fresh clone can still run `npm test`.
const dir = resolve(__dirname, '../../public/data')
const read = <T,>(path: string): T => JSON.parse(readFileSync(resolve(dir, path), 'utf8')) as T

describe.skipIf(!existsSync(resolve(dir, 'manifest.json')))('built data contract', () => {
  it('lists only files that exist and stays under the size budget', () => {
    const manifest = read<Manifest>('manifest.json')
    expect(manifest.schema).toBe(1)
    for (const file of manifest.files) expect(existsSync(resolve(dir, file)), file).toBe(true)
    expect(manifest.bytes).toBeLessThan(25_000_000)
    expect(Object.keys(manifest.products_by_category).sort()).toEqual(['FOODS', 'HOBBIES', 'HOUSEHOLD'])
  })

  it('has well-formed core tables', () => {
    const recon = read<{ method: string; level: string; wmape: number }[]>('reconciliation_metrics.json')
    expect(recon.length).toBeGreaterThan(50)
    expect(recon.every((row) => typeof row.method === 'string' && Number.isFinite(row.wmape))).toBe(true)
    const summary = read<{ served_method: string; wrmsse_holdout: Record<string, number> }>('summary.json')
    expect(summary.wrmsse_holdout[summary.served_method]).toBeGreaterThan(0)
    expect(read<unknown[]>('allocation.json')).toHaveLength(28)
  })

  it('splits item forecasts into rectangular per-store files', () => {
    const index = read<ForecastIndex>('forecast/index.json')
    expect(index.item_files.length).toBeGreaterThanOrEqual(20)
    const first = read<ColumnarForecast>(`forecast/${index.item_files[0].file}`)
    expect(first.dates).toHaveLength(28)
    for (const key of ['forecast', 'lower_95', 'upper_95'] as const) {
      expect(first[key]).toHaveLength(first.ids.length)
      expect(first[key].every((row) => row.length === 28)).toBe(true)
    }
    for (const level of ['total', 'state', 'store', 'category', 'department']) expect(index.levels[level].length).toBeGreaterThan(0)
  })

  it.skipIf(!existsSync(resolve(dir, 'twin/presets.json')))('has presets whose KPI bands and timelines are consistent', () => {
    const index = read<TwinPresetIndex>('twin/presets.json')
    expect(index.stores.length).toBeGreaterThan(0)
    const presets = read<TwinPresets>(`twin/presets/${index.stores[0]}.json`)
    expect(presets.dates).toHaveLength(28)
    for (const policy of index.policies) {
      expect(presets.results[`baseline|${policy}`].scenario).toBeNull()
      for (const key of Object.keys(index.presets)) {
        const result = presets.results[`${key}|${policy}`]
        expect(result.scenario, key).not.toBeNull()
        const kpis = result.scenario!.kpis
        for (const metric of ['fill_rate', 'in_stock_pct', 'lost_sales_value', 'inventory_value']) {
          const band: KpiBand = kpis[metric]
          expect(band.lower, metric).toBeLessThanOrEqual(band.mean + 1e-9)
          expect(band.mean, metric).toBeLessThanOrEqual(band.upper + 1e-9)
        }
        const timeline = result.scenario!.timeline
        expect(timeline.on_hand).toHaveLength(28)
        expect(timeline.on_hand[0]).toHaveLength(timeline.dept.length)
      }
    }
    expect(readdirSync(resolve(dir, 'twin/inputs')).some((name) => name.endsWith('.npz'))).toBe(true)
  })
})
