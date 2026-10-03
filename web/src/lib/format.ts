export const LEVELS = ['total', 'state', 'store', 'category', 'department', 'item'] as const
export type Level = (typeof LEVELS)[number]
export const LEVEL_LABELS: Record<string, string> = {
  total: 'Total', state: 'State', store: 'Store', category: 'Category × store', department: 'Department × store', item: 'Item × store',
}
export const METHOD_LABELS: Record<string, string> = {
  base: 'Base forecast (unreconciled)', bottom_up: 'Bottom-up', top_down: 'Top-down',
  mint_diagonal: 'MinT, in-sample weights', mint_oos: 'MinT, out-of-sample weights',
}
export const MODEL_LABELS: Record<string, string> = {
  naive: 'Naive', seasonal_naive: 'Seasonal naive', lgbm_local: 'LightGBM local', lgbm_global: 'LightGBM global', sarima: 'SARIMA',
  sarima_daily_only: 'SARIMA (daily only)', prophet: 'Prophet',
}
export const POLICY_LABELS: Record<string, string> = {
  forecast_reorder: 'Forecast-based reorder', last_week_reorder: 'Last-week reorder (current practice)',
}
export const RATIONING_LABELS: Record<string, string> = {
  days_of_cover: 'Share stock by days of cover', proportional: 'Share stock proportionally', value: 'Share stock by value',
}

export const pct = (value: number, digits = 1): string => `${(value * 100).toFixed(digits)}%`
export const signedPct = (value: number, digits = 1): string => `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(digits)}%`
export const num = (value: number, digits = 0): string =>
  value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
export const usd = (value: number): string => {
  const abs = Math.abs(value)
  const sign = value < 0 ? '−' : ''
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}k`
  return `${sign}$${abs.toFixed(0)}`
}
export const signedUsd = (value: number): string => (value > 0 ? `+${usd(value)}` : usd(value))
export const points = (value: number, digits = 1): string => `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(digits)} pts`

/** "department:dept_id=FOODS_3|store_id=CA_1" -> "dept: FOODS_3 · store: CA_1"; the state total reads "All CA stores". */
export function nodeLabel(nodeId: string): string {
  const rest = nodeId.slice(nodeId.indexOf(':') + 1)
  return rest === 'total' ? 'All CA stores' : rest.replace(/_id=/g, ': ').replace(/\|/g, ' · ')
}
export const storeOf = (nodeId: string): string => nodeId.match(/store_id=([A-Z]{2}_\d)/)?.[1] ?? ''
export const deptOf = (nodeId: string): string => nodeId.match(/dept_id=([A-Z]+_\d)/)?.[1] ?? ''

/** Relative gap to the best value in a row, 0 for the best: drives the heatmap shading. */
export function relativeGap(values: number[]): number[] {
  const best = Math.min(...values.filter((value) => Number.isFinite(value) && value > 0))
  return values.map((value) => (Number.isFinite(value) && best > 0 ? value / best - 1 : 0))
}
