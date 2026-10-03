export const SERIES = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)']
export const INK = 'var(--ink)'
export const MUTED = 'var(--muted)'
export const GRID = 'var(--grid)'
export const axisProps = { stroke: 'var(--muted)', tick: { fill: 'var(--muted)', fontSize: 12 }, tickLine: false } as const
export const gridProps = { stroke: 'var(--grid)', vertical: false } as const
export const tooltipStyle = {
  contentStyle: { background: 'var(--card)', border: '1px solid var(--line)', borderRadius: 8, color: 'var(--ink)', fontSize: 13 },
  labelStyle: { color: 'var(--ink)', fontWeight: 600 },
} as const
