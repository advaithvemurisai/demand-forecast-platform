import { Component, type ReactNode } from 'react'
import { ResponsiveContainer } from 'recharts'
import { downloadCsv, toCsv, type CsvColumn } from '../lib/csv'

export type Tone = 'good' | 'bad' | 'flat'

export function Loading({ label = 'Loading' }: { label?: string }) {
  return <div className="skeleton" role="status" aria-label={label} />
}

export function Problem({ message }: { message: string }) {
  return <div className="callout warn" role="alert"><strong>Could not load this view.</strong> <span className="muted">{message}</span></div>
}

export function Async<T>({ state, children }: { state: { data?: T; error?: string; loading: boolean }; children: (data: T) => ReactNode }) {
  if (state.error) return <Problem message={state.error} />
  if (!state.data) return <Loading />
  return <>{children(state.data)}</>
}

export function Kpi({ label, value, delta, tone = 'flat', note }: { label: string; value: string; delta?: string; tone?: Tone; note?: string }) {
  // The arrow shows direction (from the delta's sign); the colour shows whether that direction is good.
  const arrow = delta?.startsWith('+') ? '▲ ' : /^[−-]/.test(delta ?? '') ? '▼ ' : ''
  return (
    <div className="card kpi">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
      {delta && <span className={`delta ${tone}`}>{arrow}{delta}</span>}
      {note && <span className="note">{note}</span>}
    </div>
  )
}

export function Question({ children }: { children: ReactNode }) {
  return <div className="question"><strong>Business question: </strong>{children}</div>
}

export function Segmented<T extends string>({ label, options, value, onChange }: {
  label: string; options: { value: T; label: string }[]; value: T; onChange: (value: T) => void
}) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="seg" role="group" aria-label={label}>
        {options.map((option) => (
          <button key={option.value} type="button" aria-pressed={option.value === value} onClick={() => onChange(option.value)}>{option.label}</button>
        ))}
      </div>
    </div>
  )
}

export function Select<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (value: T) => void
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value as T)}>
        {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  )
}

export function ChartBox({ children, size = '' }: { children: React.ReactElement; size?: 'tall' | 'short' | '' }) {
  return <div className={`chart ${size}`}><ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer></div>
}

export class ErrorBoundary extends Component<{ children: ReactNode }, { error?: Error }> {
  state: { error?: Error } = {}
  static getDerivedStateFromError(error: Error) { return { error } }
  render() {
    return this.state.error ? <Problem message={this.state.error.message} /> : this.props.children
  }
}

/** A small "Download CSV" button for a decision table. */
export function CsvButton<T>({ filename, rows, columns, label = 'Download CSV' }: { filename: string; rows: T[]; columns: CsvColumn<T>[]; label?: string }) {
  return (
    <button type="button" className="csv" onClick={() => downloadCsv(filename, toCsv(rows, columns))} disabled={rows.length === 0}>
      {label} <span className="muted">({rows.length.toLocaleString()} rows)</span>
    </button>
  )
}
