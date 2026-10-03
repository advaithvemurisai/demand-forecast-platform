/** Planners live in spreadsheets: every decision table on the site can be taken away as CSV. */

type Cell = string | number | boolean | null | undefined

export interface CsvColumn<T> { header: string; value: (row: T) => Cell }

const escape = (cell: Cell): string => {
  if (cell === null || cell === undefined || (typeof cell === 'number' && !Number.isFinite(cell))) return ''
  const text = typeof cell === 'number' ? String(Math.round(cell * 1e4) / 1e4) : String(cell)
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

export function toCsv<T>(rows: T[], columns: CsvColumn<T>[]): string {
  return [columns.map((c) => escape(c.header)).join(','), ...rows.map((row) => columns.map((c) => escape(c.value(row))).join(','))].join('\r\n') + '\r\n'
}

export function downloadCsv(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob(['﻿', text], { type: 'text/csv;charset=utf-8' })) // BOM so Excel reads UTF-8
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
