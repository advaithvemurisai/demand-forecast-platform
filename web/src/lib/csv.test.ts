import { describe, expect, it } from 'vitest'
import { toCsv } from './csv'

describe('toCsv', () => {
  it('writes a header, quotes awkward cells and blanks non-finite numbers', () => {
    const rows = [{ item: 'FOODS_3_090', note: 'say "hi", then go', value: 1.234567 }, { item: 'X', note: '', value: NaN }]
    const text = toCsv(rows, [
      { header: 'Item', value: (r) => r.item }, { header: 'Note', value: (r) => r.note }, { header: 'Value', value: (r) => r.value },
    ])
    expect(text).toBe('Item,Note,Value\r\nFOODS_3_090,"say ""hi"", then go",1.2346\r\nX,,\r\n')
  })
})
