import { useEffect, useState } from 'react'

export function useData<T>(loader: () => Promise<T>, deps: unknown[] = []): { data?: T; error?: string; loading: boolean } {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true })
  useEffect(() => {
    let alive = true
    setState((previous) => ({ ...previous, loading: true, error: undefined }))
    loader().then(
      (data) => alive && setState({ data, loading: false }),
      (error: unknown) => alive && setState({ loading: false, error: error instanceof Error ? error.message : String(error) }),
    )
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}


/** Heatmap-style shading: darker = further behind the best in the row. */
export const shade = (gap: number): string => `color-mix(in srgb, var(--accent) ${Math.round(Math.min(gap, 1.5) / 1.5 * 38)}%, transparent)`
