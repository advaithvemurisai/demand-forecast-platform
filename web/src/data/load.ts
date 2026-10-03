const cache = new Map<string, Promise<unknown>>()

export const base = (): string => import.meta.env.BASE_URL

/** Fetch a JSON file from /data once per session; concurrent callers share one request. */
export function getJson<T>(path: string): Promise<T> {
  const url = `${base()}data/${path}`
  let pending = cache.get(url) as Promise<T> | undefined
  if (!pending) {
    pending = fetch(url).then((response) => {
      if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`)
      return response.json() as Promise<T>
    })
    pending.catch(() => cache.delete(url)) // let a retry actually retry
    cache.set(url, pending)
  }
  return pending
}

/** Like getJson but resolves to undefined when the file is absent (optional tables such as the twin's). */
export async function getOptional<T>(path: string): Promise<T | undefined> {
  try {
    return await getJson<T>(path)
  } catch {
    return undefined
  }
}
