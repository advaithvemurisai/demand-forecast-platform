import type { TwinRequest, TwinResult } from '../data/types'
import type { WorkerIn, WorkerOut } from './worker'

export type Stage = 'runtime' | 'inputs' | 'simulating'
export const STAGE_LABELS: Record<Stage, string> = {
  runtime: 'Loading the Python simulator (one-time, about 12 MB)…',
  inputs: 'Loading this store’s forecast…',
  simulating: 'Simulating…',
}

let worker: Worker | undefined
let nextId = 1
const waiting = new Map<number, { resolve: (result: TwinResult) => void; reject: (error: Error) => void; progress?: (stage: Stage) => void }>()
const base = () => import.meta.env.BASE_URL

function ensureWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (event: MessageEvent<WorkerOut>) => {
      const message = event.data
      if (message.type === 'ready') return
      const entry = waiting.get(message.id)
      if (message.type === 'progress') entry?.progress?.(message.stage)
      if (message.type === 'result') { waiting.delete(message.id); entry?.resolve(message.result) }
      if (message.type === 'error') {
        waiting.delete(message.id)
        entry?.reject(new Error(message.message))
        if (message.id === -1) for (const pending of waiting.values()) pending.reject(new Error(message.message))
      }
    }
    worker.onerror = (event) => {
      for (const pending of waiting.values()) pending.reject(new Error(event.message || 'The simulator stopped unexpectedly'))
      waiting.clear()
      worker = undefined
    }
  }
  return worker
}

/** Start downloading the runtime in the background so the first custom run is quick. */
export function warmSimulator(): void {
  ensureWorker().postMessage({ type: 'warm', base: base() } satisfies WorkerIn)
}

export function runSimulation(store: string, request: TwinRequest, progress?: (stage: Stage) => void): Promise<TwinResult> {
  const id = nextId++
  return new Promise<TwinResult>((resolve, reject) => {
    waiting.set(id, { resolve, reject, progress })
    ensureWorker().postMessage({ type: 'run', id, base: base(), store, request } satisfies WorkerIn)
  })
}
