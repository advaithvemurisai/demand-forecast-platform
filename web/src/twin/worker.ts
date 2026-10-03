/// <reference lib="webworker" />
// Runs the Python twin (src/forecasting/twin.py, copied to /py/twin.py at build time) with Pyodide, off the main thread.
import type { TwinRequest, TwinResult } from '../data/types'

export type WorkerIn = { type: 'warm'; base: string } | { type: 'run'; id: number; base: string; store: string; request: TwinRequest }
export type WorkerOut =
  | { type: 'progress'; id: number; stage: 'runtime' | 'inputs' | 'simulating' }
  | { type: 'ready' }
  | { type: 'result'; id: number; result: TwinResult }
  | { type: 'error'; id: number; message: string }

const PYODIDE_VERSION = '0.29.5'
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

interface Pyodide {
  loadPackage(name: string): Promise<void>
  runPython(code: string): unknown
  FS: { writeFile(path: string, data: string | Uint8Array): void }
  globals: { get(name: string): (...args: unknown[]) => string }
}

const RUNNER = `
import sys, json
sys.path.insert(0, '/home/pyodide')
import numpy as np
import twin
_bundles = {}
def run(store, request_json):
    if store not in _bundles:
        with np.load('/home/pyodide/' + store + '.npz', allow_pickle=False) as d:
            bundle = {k: d[k] for k in d.files}
        bundle = {k: (v.item() if v.ndim == 0 else v) for k, v in bundle.items()}
        bundle.pop('dates', None)
        bundle.pop('item_ids', None)
        _bundles[store] = bundle
    return json.dumps(twin.simulate_bundle(_bundles[store], json.loads(request_json)))
`

let runtime: Promise<Pyodide> | undefined
const loadedStores = new Set<string>()

function init(base: string): Promise<Pyodide> {
  runtime ??= (async () => {
    const { loadPyodide } = (await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`)) as { loadPyodide: (options: { indexURL: string }) => Promise<Pyodide> }
    const py = await loadPyodide({ indexURL: INDEX_URL })
    await py.loadPackage('numpy')
    const source = await (await fetch(`${base}py/twin.py`)).text()
    py.FS.writeFile('/home/pyodide/twin.py', source)
    py.runPython(RUNNER)
    return py
  })().catch((error) => {
    runtime = undefined // allow a retry after a network failure
    throw error
  })
  return runtime
}

const post = (message: WorkerOut) => (self as unknown as Worker).postMessage(message)

self.onmessage = async (event: MessageEvent<WorkerIn>) => {
  const message = event.data
  try {
    if (message.type === 'warm') {
      await init(message.base)
      post({ type: 'ready' })
      return
    }
    post({ type: 'progress', id: message.id, stage: 'runtime' })
    const py = await init(message.base)
    if (!loadedStores.has(message.store)) {
      post({ type: 'progress', id: message.id, stage: 'inputs' })
      const response = await fetch(`${message.base}data/twin/inputs/${message.store}.npz`)
      if (!response.ok) throw new Error(`No inputs for store ${message.store}`)
      py.FS.writeFile(`/home/pyodide/${message.store}.npz`, new Uint8Array(await response.arrayBuffer()))
      loadedStores.add(message.store)
    }
    post({ type: 'progress', id: message.id, stage: 'simulating' })
    const run = py.globals.get('run')
    const result = JSON.parse(run(message.store, JSON.stringify(message.request))) as TwinResult
    post({ type: 'result', id: message.id, result })
  } catch (error) {
    post({ type: 'error', id: message.type === 'run' ? message.id : -1, message: error instanceof Error ? error.message : String(error) })
  }
}
