import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { loadPyodide } from 'pyodide'

const root = resolve(__dirname, '../../..')
const twinSource = readFileSync(resolve(root, 'src/forecasting/twin.py'), 'utf8')
const fixtureSource = readFileSync(resolve(__dirname, 'fixtures/fixture.py'), 'utf8')
const snapshot = JSON.parse(readFileSync(resolve(__dirname, 'fixtures/snapshot.json'), 'utf8')) as Record<string, unknown>

/** Recursively compare, allowing tiny float differences between numpy builds. */
function expectClose(actual: unknown, expected: unknown, path = ''): void {
  if (typeof expected === 'number') {
    expect(typeof actual, path).toBe('number')
    expect(Math.abs((actual as number) - expected), `${path}: ${actual} vs ${expected}`).toBeLessThanOrEqual(1e-6 + 1e-6 * Math.abs(expected))
  } else if (Array.isArray(expected)) {
    expect(Array.isArray(actual), path).toBe(true)
    expect((actual as unknown[]).length, path).toBe(expected.length)
    expected.forEach((item, index) => expectClose((actual as unknown[])[index], item, `${path}[${index}]`))
  } else if (expected && typeof expected === 'object') {
    expect(Object.keys(actual as object).sort(), path).toEqual(Object.keys(expected).sort())
    for (const [key, value] of Object.entries(expected)) expectClose((actual as Record<string, unknown>)[key], value, `${path}.${key}`)
  } else {
    expect(actual, path).toEqual(expected)
  }
}

describe('browser simulator parity', () => {
  it('Pyodide reproduces the CPython reference results for every fixture request', async (context) => {
    const py = await loadPyodide({ indexURL: resolve(__dirname, '../../node_modules/pyodide') })
    try {
      await py.loadPackage('numpy')
    } catch (error) {
      console.warn('Skipping the Pyodide parity test: numpy could not be downloaded (offline?)', error)
      context.skip()
    }
    py.FS.mkdirTree('/home/pyodide')
    py.FS.writeFile('/home/pyodide/twin.py', twinSource)
    py.FS.writeFile('/home/pyodide/fixture.py', fixtureSource)
    py.runPython("import sys\nsys.path.insert(0, '/home/pyodide')")
    const produced = JSON.parse(
      py.runPython(
        'import json, fixture, twin\n' +
          'json.dumps({name: twin.simulate_bundle(fixture.bundle_for(request), request) for name, request in fixture.REQUESTS.items()})',
      ) as string,
    )
    expectClose(produced, snapshot)
  })
})
