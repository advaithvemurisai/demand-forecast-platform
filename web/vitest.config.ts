import { defineConfig } from 'vitest/config'

// Separate from vite.config.ts: the tests are pure TypeScript and Pyodide, so they need no React plugin.
export default defineConfig({
  test: { environment: 'node', testTimeout: 180_000, include: ['src/**/*.test.ts'] },
})
