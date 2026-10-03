import { Suspense, lazy, useEffect, useState } from 'react'
import { ErrorBoundary, Loading } from './components/ui'

// The planning flow first (plan -> allocate -> stress-test), then the evidence behind the models.
const views = {
  overview: { label: 'Overview', group: 'flow', component: lazy(() => import('./views/Overview')) },
  forecast: { label: 'Plan: forecasts', group: 'flow', component: lazy(() => import('./views/Forecast')) },
  allocation: { label: 'Allocate', group: 'flow', component: lazy(() => import('./views/Allocation')) },
  twin: { label: 'Stress-test', group: 'flow', component: lazy(() => import('./views/Twin')) },
  accuracy: { label: 'Accuracy', group: 'evidence', component: lazy(() => import('./views/Accuracy')) },
  intervals: { label: 'Intervals', group: 'evidence', component: lazy(() => import('./views/Intervals')) },
  drift: { label: 'Drift', group: 'evidence', component: lazy(() => import('./views/Drift')) },
  method: { label: 'Method', group: 'evidence', component: lazy(() => import('./views/Method')) },
} as const
type ViewKey = keyof typeof views
const KEYS = Object.keys(views) as ViewKey[]

/** "#/twin?scenario=dc_cut" -> "twin"; the query is read by the view itself. */
const readRoute = (): ViewKey => {
  const key = window.location.hash.replace(/^#\/?/, '').split('?')[0] as ViewKey
  return key in views ? key : 'overview'
}

export default function App() {
  const [route, setRoute] = useState<ViewKey>(readRoute)
  useEffect(() => {
    const onHash = () => {
      const next = readRoute()
      setRoute((previous) => {
        if (previous !== next) window.scrollTo(0, 0)
        return next
      })
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => { document.title = `${views[route].label} · Demand Planning Platform` }, [route])
  const View = views[route].component
  const link = (key: ViewKey) => <a key={key} href={`#/${key}`} aria-current={key === route ? 'page' : undefined}>{views[key].label}</a>
  return (
    <div className="shell">
      <header className="top">
        <h1>Fewer empty shelves for the stock you hold</h1>
        <p className="muted">
          Demand planning for 3,049 Walmart products in 4 California stores: forecasts that add up at every level, a fair split when the warehouse is short,
          and an inventory simulator to test a policy before it reaches the shelf. Simulated as of 22 May 2016, the last date in the public M5 data.
        </p>
      </header>
      <nav className="tabs" aria-label="Views">
        {KEYS.filter((key) => views[key].group === 'flow').map(link)}
        <span className="tab-group" aria-hidden="true">Model evidence</span>
        {KEYS.filter((key) => views[key].group === 'evidence').map(link)}
      </nav>
      <label className="nav-select">
        <span>View</span>
        <select value={route} onChange={(event) => { window.location.hash = `#/${event.target.value}` }}>
          <optgroup label="Planning">{KEYS.filter((key) => views[key].group === 'flow').map((key) => <option key={key} value={key}>{views[key].label}</option>)}</optgroup>
          <optgroup label="Model evidence">{KEYS.filter((key) => views[key].group === 'evidence').map((key) => <option key={key} value={key}>{views[key].label}</option>)}</optgroup>
        </select>
      </label>
      <main>
        <ErrorBoundary key={route}>
          <Suspense fallback={<Loading />}><View /></Suspense>
        </ErrorBoundary>
      </main>
    </div>
  )
}
