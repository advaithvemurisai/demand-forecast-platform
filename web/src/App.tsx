import { Suspense, lazy, useEffect, useState } from 'react'
import { ErrorBoundary, Loading } from './components/ui'

const views = {
  overview: { label: 'Overview', component: lazy(() => import('./views/Overview')) },
  twin: { label: 'Inventory twin', component: lazy(() => import('./views/Twin')) },
  forecast: { label: 'Forecast explorer', component: lazy(() => import('./views/Forecast')) },
  accuracy: { label: 'Accuracy', component: lazy(() => import('./views/Accuracy')) },
  intervals: { label: 'Intervals', component: lazy(() => import('./views/Intervals')) },
  allocation: { label: 'Allocation', component: lazy(() => import('./views/Allocation')) },
  drift: { label: 'Drift', component: lazy(() => import('./views/Drift')) },
  method: { label: 'Method', component: lazy(() => import('./views/Method')) },
} as const
type ViewKey = keyof typeof views

const readRoute = (): ViewKey => {
  const key = window.location.hash.replace(/^#\/?/, '') as ViewKey
  return key in views ? key : 'overview'
}

export default function App() {
  const [route, setRoute] = useState<ViewKey>(readRoute)
  useEffect(() => {
    const onHash = () => { setRoute(readRoute()); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => { document.title = `${views[route].label} · Demand Forecasting Platform` }, [route])
  const View = views[route].component
  return (
    <div className="shell">
      <header className="top">
        <h1>Hierarchical demand forecasting, allocation and inventory twin</h1>
        <p className="muted">
          Walmart M5, California: 12,196 item-store series across 4 stores, forecast at 6 hierarchy levels, with a day-by-day simulation of the warehouse-to-store chain.
        </p>
      </header>
      <nav className="tabs" aria-label="Views">
        {(Object.keys(views) as ViewKey[]).map((key) => (
          <a key={key} href={`#/${key}`} aria-current={key === route ? 'page' : undefined}>{views[key].label}</a>
        ))}
      </nav>
      <main>
        <ErrorBoundary key={route}>
          <Suspense fallback={<Loading />}><View /></Suspense>
        </ErrorBoundary>
      </main>
    </div>
  )
}
