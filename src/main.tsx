import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Dashboard from './Dashboard.tsx'
import PublicArena from './PublicArena.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {new URLSearchParams(window.location.search).get('dashboard') === '1' ? <Dashboard /> : <PublicArena />}
  </StrictMode>,
)
