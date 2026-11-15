import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { resolveApiBase } from '../shared/lib/api'

resolveApiBase().finally(() => createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
))
