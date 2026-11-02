import { useEffect, useState } from 'react'
import App from './App'
import type { WebBot } from './neural/types'
import { SIMULATIONS } from './presets'
import './public-arena.css'

function parseBots(value: unknown): WebBot[] {
  const entries: unknown = Array.isArray(value) ? value : value && typeof value === 'object' && 'bots' in value ? value.bots : null
  if (!Array.isArray(entries)) throw new Error('The bot catalogue has an invalid format.')
  const ids = new Set<string>()
  return entries.map((entry: unknown) => {
    if (!entry || typeof entry !== 'object') throw new Error('The bot catalogue contains an invalid bot.')
    const bot = entry as Partial<WebBot>
    if (typeof bot.id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(bot.id) || ids.has(bot.id)
      || typeof bot.name !== 'string' || !bot.name.trim() || typeof bot.url !== 'string' || !bot.url
      || typeof bot.sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(bot.sha256)
      || typeof bot.bytes !== 'number' || !Number.isSafeInteger(bot.bytes) || bot.bytes < 1
      || bot.planes !== 19 || typeof bot.simulations !== 'number'
      || !SIMULATIONS.includes(bot.simulations)) {
      throw new Error('The bot catalogue contains an invalid or unsupported bot.')
    }
    ids.add(bot.id)
    return bot as WebBot
  })
}

export default function PublicArena() {
  const [bots, setBots] = useState<WebBot[] | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    void fetch(`${import.meta.env.BASE_URL}bots.json`, { signal: controller.signal, cache: 'no-cache' })
      .then(response => {
        if (!response.ok) throw new Error(`Could not load the bot catalogue (${response.status}).`)
        return response.json() as Promise<unknown>
      })
      .then(data => { if (!controller.signal.aborted) setBots(parseBots(data)) })
      .catch(reason => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load the bot catalogue.')
          setBots([])
        }
      })
    return () => controller.abort()
  }, [attempt])

  if (bots === null) return <div className="app"><h1>Quoridoraner</h1><p role="status">Loading the arena…</p></div>
  return <>
    {error && <div className="app"><div className="arena-error" role="alert">{error} The built-in engine is ready to play. <button onClick={() => { setError(''); setAttempt(value => value + 1) }}>Retry catalogue</button></div></div>}
    <App publicArena bots={bots}/>
  </>
}
