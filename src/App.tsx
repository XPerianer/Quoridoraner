import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Board, moveToString, START_WALLS } from './engine/board'
import { EngineClient } from './engine/client'
import { DEFAULT_OPTIONS } from './engine/search'
import type { Move, MoveGen, Player, SearchOptions, SearchSnapshot } from './engine/types'
import { BoardView, type Overlay } from './components/BoardView'
import { SearchPanel } from './components/SearchPanel'
import { api, type Run } from './dashboard-api'

type Controller = string
const NAMES = ['White', 'Black'] as const
const MIN_ENGINE_MOVE_MS = 400
const NODE_LIMITS = [10_000, 50_000, 200_000, 1_000_000, 5_000_000, DEFAULT_OPTIONS.maxNodes]

interface Search {
  snapshot: SearchSnapshot
  ply: number
}

function replay(history: Move[]): Board {
  const b = Board.initial()
  for (const m of history) b.make(m)
  return b
}

export default function App({ runs = [] }: { runs?: Run[] }) {
  const [history, setHistory] = useState<Move[]>([])
  // ?white=…&black=…&sims=… preselects the arena players (values as in the player menus).
  const query = new URLSearchParams(window.location.search)
  const [controllers, setControllers] = useState<[Controller, Controller]>([query.get('white') ?? 'human', query.get('black') ?? 'engine'])
  const [options, setOptions] = useState<SearchOptions>(DEFAULT_OPTIONS)
  const [paused, setPaused] = useState(false)
  const [overlay, setOverlay] = useState<Overlay>('candidates')
  const [showPv, setShowPv] = useState(true)
  const [allWalls, setAllWalls] = useState(false)
  const [search, setSearch] = useState<Search | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [highlight, setHighlight] = useState<Move | null>(null)
  const [botError, setBotError] = useState('')
  const [simulations, setSimulations] = useState(Number(query.get('sims')) || 64)
  const positions = useMemo(() => {
    const b = Board.initial(); const result = [b.toPosition()]
    for (const move of history) { b.make(move); result.push(b.toPosition()) }
    return result
  }, [history])
  const cutoff = history.length >= 200 ? 'Move limit (200 plies)' : positions.filter(p => JSON.stringify(p) === JSON.stringify(positions.at(-1))).length >= 3 ? 'Threefold repetition' : null

  const board = useMemo(() => replay(history), [history])
  const winner = board.winner()
  const turn = board.turn as Player
  const engineToMove = winner === -1 && controllers[turn] !== 'human' && !cutoff
  const thinking = engineToMove && !paused && !botError

  const client = useRef<EngineClient | null>(null)
  useEffect(() => {
    client.current = new EngineClient()
    return () => client.current!.dispose()
  }, [])

  const setOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => setOptions((o) => ({ ...o, [key]: value }))

  const runSearch = useCallback(
    (onDone: (s: SearchSnapshot | null) => void) => {
      const ply = history.length
      void client.current!.search(board.toPosition(), options, (snapshot) => setSearch({ snapshot, ply })).then((result) => {
        if (result) setSearch({ snapshot: result, ply })
        onDone(result)
      })
    },
    [board, history.length, options],
  )

  useEffect(() => {
    if (!thinking) return
    const started = performance.now()
    let timer = 0
    let active = true
    const controller = new AbortController()
    const finish = (move: Move) => {
      if (!active) return
      if (!board.legalMoves().includes(move)) { setBotError('Bot returned an illegal move'); return }
      timer = window.setTimeout(() => { if (active) setHistory(h => [...h, move]) }, Math.max(0, MIN_ENGINE_MOVE_MS - (performance.now() - started)))
    }
    if (controllers[turn] === 'engine') {
      runSearch(result => { if (result) finish(result.bestMove) })
    } else {
      const [run, mode] = JSON.parse(controllers[turn]) as [string, string]
      void api<{ move: number }>('/api/move', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ run, mode, simulations, positions }),
      }).then(result => finish(result.move)).catch(e => { if (active) setBotError(e.message) })
    }
    return () => {
      active = false
      controller.abort()
      client.current!.cancel()
      clearTimeout(timer)
    }
  }, [thinking, runSearch, board, controllers, turn, simulations, positions])

  const analyze = () => {
    setAnalyzing(true)
    runSearch(() => setAnalyzing(false))
  }

  const stopAnalysis = () => {
    client.current!.cancel()
    setAnalyzing(false)
  }

  const play = (m: Move) => {
    stopAnalysis()
    setHistory((h) => [...h, m])
  }

  const undo = () => {
    stopAnalysis()
    setHistory((h) => {
      let n = h.length - 1
      const humanIdx = controllers.indexOf('human')
      if (humanIdx >= 0 && controllers[0] !== controllers[1]) while (n > 0 && n % 2 !== humanIdx) n--
      return h.slice(0, Math.max(0, n))
    })
  }

  const newGame = () => {
    stopAnalysis()
    setSearch(null)
    setBotError('')
    setHistory([])
  }

  const stale = search !== null && search.ply !== history.length
  const liveSnapshot = search && !stale ? search.snapshot : null
  const humanToMove = winner === -1 && !cutoff && controllers[turn] === 'human'
  const lastMove = history.length ? history[history.length - 1] : null

  const status =
    winner !== -1
      ? `${NAMES[winner]} wins`
      : cutoff
        ? `Draw adjudication: ${cutoff}`
      : thinking
        ? `${NAMES[turn]} (engine) is thinking…`
        : engineToMove && paused
          ? `${NAMES[turn]} (engine) to move — paused`
          : `${NAMES[turn]} to move`

  return (
    <div className="app">
      <header className="top">
        <h1>Quoridoraner</h1>
        <span className="muted">Choose a human, the built-in engine, or an exported neural bot for either side</span>
      </header>
      {botError && <div className="dash-error" role="alert">{botError} <button onClick={() => setBotError('')}>Retry move</button></div>}
      <main>
        <section className="play-col">
          <div className="status-row">
            <span className={`turn-dot pawn${winner === -1 ? turn : winner}`} />
            <span className="status">{status}</span>
          </div>
          <div className="walls-row">
            {([0, 1] as const).map((p) => (
              <div key={p} className={`walls-left ${turn === p && winner === -1 ? 'active' : ''}`}>
                <span className={`turn-dot pawn${p}`} />
                <span>{NAMES[p]}</span>
                <span className="pips" aria-label={`${board.wallsLeft[p]} walls left`}>
                  {Array.from({ length: START_WALLS }, (_, i) => (
                    <span key={i} className={i < board.wallsLeft[p] ? 'pip on' : 'pip'} />
                  ))}
                </span>
              </div>
            ))}
          </div>
          <BoardView
            board={board}
            interactive={humanToMove}
            onMove={play}
            lastMove={lastMove}
            snapshot={liveSnapshot}
            overlay={overlay}
            showPv={showPv}
            allWalls={allWalls}
            highlight={highlight}
            onHighlight={setHighlight}
          />
          <div className="button-row">
            <button onClick={newGame}>New game</button>
            <button onClick={undo} disabled={history.length === 0}>
              Undo
            </button>
            {humanToMove && (
              <button onClick={() => (analyzing ? client.current!.stop() : analyze())}>{analyzing ? 'Stop analysis' : 'Analyze'}</button>
            )}
            {thinking && controllers[turn] === 'engine' && <button onClick={() => client.current!.stop()}>Move now</button>}
            {controllers[0] !== 'human' && controllers[1] !== 'human' && (
              <button onClick={() => setPaused(!paused)}>{paused ? 'Resume' : 'Pause'}</button>
            )}
          </div>

          <div className="panel settings">
            <div className="field-row">
              {([0, 1] as const).map((p) => (
                <label key={p}>
                  {NAMES[p]}
                  <select
                    value={controllers[p]}
                    onChange={(e) => {
                      const next = [...controllers] as [Controller, Controller]
                      next[p] = e.target.value as Controller
                      setBotError('')
                      setControllers(next)
                    }}
                  >
                    <option value="human">Human</option>
                    <option value="engine">Built-in negamax</option>
                    <option value={JSON.stringify(['', 'negamax:2000'])}>Reference negamax (C++, 2000 nodes)</option>
                    {runs.flatMap(run => run.bots.map(mode => <option key={`${run.id}/${mode}`} value={JSON.stringify([run.id,mode])}>{run.id} / {mode}</option>))}
                  </select>
                </label>
              ))}
              <label>
                Neural simulations
                <select value={simulations} onChange={e => setSimulations(Number(e.target.value))}>
                  {[8,16,32,64,128,256,400,512].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <label>
                Think time
                <select value={options.timeMs} onChange={(e) => setOption('timeMs', Number(e.target.value))}>
                  {[250, 500, 1000, 2000, 5000, 10000, 30000].map((t) => (
                    <option key={t} value={t}>
                      {t / 1000}s
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Max depth
                <select value={options.maxDepth} onChange={(e) => setOption('maxDepth', Number(e.target.value))}>
                  {[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20].map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Walls considered
                <select value={options.moveGen} onChange={(e) => setOption('moveGen', e.target.value as MoveGen)}>
                  <option value="all">All legal</option>
                  <option value="likely">Near pawns/walls</option>
                </select>
              </label>
              <label>
                Node limit
                <select value={options.maxNodes} onChange={(e) => setOption('maxNodes', Number(e.target.value))}>
                  {NODE_LIMITS.map((n) => (
                    <option key={n} value={n}>
                      {n === DEFAULT_OPTIONS.maxNodes ? 'None' : n.toLocaleString()}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="field-row">
              <label title="Eval points per wall-in-hand advantage; one path step = 10">
                Wall value
                <input type="number" step={1} value={options.wallWeight} onChange={(e) => setOption('wallWeight', e.target.valueAsNumber || 0)} />
              </label>
              <label title="Eval bonus for the side to move; one path step = 10">
                Tempo
                <input type="number" step={1} value={options.tempo} onChange={(e) => setOption('tempo', e.target.valueAsNumber || 0)} />
              </label>
              <label className="check">
                <input type="checkbox" checked={options.exactRoot} onChange={(e) => setOption('exactRoot', e.target.checked)} />
                Exact score for every candidate (slower)
              </label>
              <button className="link" onClick={() => setOptions(DEFAULT_OPTIONS)}>
                Reset engine settings
              </button>
            </div>
            <div className="field-row">
              <span className="muted small">Board overlay</span>
              {(['candidates', 'visits', 'none'] as const).map((o) => (
                <label key={o} className="check">
                  <input type="radio" name="overlay" checked={overlay === o} onChange={() => setOverlay(o)} />
                  {o === 'candidates' ? 'Candidate scores' : o === 'visits' ? 'Search visits' : 'None'}
                </label>
              ))}
              <label className="check">
                <input type="checkbox" checked={showPv} onChange={(e) => setShowPv(e.target.checked)} />
                PV
              </label>
              {overlay === 'candidates' && (
                <label className="check">
                  <input type="checkbox" checked={allWalls} onChange={(e) => setAllWalls(e.target.checked)} />
                  all walls
                </label>
              )}
            </div>
          </div>

          <div className="panel history">
            <h3>Moves</h3>
            <ol>
              {Array.from({ length: Math.ceil(history.length / 2) }, (_, i) => (
                <li key={i}>
                  <span>{moveToString(history[2 * i])}</span>
                  <span>{history[2 * i + 1] !== undefined ? moveToString(history[2 * i + 1]) : ''}</span>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <SearchPanel snapshot={search?.snapshot ?? null} stale={stale} highlight={highlight} onHighlight={setHighlight} maxDepth={options.maxDepth} />
      </main>
    </div>
  )
}
