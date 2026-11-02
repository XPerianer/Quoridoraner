import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Board, moveToString, START_WALLS } from './engine/board'
import { EngineClient } from './engine/client'
import { DEFAULT_OPTIONS } from './engine/search'
import type { Move, Player, SearchOptions, SearchSnapshot } from './engine/types'
import { BoardView, type Overlay } from './components/BoardView'
import { EngineOptionsFields } from './components/EngineOptionsFields'
import { Icon, IconButton } from './components/Icons'
import { PlaySetup, WatchSetup } from './components/PresetControls'
import { SearchPanel } from './components/SearchPanel'
import { SharePanel } from './components/SharePanel'
import { StartScreen } from './components/StartScreen'
import { api, type Run } from './dashboard-api'
import { NeuralClient } from './neural/client'
import type { WebBot } from './neural/types'
import { controllerName, matchPreset, presetSide, SIMULATIONS, STRENGTH_NAMES, type SideConfig } from './presets'
import { baseUrl, readShare, shareUrl } from './share'

type Sides = [SideConfig, SideConfig]
const NAMES = ['White', 'Black'] as const
const EMPTY_BOTS: WebBot[] = []
const MIN_ENGINE_MOVE_MS = 400
const REPLAY_MOVE_MS = 700

interface Search {
  snapshot: SearchSnapshot
  ply: number
  maxDepth: number
}

function replay(history: Move[]): Board {
  const b = Board.initial()
  for (const m of history) b.make(m)
  return b
}

export default function App({ runs = [], bots = EMPTY_BOTS, publicArena = false }: { runs?: Run[]; bots?: WebBot[]; publicArena?: boolean }) {
  // ?white=…&black=… preselect the players and ?moves=… replays a recorded game (e.g. from external_arena.py
  // records or a shared link); share.ts documents the full link format.
  const [shared] = useState(() => readShare(window.location.search, bots, publicArena))
  const [screen, setScreen] = useState<'start' | 'game'>(publicArena && !shared.fromLink ? 'start' : 'game')
  const [recorded, setRecorded] = useState<Move[]>(shared.moves)
  const [replaying, setReplaying] = useState(shared.moves.length > 0)
  const [history, setHistory] = useState<Move[]>([])
  const [sides, setSides] = useState<Sides>(shared.sides)
  const [analysisOptions, setAnalysisOptions] = useState<SearchOptions>(DEFAULT_OPTIONS)
  const [paused, setPaused] = useState(false)
  const [overlay, setOverlay] = useState<Overlay>('candidates')
  const [showPv, setShowPv] = useState(true)
  const [allWalls, setAllWalls] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(!publicArena)
  const [analysisOpen, setAnalysisOpen] = useState(!publicArena)
  const [sharing, setSharing] = useState(false)
  const [search, setSearch] = useState<Search | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [highlight, setHighlight] = useState<Move | null>(null)
  const [botError, setBotError] = useState('')
  const [botStatus, setBotStatus] = useState('')
  const [generation, setGeneration] = useState(0)
  const requestGeneration = useRef(0)
  const neural = useRef<NeuralClient | null>(null)
  const positions = useMemo(() => {
    const b = Board.initial(); const result = [b.toPosition()]
    for (const move of history) { b.make(move); result.push(b.toPosition()) }
    return result
  }, [history])
  const cutoff = history.length >= 200 ? 'Move limit (200 plies)' : positions.filter(p => JSON.stringify(p) === JSON.stringify(positions.at(-1))).length >= 3 ? 'Threefold repetition' : null

  const board = useMemo(() => replay(history), [history])
  const winner = board.winner()
  const turn = board.turn as Player
  const mover = sides[turn]
  const engineToMove = winner === -1 && mover.controller !== 'human' && !cutoff
  const thinking = engineToMove && !paused && !botError && screen === 'game'

  const client = useRef<EngineClient | null>(null)
  useEffect(() => {
    client.current = new EngineClient()
    neural.current = new NeuralClient()
    return () => { client.current!.dispose(); neural.current!.dispose() }
  }, [])

  const runSearch = useCallback(
    (searchOptions: SearchOptions, onDone: (s: SearchSnapshot | null) => void) => {
      const ply = history.length
      const maxDepth = searchOptions.maxDepth
      void client.current!.search(board.toPosition(), searchOptions, (snapshot) => setSearch({ snapshot, ply, maxDepth })).then((result) => {
        if (result) setSearch({ snapshot: result, ply, maxDepth })
        onDone(result)
      })
    },
    [board, history.length],
  )

  useEffect(() => {
    if (!thinking) return
    const started = performance.now()
    let timer = 0
    let active = true
    const request = requestGeneration.current
    const controller = new AbortController()
    const finish = (move: Move) => {
      if (!active || request !== requestGeneration.current) return
      if (!board.legalMoves().includes(move)) { setBotError('Bot returned an illegal move'); return }
      timer = window.setTimeout(() => { if (active && request === requestGeneration.current) setHistory(h => [...h, move]) }, Math.max(0, MIN_ENGINE_MOVE_MS - (performance.now() - started)))
    }
    if (mover.controller === 'engine') {
      runSearch(mover.engine, result => { if (result) finish(result.bestMove) })
    } else if (mover.controller.startsWith('browser:')) {
      const bot = bots.find(item => `browser:${item.id}` === mover.controller)
      if (!bot) { setBotError('This browser bot is unavailable. Choose another player.'); return }
      void neural.current!.search(bot, positions, mover.simulations, message => {
        if (active && request === requestGeneration.current) setBotStatus(message)
      }).then(result => finish(result.move)).catch(e => {
        if (active && request === requestGeneration.current) setBotError(e instanceof Error ? e.message : String(e))
      })
    } else if (!publicArena) {
      const [run, mode] = JSON.parse(mover.controller) as [string, string]
      void api<{ move: number }>('/api/move', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ run, mode, simulations: mover.simulations, positions }),
      }).then(result => finish(result.move)).catch(e => { if (active && request === requestGeneration.current) setBotError(e.message) })
    }
    return () => {
      active = false
      controller.abort()
      client.current!.cancel()
      neural.current!.cancel()
      clearTimeout(timer)
    }
  }, [thinking, runSearch, board, mover, turn, positions, bots, publicArena, generation])

  // Replay follows the recorded game only while the board is still on it.
  const onRecord = history.length < recorded.length && history.every((m, i) => m === recorded[i])
  const autoplay = replaying && onRecord && screen === 'game'
  useEffect(() => {
    if (!autoplay) return
    const timer = window.setTimeout(() => setHistory(h => recorded.slice(0, h.length + 1)), REPLAY_MOVE_MS)
    return () => clearTimeout(timer)
  }, [autoplay, history, recorded])

  const stepReplay = (delta: number) => {
    stopAnalysis()
    setReplaying(false)
    setHistory(h => recorded.slice(0, Math.max(0, Math.min(recorded.length, h.length + delta))))
  }

  const analyze = () => {
    setAnalyzing(true)
    runSearch(analysisOptions, () => setAnalyzing(false))
  }

  const stopAnalysis = () => {
    requestGeneration.current++
    neural.current!.cancel()
    setBotStatus('')
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
      const humanIdx = sides.findIndex(side => side.controller === 'human')
      if (humanIdx >= 0 && sides[0].controller !== sides[1].controller) while (n > 0 && n % 2 !== humanIdx) n--
      return h.slice(0, Math.max(0, n))
    })
  }

  const newGame = () => {
    stopAnalysis()
    setSearch(null)
    setBotError('')
    setReplaying(false)
    setHistory([])
    setGeneration(value => value + 1)
  }

  // Player changes restart the pending bot move, which stopAnalysis invalidates.
  const updateSides = (update: (sides: Sides) => Sides) => {
    stopAnalysis()
    setBotError('')
    setSides(update)
    setGeneration(value => value + 1)
  }
  const setSide = (p: Player, side: SideConfig) => updateSides(current => (p ? [current[0], side] : [side, current[1]]))

  const start = (next: Sides) => {
    newGame()
    setPaused(false)
    setRecorded([])
    setSharing(false)
    setSides(next)
    setScreen('game')
    window.history.replaceState(null, '', baseUrl(window.location))
  }

  const home = () => {
    stopAnalysis()
    setSharing(false)
    setScreen('start')
  }

  const stale = search !== null && search.ply !== history.length
  const liveSnapshot = search && !stale ? search.snapshot : null
  const humanToMove = winner === -1 && !cutoff && mover.controller === 'human'
  const canAnalyze = winner === -1 && !cutoff && !thinking && !autoplay
  const lastMove = history.length ? history[history.length - 1] : null

  const humans = sides.filter(side => side.controller === 'human').length
  const mode = humans === 1 ? 'play' : humans === 0 ? 'watch' : 'local'
  const you = sides[0].controller === 'human' ? 0 : 1
  const strengths = sides.map(side => (side.controller === 'human' ? null : matchPreset(side, bots))) as [ReturnType<typeof matchPreset>, ReturnType<typeof matchPreset>]
  const botName = (p: Player) => { const strength = strengths[p]; return strength ? `${STRENGTH_NAMES[strength]} bot` : controllerName(sides[p].controller, bots) }
  const who = (p: Player) => (sides[p].controller === 'human' ? NAMES[p] : `${NAMES[p]} (${botName(p)})`)
  const matchup = mode === 'play' ? `You (${NAMES[you]}) vs ${botName((1 - you) as Player)}` : mode === 'watch' ? `${botName(0)} vs ${botName(1)}` : 'Two players on this device'

  const status =
    winner !== -1
      ? mode === 'play' ? (winner === you ? 'You win!' : `${botName(winner as Player)} wins`) : `${NAMES[winner]} wins`
      : cutoff
        ? `Draw adjudication: ${cutoff}`
      : thinking
        ? `${who(turn)} is thinking…`
        : engineToMove && paused
          ? `${who(turn)} to move — paused`
          : humanToMove && mode === 'play'
            ? 'Your move'
            : `${NAMES[turn]} to move`

  if (screen === 'start') {
    return (
      <div className="app">
        <header className="top">
          <h1>Quoridoraner</h1>
          <span className="muted">Play a bot. Watch them compete.</span>
        </header>
        <StartScreen bots={bots} onStart={start} onResume={history.length ? () => setScreen('game') : undefined} />
      </div>
    )
  }

  return (
    <div className="app">
      <header className="top">
        {publicArena ? <button type="button" className="brand" onClick={home} title="Back to the start screen"><h1>Quoridoraner</h1></button> : <h1>Quoridoraner</h1>}
        <span className="mode-chip">{matchup}</span>
      </header>
      {botError && <div className="arena-error" role="alert">{botError} <button onClick={() => setBotError('')}>Retry move</button></div>}
      <main>
        <section className="play-col">
          <div className="status-row">
            <span className={`turn-dot pawn${winner === -1 ? turn : winner}`} />
            <span className="status">{status}</span>
            {thinking && botStatus && <span className="arena-runtime-status small" role="status" aria-live="polite">{botStatus}</span>}
          </div>
          <div className="walls-row">
            {([0, 1] as const).map((p) => (
              <div key={p} className={`walls-left ${turn === p && winner === -1 ? 'active' : ''}`}>
                <span className={`turn-dot pawn${p}`} />
                <span>{mode === 'play' && p === you ? `You (${NAMES[p]})` : NAMES[p]}</span>
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
            overlay={analysisOpen ? overlay : 'none'}
            showPv={analysisOpen && showPv}
            allWalls={allWalls}
            highlight={highlight}
            onHighlight={setHighlight}
          />
          <div className="control-bar">
            {publicArena && <IconButton icon="home" label="Start screen" onClick={home} />}
            <IconButton icon="restart" label="New game" onClick={newGame} />
            <IconButton icon="undo" label="Undo" onClick={undo} disabled={history.length === 0} />
            {humans < 2 && (
              <IconButton icon={paused ? 'play' : 'pause'} label={paused ? 'Resume' : 'Pause'} disabled={winner !== -1 || !!cutoff}
                onClick={() => { stopAnalysis(); setPaused(!paused) }} />
            )}
            {sides.some(side => side.controller === 'engine') && (
              <IconButton icon="skip" label="Move now" disabled={!thinking || mover.controller !== 'engine'} onClick={() => client.current!.stop()} />
            )}
            <IconButton icon="link" label="Share" className="push" aria-expanded={sharing} aria-pressed={sharing} onClick={() => setSharing(!sharing)} />
          </div>
          {sharing && (
            <SharePanel moves={history.length}
              gameUrl={shareUrl(window.location, sides, bots, history)}
              settingsUrl={shareUrl(window.location, sides, bots)} />
          )}
          {recorded.length > 0 && (
            <div className="control-bar">
              <IconButton icon="back" label="Back" onClick={() => stepReplay(-1)} disabled={history.length === 0} />
              <IconButton icon={autoplay ? 'pause' : 'play'} label={autoplay ? 'Pause replay' : 'Play replay'}
                onClick={() => { if (!onRecord) setHistory([]); setReplaying(!autoplay) }} />
              <IconButton icon="forward" label="Forward" onClick={() => stepReplay(1)} disabled={history.length >= recorded.length} />
              <span className="muted small">Replay · move {history.length} / {recorded.length}</span>
            </div>
          )}

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

        <aside className="sidebar">
          {publicArena && (
            <section className="panel section">
              <h2>Game</h2>
              <div className="section-body">
                {mode === 'play' && (
                  <PlaySetup strength={strengths[1 - you]} color={you ? 'black' : 'white'} bots={bots}
                    onStrength={value => setSide((1 - you) as Player, presetSide(value, bots))}
                    onColor={color => { if ((color === 'black') !== (you === 1)) updateSides(([white, black]) => [black, white]) }} />
                )}
                {mode === 'watch' && <WatchSetup strengths={strengths} bots={bots} onChange={(p, value) => setSide(p, presetSide(value, bots))} />}
                {mode === 'local' && <p className="muted">Two people take turns on this device. Pick a bot for either side under Advanced bot settings.</p>}
              </div>
            </section>
          )}

          <details className="panel section" open={advancedOpen} onToggle={e => setAdvancedOpen(e.currentTarget.open)}>
            <summary><h2>Advanced bot settings</h2></summary>
            <div className="section-body">
              {([0, 1] as const).map((p) => {
                const side = sides[p]
                return (
                  <fieldset key={p} className="side-fields">
                    <legend><span className={`turn-dot pawn${p}`} /> {NAMES[p]}</legend>
                    <div className="field-row">
                      <label>
                        Player
                        <select
                          aria-label={`${NAMES[p]} player`}
                          value={side.controller}
                          onChange={(e) => {
                            const bot = bots.find(item => `browser:${item.id}` === e.target.value)
                            setSide(p, { ...side, controller: e.target.value, simulations: bot?.simulations ?? side.simulations })
                          }}
                        >
                          <option value="human">Human</option>
                          <option value="engine">Built-in negamax</option>
                          {bots.map(bot => <option key={bot.id} value={`browser:${bot.id}`}>{bot.name} · browser neural bot</option>)}
                          {!publicArena && <option value={JSON.stringify(['', 'negamax:2000'])}>Reference negamax (C++, 2000 nodes)</option>}
                          {!publicArena && runs.flatMap(run => run.bots.map(botMode => <option key={`${run.id}/${botMode}`} value={JSON.stringify([run.id, botMode])}>{run.id} / {botMode}</option>))}
                        </select>
                      </label>
                      {side.controller !== 'human' && side.controller !== 'engine' && (
                        <label>
                          Neural simulations
                          <select aria-label={`${NAMES[p]} simulations`} value={side.simulations} onChange={e => setSide(p, { ...side, simulations: Number(e.target.value) })}>
                            {SIMULATIONS.map(n => <option key={n} value={n}>{n}</option>)}
                          </select>
                        </label>
                      )}
                    </div>
                    {side.controller === 'engine' && <EngineOptionsFields options={side.engine} onChange={engine => setSide(p, { ...side, engine })} />}
                  </fieldset>
                )
              })}
            </div>
          </details>

          <details className="panel section" open={analysisOpen} onToggle={e => setAnalysisOpen(e.currentTarget.open)}>
            <summary><h2>Analysis engine</h2>{analyzing && <span className="badge live">running</span>}</summary>
            <div className="section-body">
              <p className="muted small">The built-in negamax engine evaluates the current position. Its settings here don't affect the bots.</p>
              <div className="button-row">
                <button type="button" className="primary" disabled={!analyzing && !canAnalyze} onClick={() => (analyzing ? client.current!.stop() : analyze())}>
                  <Icon name={analyzing ? 'stop' : 'search'} /> {analyzing ? 'Stop analysis' : 'Analyze position'}
                </button>
              </div>
              <EngineOptionsFields options={analysisOptions} onChange={setAnalysisOptions} />
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
              <SearchPanel snapshot={search?.snapshot ?? null} stale={stale} highlight={highlight} onHighlight={setHighlight} maxDepth={search?.maxDepth ?? analysisOptions.maxDepth} />
            </div>
          </details>
        </aside>
      </main>
    </div>
  )
}
