import { useMemo, useState } from 'react'
import { moveToString } from '../engine/board'
import type { Move, SearchSnapshot } from '../engine/types'
import { compact, formatBounded, formatScore, scoreToUnit } from '../lib/format'
import { scales, useDark } from '../lib/theme'
import { Icicle, LogBars } from './charts'

const ROWS = 12

interface Props {
  snapshot: SearchSnapshot | null
  stale: boolean
  highlight: Move | null
  onHighlight: (m: Move | null) => void
  maxDepth: number
}

export function SearchPanel({ snapshot, stale, highlight, onHighlight, maxDepth }: Props) {
  const dark = useDark()
  const sc = useMemo(() => scales(dark), [dark])
  const [expanded, setExpanded] = useState(false)

  if (!snapshot) {
    return (
      <section className="panel search-panel empty">
        <h2>Search</h2>
        <p className="muted">Let the engine move or press Analyze to watch it explore the game tree.</p>
      </section>
    )
  }

  const s = snapshot
  const its = s.iterations
  const ebf = its.length >= 2 ? its[its.length - 1].nodes / Math.max(1, its[its.length - 2].nodes) : null
  const searched = s.root.filter((r) => r.depth === s.depth).length
  const totalNodes = s.root.reduce((a, r) => a + r.nodes, 0)
  const rows = expanded ? s.root : s.root.slice(0, ROWS)

  return (
    <section className={`panel search-panel ${stale ? 'stale' : ''}`}>
      <header className="panel-head">
        <h2>Search</h2>
        <span className={`badge ${s.done ? '' : 'live'}`}>{s.done ? 'finished' : 'searching'}</span>
        {stale && <span className="muted small">for the previous position</span>}
        <span className="muted small push">backend: {s.backend}</span>
      </header>

      <div className="tiles">
        <Tile label="Depth" value={`${its.length}${s.done ? '' : ` → ${s.depth}`}`} sub={`max ${maxDepth}`} />
        <Tile label="Eval" value={formatScore(s.score)} sub={`best ${moveToString(s.bestMove)}`} />
        <Tile label="Nodes" value={compact(s.nodes)} sub={`${compact(s.nodes / Math.max(1, s.elapsedMs / 1000))}/s`} />
        <Tile label="Time" value={`${(s.elapsedMs / 1000).toFixed(2)}s`} sub={`${s.root.length} root moves`} />
        <Tile label="TT hits" value={`${s.ttProbes ? Math.round((100 * s.ttHits) / s.ttProbes) : 0}%`} sub={`${compact(s.ttProbes)} probes`} />
        <Tile label="Branching" value={ebf === null ? '–' : ebf.toFixed(1)} sub="effective" />
      </div>

      {!s.done && (
        <div className="progress" title={`depth ${s.depth}: ${searched}/${s.root.length} root moves done`}>
          <div className="progress-bar" style={{ width: `${(100 * Math.max(0, s.currentIndex)) / s.root.length}%` }} />
          <span className="small">
            depth {s.depth}: move {s.currentIndex + 1}/{s.root.length} ({s.currentMove >= 0 ? moveToString(s.currentMove) : '–'})
          </span>
        </div>
      )}

      <div className="pv-line">
        <span className="muted small">Principal variation</span>
        <div className="pv-moves">
          {s.pv.map((m, i) => (
            <span key={i} className={`pv-chip pv${i % 2}`}>
              {moveToString(m)}
            </span>
          ))}
        </div>
      </div>

      <h3>Search tree</h3>
      <Icicle root={s.root} diverging={sc.diverging} highlight={highlight} current={s.done ? -1 : s.currentMove} onHighlight={onHighlight} />

      <div className="chart-row">
        <div>
          <h3>Nodes per iteration</h3>
          <LogBars
            caption="Nodes searched per iterative deepening depth"
            bars={[
              ...its.map((it) => ({
                key: it.depth,
                label: String(it.depth),
                value: it.nodes,
                tooltip: (
                  <>
                    <b>depth {it.depth}</b>
                    <div>{compact(it.nodes)} nodes · {it.timeMs.toFixed(0)} ms</div>
                    <div>best {moveToString(it.bestMove)} ({formatScore(it.score)})</div>
                  </>
                ),
              })),
              ...(s.depth <= its.length
                ? []
                : [
                    {
                      key: 'cur',
                      label: `${s.depth}…`,
                      value: s.nodes - its.reduce((a, it) => a + it.nodes, 0),
                      emphasis: true,
                      tooltip: (
                        <>
                          <b>depth {s.depth} ({s.done ? 'stopped by time limit' : 'in progress'})</b>
                          <div>{compact(s.nodes - its.reduce((a, it) => a + it.nodes, 0))} nodes so far</div>
                        </>
                      ),
                    },
                  ]),
            ]}
          />
        </div>
        <div>
          <h3>Nodes per ply</h3>
          <LogBars
            caption="Nodes visited at each ply of the tree"
            bars={s.plyNodes.slice(1).map((n, i) => ({
              key: i + 1,
              label: String(i + 1),
              value: n,
              tooltip: (
                <>
                  <b>ply {i + 1}</b>
                  <div>{compact(n)} nodes</div>
                  <div>
                    {compact(s.plyCutoffs[i + 1] ?? 0)} beta cutoffs ({n ? Math.round((100 * (s.plyCutoffs[i + 1] ?? 0)) / n) : 0}%)
                  </div>
                </>
              ),
            }))}
          />
        </div>
      </div>

      <h3>Candidate moves</h3>
      <table className="root-table">
        <thead>
          <tr>
            <th>#</th>
            <th>move</th>
            <th className="num">eval</th>
            <th className="bar-col">score</th>
            <th className="num">nodes</th>
            <th className="num">d</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const t = scoreToUnit(r.score)
            return (
              <tr
                key={r.move}
                className={`${r.move === highlight ? 'hi' : ''} ${!s.done && r.move === s.currentMove ? 'current' : ''}`}
                onPointerEnter={() => onHighlight(r.move)}
                onPointerLeave={() => onHighlight(null)}
              >
                <td className="muted">{i + 1}</td>
                <td>
                  <b>{moveToString(r.move)}</b>
                </td>
                <td className="num">{r.depth ? formatBounded(r.score, r.bound) : '–'}</td>
                <td className="bar-col">
                  <div className="div-bar">
                    <span className="mid" />
                    {r.depth > 0 && (
                      <span
                        className={`fill ${r.bound === 'exact' ? '' : 'bounded'}`}
                        style={{
                          left: t >= 0 ? '50%' : `${50 + 50 * t}%`,
                          width: `${50 * Math.abs(t)}%`,
                          background: sc.diverging(t >= 0 ? 1 : -1),
                        }}
                      />
                    )}
                  </div>
                </td>
                <td className="num">{totalNodes ? `${((100 * r.nodes) / totalNodes).toFixed(r.nodes / totalNodes < 0.1 ? 1 : 0)}%` : '–'}</td>
                <td className="num muted">{r.depth || '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {s.root.length > ROWS && (
        <button className="link" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show fewer' : `Show all ${s.root.length} moves`}
        </button>
      )}
      <p className="muted small">
        Eval is from the mover's view in path steps: (opponent distance − own distance) + 0.2 × wall difference. ≤ marks scores that
        alpha-beta only proved to be an upper bound; #n is a forced win in n moves.
      </p>
    </section>
  )
}

function Tile({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      <div className="tile-sub">{sub}</div>
    </div>
  )
}
