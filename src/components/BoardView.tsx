import { useMemo, useRef, useState, type PointerEvent } from 'react'
import {
  Board,
  cellX,
  cellY,
  H_BASE,
  isPawnMove,
  isVerticalWall,
  moveToString,
  SIZE,
  V_BASE,
  WALL_SIZE,
  WALL_SLOTS,
  wallIndex,
  wallMove,
} from '../engine/board'
import type { Move, RootMoveStat, SearchSnapshot } from '../engine/types'
import { compact, formatBounded, scoreToUnit } from '../lib/format'
import { scales, useDark } from '../lib/theme'

export type Overlay = 'candidates' | 'visits' | 'none'

const S = 48
const G = 12
const P = S + G
const BOARD = G + SIZE * P
const HIT = G / 2 + 9
const TOP_WALLS = 10

const cellRect = (c: number) => ({ x: G + cellX(c) * P, y: G + cellY(c) * P })

function wallRect(m: Move, inset = 0) {
  const i = wallIndex(m)
  const x = i % WALL_SIZE
  const y = (i / WALL_SIZE) | 0
  return isVerticalWall(m)
    ? { x: G + x * P + S + inset, y: G + y * P + inset, width: G - 2 * inset, height: 2 * S + G - 2 * inset }
    : { x: G + x * P + inset, y: G + y * P + S + inset, width: 2 * S + G - 2 * inset, height: G - 2 * inset }
}

function centerOf(m: Move) {
  if (isPawnMove(m)) {
    const r = cellRect(m)
    return { cx: r.x + S / 2, cy: r.y + S / 2 }
  }
  const r = wallRect(m)
  return { cx: r.x + r.width / 2, cy: r.y + r.height / 2 }
}

const clampWall = (v: number) => Math.max(0, Math.min(WALL_SIZE - 1, v))

function targetAt(px: number, py: number): Move | null {
  const ux = px - G
  const uy = py - G
  if (ux < -G / 2 || uy < -G / 2 || ux > SIZE * P || uy > SIZE * P) return null
  const gy = Math.round((uy - S - G / 2) / P)
  const gx = Math.round((ux - S - G / 2) / P)
  const dy = gy >= 0 && gy < WALL_SIZE ? Math.abs(uy - (gy * P + S + G / 2)) : Infinity
  const dx = gx >= 0 && gx < WALL_SIZE ? Math.abs(ux - (gx * P + S + G / 2)) : Infinity
  if (dy < HIT && dy <= dx) return wallMove(false, clampWall(gx), gy)
  if (dx < HIT) return wallMove(true, gx, clampWall(gy))
  const cx = Math.floor(ux / P)
  const cy = Math.floor(uy / P)
  if (cx < 0 || cy < 0 || cx >= SIZE || cy >= SIZE) return null
  return cy * SIZE + cx
}

interface Props {
  board: Board
  interactive: boolean
  onMove: (m: Move) => void
  lastMove: Move | null
  snapshot: SearchSnapshot | null
  overlay: Overlay
  showPv: boolean
  allWalls: boolean
  highlight: Move | null
  onHighlight: (m: Move | null) => void
}

export function BoardView(props: Props) {
  const { board, interactive, onMove, lastMove, snapshot, overlay, showPv, allWalls, highlight, onHighlight } = props
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<Move | null>(null)
  const dark = useDark()
  const sc = useMemo(() => scales(dark), [dark])

  const legal = useMemo(() => new Set(board.legalMoves()), [board])
  const pawnTargets = useMemo(() => [...legal].filter(isPawnMove), [legal])
  const rootByMove = useMemo(() => new Map(snapshot?.root.map((r) => [r.move, r]) ?? []), [snapshot])

  const pointer = (e: PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current!
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM()!.inverse())
    const t = targetAt(pt.x, pt.y)
    setHover(t)
    onHighlight(t !== null && rootByMove.has(t) ? t : null)
    return t
  }

  const hoverStat = hover !== null ? rootByMove.get(hover) : undefined
  const hoverLegal = hover !== null && legal.has(hover)

  return (
    <div className="board-wrap">
      <svg
        ref={svgRef}
        className={`board ${interactive ? 'interactive' : ''}`}
        viewBox={`-22 -4 ${BOARD + 26} ${BOARD + 26}`}
        onPointerMove={pointer}
        onPointerLeave={() => {
          setHover(null)
          onHighlight(null)
        }}
        onPointerDown={(e) => {
          const t = pointer(e)
          if (interactive && t !== null && legal.has(t)) onMove(t)
        }}
        role="img"
        aria-label="Quoridor board"
      >
        <rect className="board-bg" x={0} y={0} width={BOARD} height={BOARD} rx={10} />
        {Array.from({ length: SIZE }, (_, i) => (
          <g key={i} className="coord">
            <text x={-11} y={G + i * P + S / 2 + 4} textAnchor="middle">
              {SIZE - i}
            </text>
            <text x={G + i * P + S / 2} y={BOARD + 16} textAnchor="middle">
              {String.fromCharCode(97 + i)}
            </text>
          </g>
        ))}
        {Array.from({ length: SIZE * SIZE }, (_, c) => (
          <rect key={c} className={`cell ${cellY(c) === 0 ? 'goal0' : cellY(c) === SIZE - 1 ? 'goal1' : ''}`} {...cellRect(c)} width={S} height={S} rx={6} />
        ))}

        {overlay === 'visits' && snapshot && <VisitLayer snapshot={snapshot} seq={sc.sequential} />}
        {overlay === 'candidates' && snapshot && (
          <CandidateLayer root={snapshot.root} diverging={sc.diverging} allWalls={allWalls} highlight={highlight} board={board} />
        )}

        {lastMove !== null && isPawnMove(lastMove) && <rect className="last-move" {...cellRect(lastMove)} width={S} height={S} rx={6} />}

        {[...board.hWall.keys()].filter((i) => board.hWall[i]).map((i) => (
          <rect key={`h${i}`} className={`wall ${lastMove === H_BASE + i ? 'last' : ''}`} {...wallRect(H_BASE + i, 1)} rx={3} />
        ))}
        {[...board.vWall.keys()].filter((i) => board.vWall[i]).map((i) => (
          <rect key={`v${i}`} className={`wall ${lastMove === V_BASE + i ? 'last' : ''}`} {...wallRect(V_BASE + i, 1)} rx={3} />
        ))}

        {interactive &&
          pawnTargets.map((c) => {
            const { cx, cy } = centerOf(c)
            return <circle key={c} className="target-dot" cx={cx} cy={cy} r={6} />
          })}

        {snapshot && !snapshot.done && snapshot.currentMove >= 0 && <SearchingMarker move={snapshot.currentMove} />}

        {([0, 1] as const).map((p) => {
          const { cx, cy } = centerOf(board.pawn[p])
          return (
            <g key={p} className="pawn-g" style={{ transform: `translate(${cx}px, ${cy}px)` }}>
              <circle className={`pawn pawn${p} ${board.turn === p ? 'to-move' : ''}`} r={17} />
            </g>
          )
        })}

        {showPv && snapshot && <PvLayer pv={snapshot.pv} firstMover={board.turn} />}

        {interactive && hover !== null && !isPawnMove(hover) && (
          <rect className={`wall-preview ${hoverLegal ? 'ok' : 'bad'}`} {...wallRect(hover, 1)} rx={3} />
        )}
      </svg>
      <div className="board-caption">
        {hover === null ? (
          <span className="muted">{interactive ? 'Click a square to move, or a gap between squares to place a wall.' : ' '}</span>
        ) : (
          <>
            <b>{moveToString(hover)}</b>
            {!hoverLegal && interactive && <span className="muted"> · not legal</span>}
            {hoverStat && (
              <span>
                {' '}· engine {formatBounded(hoverStat.score, hoverStat.bound)} at depth {hoverStat.depth} · {compact(hoverStat.nodes)} nodes
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function CandidateLayer({
  root,
  diverging,
  allWalls,
  highlight,
  board,
}: {
  root: RootMoveStat[]
  diverging: (t: number) => string
  allWalls: boolean
  highlight: Move | null
  board: Board
}) {
  const walls = root.filter((r) => !isPawnMove(r.move))
  const shown = allWalls ? walls : walls.slice(0, TOP_WALLS)
  return (
    <g className="candidates">
      {root
        .filter((r) => isPawnMove(r.move) && r.depth > 0)
        .map((r) => {
          const rect = cellRect(r.move)
          return (
            <g key={r.move}>
              <rect {...rect} width={S} height={S} rx={6} fill={diverging(scoreToUnit(r.score))} opacity={r.bound === 'exact' ? 1 : 0.55} />
              {r.move !== board.pawn[0] && r.move !== board.pawn[1] && (
                <text className="cand-label" x={rect.x + S / 2} y={rect.y + S - 6} textAnchor="middle">
                  {formatBounded(r.score, r.bound)}
                </text>
              )}
            </g>
          )
        })}
      {shown
        .filter((r) => r.depth > 0)
        .map((r) => (
          <rect
            key={r.move}
            className="cand-wall"
            {...wallRect(r.move, 2.5)}
            rx={2}
            fill={diverging(scoreToUnit(r.score))}
            opacity={r.bound === 'exact' ? 0.95 : 0.6}
          />
        ))}
      {highlight !== null && (
        <rect className="highlight" {...(isPawnMove(highlight) ? { ...cellRect(highlight), width: S, height: S } : wallRect(highlight, -1))} rx={5} />
      )}
    </g>
  )
}

function VisitLayer({ snapshot, seq }: { snapshot: SearchSnapshot; seq: (t: number) => string }) {
  const cells = snapshot.pawnHeat[0].map((v, i) => v + snapshot.pawnHeat[1][i])
  const maxCell = Math.max(1, ...cells)
  const walls = [...snapshot.wallHeat[0], ...snapshot.wallHeat[1]]
  const maxWall = Math.max(1, ...walls)
  const t = (v: number, max: number) => Math.log1p(v) / Math.log1p(max)
  return (
    <g className="visits">
      {cells.map((v, c) =>
        v > 0 ? <rect key={c} {...cellRect(c)} width={S} height={S} rx={6} fill={seq(t(v, maxCell))} /> : null,
      )}
      {walls.map((v, i) =>
        v > 0 ? (
          <rect
            key={i}
            {...wallRect((i < WALL_SLOTS ? H_BASE : V_BASE - WALL_SLOTS) + i, 4)}
            rx={2}
            fill={seq(t(v, maxWall))}
            opacity={0.1 + 0.6 * t(v, maxWall) ** 3}
          />
        ) : null,
      )}
    </g>
  )
}

function PvLayer({ pv, firstMover }: { pv: Move[]; firstMover: number }) {
  return (
    <g className="pv">
      {pv.map((m, i) => {
        const mover = (firstMover + i) % 2
        const { cx, cy } = centerOf(m)
        return (
          <g key={i}>
            {!isPawnMove(m) && <rect className={`pv-wall pv${mover}`} {...wallRect(m, 1)} rx={3} />}
            <circle className={`pv-dot pv${mover}`} cx={cx} cy={cy} r={8} />
            <text className={`pv-num pv${mover}`} x={cx} y={cy + 3.5} textAnchor="middle">
              {i + 1}
            </text>
          </g>
        )
      })}
    </g>
  )
}

function SearchingMarker({ move }: { move: Move }) {
  return isPawnMove(move) ? (
    <rect className="searching" {...cellRect(move)} width={S} height={S} rx={6} />
  ) : (
    <rect className="searching" {...wallRect(move, -2)} rx={4} />
  )
}
