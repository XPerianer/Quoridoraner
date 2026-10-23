import { hierarchy, partition } from 'd3-hierarchy'
import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { moveToString } from '../engine/board'
import type { Bound, Move, RootMoveStat } from '../engine/types'
import { compact, formatBounded, scoreToUnit } from '../lib/format'

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(320)
  useLayoutEffect(() => {
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(120, e.contentRect.width)))
    ro.observe(ref.current!)
    return () => ro.disconnect()
  }, [])
  return [ref, width]
}

interface Tip {
  x: number
  y: number
  content: ReactNode
}

function Tooltip({ tip, width }: { tip: Tip | null; width: number }) {
  if (!tip) return null
  const left = Math.min(tip.x + 12, width - 180)
  return (
    <div className="tooltip" style={{ left: Math.max(0, left), top: tip.y + 12 }}>
      {tip.content}
    </div>
  )
}

export interface BarDatum {
  key: string | number
  label: string
  value: number
  tooltip: ReactNode
  emphasis?: boolean
}

export function LogBars({ bars, height = 120, caption }: { bars: BarDatum[]; height?: number; caption: string }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const left = 36
  const bottom = 18
  const top = 14
  const plotH = height - bottom - top
  const max = Math.max(10, ...bars.map((b) => b.value))
  const decades = Math.ceil(Math.log10(max))
  const y = (v: number) => top + plotH - (v <= 0 ? 0 : (Math.log10(Math.max(1, v)) / decades) * plotH)
  const slot = (width - left) / Math.max(1, bars.length)
  const barW = Math.max(3, Math.min(28, slot - 2))
  const last = bars.length - 1
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={caption}>
        {Array.from({ length: decades + 1 }, (_, d) => (
          <g key={d} className="grid">
            <line x1={left} x2={width} y1={y(10 ** d)} y2={y(10 ** d)} />
            <text x={left - 6} y={y(10 ** d) + 3} textAnchor="end">
              {decadeLabel(d)}
            </text>
          </g>
        ))}
        <line className="axis" x1={left} x2={width} y1={top + plotH} y2={top + plotH} />
        {bars.map((b, i) => {
          const x = left + i * slot + (slot - barW) / 2
          const h = top + plotH - y(b.value)
          return (
            <g
              key={b.key}
              onPointerMove={(e) => {
                const r = ref.current!.getBoundingClientRect()
                setTip({ x: e.clientX - r.left, y: e.clientY - r.top, content: b.tooltip })
              }}
              onPointerLeave={() => setTip(null)}
            >
              <rect className="hit" x={left + i * slot} y={top} width={slot} height={plotH + bottom} />
              {h > 0 && <path className={`bar ${b.emphasis ? 'emphasis' : ''}`} d={roundedTop(x, y(b.value), barW, h, Math.min(4, barW / 2))} />}
              {(slot > 16 || i % Math.ceil(16 / slot) === 0) && (
                <text className="tick" x={left + i * slot + slot / 2} y={height - 4} textAnchor="middle">
                  {b.label}
                </text>
              )}
              {i === last && h > 0 && (
                <text className="value" x={x + barW / 2} y={y(b.value) - 4} textAnchor="middle">
                  {compact(b.value)}
                </text>
              )}
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  )
}

function decadeLabel(d: number): string {
  return ['1', '10', '100', '1k', '10k', '100k', '1M', '10M', '100M', '1G'][d] ?? `1e${d}`
}

function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.min(r, h)
  return `M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z`
}

interface TreeDatum {
  kind: 'root' | 'move' | 'reply'
  move: Move
  parent: Move
  score: number
  bound: Bound
  nodes: number
  self: number
  replyCount?: number
  repliesSearched?: number
  depth?: number
  children?: TreeDatum[]
}

export function Icicle({
  root,
  diverging,
  highlight,
  current,
  onHighlight,
}: {
  root: RootMoveStat[]
  diverging: (t: number) => string
  highlight: Move | null
  current: Move
  onHighlight: (m: Move | null) => void
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip | null>(null)
  const band = 46
  const height = 2 * band + 2

  const nodes = useMemo(() => {
    const data: TreeDatum = {
      kind: 'root',
      move: -1,
      parent: -1,
      score: 0,
      bound: 'exact',
      nodes: 0,
      self: 0,
      children: root
        .filter((r) => r.depth > 0)
        .map((r) => ({
          kind: 'move',
          move: r.move,
          parent: -1,
          score: r.score,
          bound: r.bound,
          nodes: r.nodes,
          self: Math.max(1, r.nodes - r.replies.reduce((a, c) => a + c.nodes, 0)),
          replyCount: r.replyCount,
          repliesSearched: r.repliesSearched,
          depth: r.depth,
          children: r.replies.map((c) => ({ kind: 'reply', move: c.move, parent: r.move, score: c.score, bound: c.bound, nodes: c.nodes, self: c.nodes })),
        })),
    }
    const h = hierarchy(data)
      .sum((d) => d.self)
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
    return partition<TreeDatum>().size([width, 3 * band])(h).descendants().filter((d) => d.depth > 0)
  }, [root, width])

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Search tree: root moves and replies, width proportional to nodes searched">
        {nodes.map((d, i) => {
          const w = d.x1 - d.x0 - 2
          if (w < 0.5) return null
          const datum = d.data
          const owner = datum.kind === 'move' ? datum.move : datum.parent
          const yy = (d.depth - 1) * band
          const isHi = highlight !== null && owner === highlight
          return (
            <g
              key={i}
              onPointerMove={(e) => {
                const r = ref.current!.getBoundingClientRect()
                setTip({ x: e.clientX - r.left, y: e.clientY - r.top, content: treeTip(datum) })
                onHighlight(owner)
              }}
              onPointerLeave={() => {
                setTip(null)
                onHighlight(null)
              }}
            >
              <rect
                x={d.x0 + 1}
                y={yy + 1}
                width={w}
                height={band - 2}
                rx={Math.min(4, w / 2)}
                fill={diverging(scoreToUnit(datum.score))}
                opacity={datum.bound === 'exact' ? 1 : 0.55}
                className={`seg ${isHi ? 'hi' : ''} ${datum.kind === 'move' && datum.move === current ? 'current' : ''}`}
              />
              {w > 38 && (
                <text className="seg-label" x={d.x0 + 1 + w / 2} y={yy + band / 2 + 4} textAnchor="middle">
                  {moveToString(datum.move)}
                </text>
              )}
            </g>
          )
        })}
      </svg>
      <div className="icicle-legend muted">
        <span>top: candidate moves</span>
        <span>bottom: opponent replies</span>
        <span>width = nodes searched</span>
      </div>
      <Tooltip tip={tip} width={width} />
    </div>
  )
}

function treeTip(d: TreeDatum): ReactNode {
  return (
    <>
      <div>
        <b>{d.kind === 'reply' ? `${moveToString(d.parent)} → ${moveToString(d.move)}` : moveToString(d.move)}</b>
      </div>
      <div>score {formatBounded(d.score, d.bound)} {d.bound !== 'exact' && <span className="muted">(bound from pruning)</span>}</div>
      <div>{compact(d.nodes)} nodes</div>
      {d.kind === 'move' && (
        <div>
          {d.repliesSearched}/{d.replyCount} replies searched{d.repliesSearched! < d.replyCount! ? ' (cutoff)' : ''}
        </div>
      )}
    </>
  )
}
