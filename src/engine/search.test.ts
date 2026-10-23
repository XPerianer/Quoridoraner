import { describe, expect, it } from 'vitest'
import { cellY } from './board'
import { DEFAULT_OPTIONS, TsSearcher, WIN } from './search'
import type { Position, SearchOptions } from './types'

const opts: SearchOptions = { ...DEFAULT_OPTIONS, maxDepth: 3, timeMs: 10_000, moveGen: 'likely', reportEveryMs: 50 }

describe('TsSearcher', () => {
  it('takes an immediate win', () => {
    const p: Position = { pawns: [1 * 9 + 2, 7 * 9 + 6], wallsLeft: [0, 10], hWalls: [], vWalls: [], turn: 0 }
    const s = new TsSearcher().search(p, opts, () => {})
    expect(cellY(s.bestMove)).toBe(0)
    expect(s.score).toBe(WIN - 1)
  })

  it('blocks an opponent one step from winning', () => {
    const p: Position = { pawns: [8 * 9 + 4, 7 * 9 + 4], wallsLeft: [10, 10], hWalls: [], vWalls: [], turn: 0 }
    const s = new TsSearcher().search(p, { ...opts, maxDepth: 2 }, () => {})
    expect(s.bestMove).toBeGreaterThanOrEqual(81)
  })

  it('exact root gives every root move an exact score', () => {
    const p: Position = { pawns: [76, 4], wallsLeft: [10, 10], hWalls: [], vWalls: [], turn: 0 }
    const s = new TsSearcher().search(p, { ...opts, maxDepth: 2, exactRoot: true }, () => {})
    expect(s.root.every((r) => r.bound === 'exact' && r.depth === 2)).toBe(true)
    expect(s.root[0].score).toBe(s.score)
    expect(s.iterations.map((i) => i.depth)).toEqual([1, 2])
  })

  it('pruned root agrees with exact root on the best score', () => {
    const p: Position = { pawns: [5 * 9 + 3, 3 * 9 + 5], wallsLeft: [6, 7], hWalls: [4 * 8 + 3], vWalls: [2 * 8 + 5], turn: 1 }
    const exact = new TsSearcher().search(p, { ...opts, exactRoot: true }, () => {})
    const pruned = new TsSearcher().search(p, opts, () => {})
    expect(pruned.score).toBe(exact.score)
  })

  it('respects the node budget deterministically', () => {
    const p: Position = { pawns: [76, 4], wallsLeft: [10, 10], hWalls: [], vWalls: [], turn: 0 }
    const budget = { ...opts, maxDepth: 20, timeMs: 60_000, maxNodes: 20_000, moveGen: 'all' as const }
    const a = new TsSearcher().search(p, budget, () => {})
    const b = new TsSearcher().search(p, budget, () => {})
    expect(a.nodes).toBeLessThan(20_000 + 1024)
    expect(b.bestMove).toBe(a.bestMove)
    expect(b.nodes).toBe(a.nodes)
  })
})
