import { describe, expect, it } from 'vitest'
import { Board, cellY, H_BASE, mirrorMove, V_BASE } from './board'
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

  it('reuses allocations with a cold cache without changing the fixed opponent', () => {
    const p = { pawns: [76, 4], wallsLeft: [10, 10], hWalls: [], vWalls: [], turn: 0 } as Position
    const budget = { ...opts, maxNodes: 2000 }
    const reused = new TsSearcher()
    reused.search(p, budget, () => {})
    reused.clearCache()
    const actual = reused.search(p, budget, () => {})
    const expected = new TsSearcher().search(p, budget, () => {})
    expect(actual.bestMove).toBe(expected.bestMove)
    expect(actual.score).toBe(expected.score)
    expect(actual.nodes).toBe(expected.nodes)
  })

  it('deferred search agrees with eager legality around sealing walls and goals', () => {
    const positions: Position[] = [
      { pawns: [72, 4], wallsLeft: [3, 3], hWalls: [56], vWalls: [], turn: 0 },
      { pawns: [72, 67], wallsLeft: [3, 3], hWalls: [56], vWalls: [], turn: 0 },
      { pawns: [47, 31], wallsLeft: [3, 4], hWalls: [35, 9], vWalls: [21, 49], turn: 1 },
    ]
    let speculativeWalls = 0
    const deferred = new TsSearcher((p) => {
      const board = Board.from(p)
      const generate = board.generate.bind(board)
      board.generate = (out, likely, defer) => {
        const n = generate(out, likely, defer)
        if (defer) speculativeWalls += [...out.subarray(0, n)].filter((m) => m >= H_BASE).length
        return n
      }
      return board
    })
    const eager = new TsSearcher((p) => {
      const board = Board.from(p)
      const generate = board.generate.bind(board)
      board.generate = (out, likely) => generate(out, likely, false)
      return board
    })
    for (const position of positions) {
      const settings = { ...opts, maxDepth: 4, exactRoot: true }
      deferred.clearCache()
      eager.clearCache()
      const actual = deferred.search(position, settings, () => {})
      const expected = eager.search(position, settings, () => {})
      expect(actual.iterations.at(-1)?.depth).toBe(4)
      expect(actual.score).toBe(expected.score)
      expect(new Map(actual.root.map((r) => [r.move, r.score])))
        .toEqual(new Map(expected.root.map((r) => [r.move, r.score])))
      const board = Board.from(position)
      for (const move of actual.pv) {
        expect(board.legalMoves()).toContain(move)
        board.make(move)
      }
    }
    expect(speculativeWalls).toBeGreaterThan(0)
  })

  it('shares transpositions across horizontal reflection and remaps legal PV moves', () => {
    const p: Position = { pawns: [47, 31], wallsLeft: [3, 4], hWalls: [35, 9], vWalls: [21, 49], turn: 1 }
    const reflected: Position = { ...p,
      pawns: [mirrorMove(p.pawns[0]), mirrorMove(p.pawns[1])],
      hWalls: p.hWalls.map((i) => mirrorMove(H_BASE + i) - H_BASE),
      vWalls: p.vWalls.map((i) => mirrorMove(V_BASE + i) - V_BASE),
    }
    const searcher = new TsSearcher()
    const first = searcher.search(p, opts, () => {})
    const warm = searcher.search(reflected, opts, () => {})
    const cold = new TsSearcher().search(reflected, opts, () => {})
    expect(warm.score).toBe(first.score)
    expect(warm.score).toBe(cold.score)
    expect(warm.nodes).toBeLessThan(cold.nodes)
    const board = Board.from(reflected)
    for (const move of warm.pv) {
      expect(board.legalMoves()).toContain(move)
      board.make(move)
    }
  })

  it('invalidates cached scores when evaluation or move-generation settings change', () => {
    const p: Position = { pawns: [47, 31], wallsLeft: [3, 4], hWalls: [35, 9], vWalls: [21, 49], turn: 1 }
    const searcher = new TsSearcher()
    searcher.search(p, opts, () => {})
    for (const changed of [{ ...opts, wallWeight: 70 }, { ...opts, tempo: 13 }, { ...opts, moveGen: 'all' as const }]) {
      const actual = searcher.search(p, changed, () => {})
      const expected = new TsSearcher().search(p, changed, () => {})
      expect(actual.score).toBe(expected.score)
      expect(actual.bestMove).toBe(expected.bestMove)
      expect(actual.nodes).toBe(expected.nodes)
    }
  })

  it('restores the board when a search is interrupted', () => {
    let board = Board.initial()
    const position = board.toPosition()
    new TsSearcher((p) => { board = Board.from(p); return board }).search(position,
      { ...opts, maxDepth: 20, maxNodes: 2000 }, () => {})
    expect(board.toPosition()).toEqual(position)
    expect(board.checkHash()).toBe(true)
  })

  it('rejects illegal speculative terminal and immobile positions instead of caching scores', () => {
    const searcher = new TsSearcher()
    searcher.search(Board.initial().toPosition(), { ...opts, maxDepth: 1 }, () => {})
    // Exercise states reachable only inside speculative search, never through the public legal-move API.
    const internal = searcher as unknown as {
      board: Board
      negamax(depth: number, alpha: number, beta: number, ply: number): number | null
    }
    internal.board = Board.from({ pawns: [72, 76], wallsLeft: [0, 0], hWalls: [56], vWalls: [57], turn: 0 })
    expect(internal.board.hasReachedGoal(1)).toBe(true)
    expect(internal.negamax(2, -32000, 32000, 3)).toBeNull()
    internal.board = Board.from({ pawns: [63, 64], wallsLeft: [0, 0], hWalls: [48, 56], vWalls: [49], turn: 0 })
    expect(internal.board.pawnMoves(new Int16Array(8))).toBe(0)
    expect(internal.negamax(2, -32000, 32000, 3)).toBeNull()
  })

})
