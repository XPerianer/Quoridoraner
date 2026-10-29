import { existsSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { Board } from '../src/engine/board'
import { DEFAULT_OPTIONS, TsSearcher } from '../src/engine/search'
import type { Position } from '../src/engine/types'
import { corpus } from '../bench/positions'
import { loadNative, nativeRules } from './native'
import { MctsPool, positionKey } from './mcts'

const available = existsSync(new URL('./native/build/engine.node', import.meta.url))
describe.skipIf(!available)('native engine differential checks', () => {
  const positions = corpus()
  it('matches move order, distances, hashes, and transitions on reachable positions', () => {
    const native = loadNative()
    for (const position of positions) {
      const board = Board.from(position)
      const actual = native.inspect(position)
      expect(actual.legal).toEqual(board.legalMoves())
      expect(actual.distances).toEqual([board.distance(0), board.distance(1)])
      expect(actual.hash1).toBe(board.hash1 >>> 0)
      expect(actual.hash2).toBe(board.hash2 >>> 0)
      for (const move of board.legalMoves().filter((_, i) => i % 7 === 0)) {
        board.make(move)
        expect(native.step(position, move)).toEqual(board.toPosition())
        board.unmake(move)
      }
    }
  })

  it('matches goals, straight/diagonal jumps, and rejects illegal moves', () => {
    const native = loadNative()
    for (const position of [
      { pawns: [49,40], hWalls: [], vWalls: [], wallsLeft: [10,10], turn: 0 },
      { pawns: [49,40], hWalls: [28], vWalls: [], wallsLeft: [10,10], turn: 0 },
      { pawns: [2,76], hWalls: [], vWalls: [], wallsLeft: [0,0], turn: 1 },
      { pawns: [72,4], hWalls: [56], vWalls: [], wallsLeft: [10,10], turn: 1 },
    ] as Position[]) {
      const board = Board.from(position)
      expect(native.inspect(position).winner).toBe(board.winner())
      expect(native.legal(position)).toEqual(board.winner() < 0 ? board.legalMoves() : [])
    }
    expect(() => native.step(Board.initial().toPosition(),0)).toThrow('Illegal')
    expect(() => native.legal({ ...Board.initial().toPosition(), pawns: [-1,4] })).toThrow()
    expect(() => native.legal({ ...Board.initial().toPosition(), hWalls: [3,4] })).toThrow()
  })

  it('matches fixed-depth negamax scores and returns optimal legal moves', () => {
    const native = loadNative()
    const engine = new TsSearcher()
    for (const position of positions.filter((_, i) => i % 5 === 0)) {
      for (const maxDepth of [2, 3]) {
        engine.clearCache()
        const expected = engine.search(position, { ...DEFAULT_OPTIONS, exactRoot: true, maxNodes: 1e9, maxDepth, timeMs: 1e9, reportEveryMs: 1e9 }, () => {})
        const actual = native.search(position, 1e9, maxDepth)
        expect(actual.score, JSON.stringify({ position, maxDepth })).toBe(expected.score)
        expect(expected.root.find(r => r.move === actual.move)?.score).toBe(expected.score)
        expect(actual.nodes).toBeGreaterThan(0)
      }
      const limited = native.search(position, 2000)
      expect(Board.from(position).legalMoves()).toContain(limited.move)
      expect(Number.isFinite(limited.score)).toBe(true)
    }
  }, 30000)

  it('preserves batched MCTS policies and accepts shared tree checkpoints', () => {
    const run = (pool: MctsPool, position: Position) => {
      let result = pool.begin([{ id: 'game', position, history: { [positionKey(position)]: 1 }, ply: 0, maxPlies: 100, simulations: 64 }])
      while (!result.done) result = pool.step(result.leaves.map(leaf => ({
        id: leaf.id, value: 0.2, policy: leaf.legal.map((_, i) => 1/(i+1)),
      })))
      return result
    }
    for (const position of positions.filter((_, i) => i % 7 === 0)) {
      const ts = new MctsPool(), cpp = new MctsPool(nativeRules(loadNative()))
      expect(run(cpp,position)).toEqual(run(ts,position))
      cpp.restore(JSON.parse(JSON.stringify(ts.snapshot())) as ReturnType<MctsPool['snapshot']>)
      expect(run(cpp,position)).toEqual(run(ts,position))
    }
  })
})
