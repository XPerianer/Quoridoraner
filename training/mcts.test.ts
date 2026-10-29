import { describe, expect, it } from 'vitest'
import { Board } from '../src/engine/board'
import { MctsPool, positionKey } from './mcts'
import type { SearchRequest } from './mcts'

function run(pool: MctsPool, requests: SearchRequest[]) {
  let result = pool.begin(requests)
  let maxBatch = 0
  while (!result.done) {
    maxBatch = Math.max(maxBatch, result.leaves.length)
    result = pool.step(result.leaves.map(leaf => ({
      id: leaf.id, value: 0, policy: leaf.legal.map(() => 1 / leaf.legal.length),
    })))
  }
  return { ...result, maxBatch }
}
function request(id: string, position = Board.initial().toPosition()): SearchRequest {
  return { id, position, history: { [positionKey(position)]: 1 }, ply: 0, maxPlies: 100, simulations: 32 }
}

describe('batched MCTS', () => {
  it('batches independent games and preserves the single-game policy', () => {
    const a = request('a')
    const b = request('b', { pawns: [40, 63], wallsLeft: [0, 0], hWalls: [], vWalls: [], turn: 1 })
    const batched = run(new MctsPool(), [a, b])
    expect(batched.maxBatch).toBe(2)
    expect(batched.policies.a).toEqual(run(new MctsPool(), [a]).policies.a)
    expect(batched.policies.b).toEqual(run(new MctsPool(), [b]).policies.b)
    expect(batched.policies.b!.indexOf(Math.max(...batched.policies.b!))).toBe(72)
  })

  it('retains a subtree and restores it exactly from a JSON checkpoint', () => {
    const pool = new MctsPool()
    const a = request('a')
    const first = run(pool, [a]).policies.a!
    const move = first.indexOf(Math.max(...first))
    const position = pool.advance('a', a.position, move)
    const next = { ...a, position, ply: 1, history: { ...a.history, [positionKey(position)]: 1 } }
    const snapshot = JSON.parse(JSON.stringify(pool.snapshot())) as ReturnType<MctsPool['snapshot']>
    const restored = new MctsPool()
    restored.restore(snapshot)
    const continued = run(pool, [next])
    expect(continued.reused).toBeGreaterThan(0)
    expect(run(restored, [next])).toEqual(continued)
    expect(continued.policies.a!.reduce((x, y) => x + y, 0)).toBeCloseTo(1)
    const legal = Board.from(position).legalMoves()
    expect(continued.policies.a!.every((p, i) => p === 0 || legal.includes(i))).toBe(true)
  })

  it('backs up draws without neural evaluation at the horizon or third repetition', () => {
    const a = request('a', { pawns: [76, 4], wallsLeft: [0, 0], hWalls: [], vWalls: [], turn: 0 })
    for (const move of Board.from(a.position).legalMoves()) {
      const board = Board.from(a.position)
      board.make(move)
      a.history[positionKey(board.toPosition())] = 2
    }
    const originalHistory = { ...a.history }
    const pool = new MctsPool()
    const initial = pool.begin([a])
    expect(initial.leaves).toHaveLength(1)
    const result = pool.step([{ id: 'a', value: 0.9, policy: initial.leaves[0]!.legal.map(() => 1) }])
    expect(result.done).toBe(true)
    expect(result.leaves).toHaveLength(0)
    const tree = pool.snapshot()[0]![1]
    expect(tree.total).toBe(0)
    expect(tree.visits).toBe(32)
    expect(a.history).toEqual(originalHistory)
  })

  it('does not bake root exploration noise into a retained network policy', () => {
    const a = request('a')
    a.noise = Board.initial().legalMoves().map((_, i) => i === 0 ? 1 : 0)
    const pool = new MctsPool()
    run(pool, [a])
    const priors = pool.snapshot()[0]![1].priors!
    expect(priors.every(([, p]) => Math.abs(p - 1 / priors.length) < 1e-10)).toBe(true)
    pool.clear()
    expect(pool.snapshot()).toEqual([])
  })
})
