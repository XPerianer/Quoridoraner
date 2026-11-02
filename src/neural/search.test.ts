import { describe, expect, it, vi } from 'vitest'
import { Board } from '../engine/board'
import type { Position } from '../engine/types'
import { canonicalMove } from './features'
import { positionKey, searchNeural } from './search'

const start = () => Board.initial().toPosition()
const uniform = async () => ({ logits: new Float32Array(209), value: 0 })
describe('browser PUCT', () => {
  it('masks illegal logits and converts Black canonical actions', async () => {
    const board = Board.initial(); board.make(67)
    const position = board.toPosition()
    const result = await searchNeural({ positions: [start(), position], simulations: 8, evaluate: async () => {
      const logits = new Float32Array(209).fill(-20)
      logits[canonicalMove(13, 1)] = 20
      logits[canonicalMove(80, 1)] = 100 // Illegal square must never receive probability.
      return { logits, value: 0 }
    } })
    expect(result).toEqual({ move: 13, simulations: 8 })
  })
  it('backs up terminal wins with the correct value sign for both players', async () => {
    for (const turn of [0, 1] as const) {
      const p: Position = { pawns: turn === 0 ? [13, 40] : [40, 67], wallsLeft: [0, 0], hWalls: [], vWalls: [], turn }
      const target = turn === 0 ? 4 : 76
      const evaluate = vi.fn(async () => {
        const logits = new Float32Array(209).fill(-1)
        logits[canonicalMove(target, turn)] = 2
        return { logits, value: -0.5 }
      })
      const result = await searchNeural({ positions: [p], simulations: 24, evaluate })
      expect(result.move).toBe(target)
      expect(evaluate.mock.calls.length).toBeLessThan(24)
    }
  })
  it('uses per-branch repetition counts and draws before evaluating a third occurrence', async () => {
    const initial = start(), next = Board.initial(); next.make(67)
    const nextPosition = next.toPosition()
    const evaluate = vi.fn(async () => {
      const logits = new Float32Array(209).fill(-100)
      logits[67] = 100
      return { logits, value: .5 }
    })
    const result = await searchNeural({ positions: [nextPosition, nextPosition, initial], simulations: 2, evaluate })
    expect(result.move).toBe(67)
    expect(evaluate).toHaveBeenCalledTimes(1)
  })
  it('uses the training horizon then the public 200-ply horizon', async () => {
    for (const ply of [99, 100, 199]) {
      const p = start()
      // Unique preceding positions keep this fixture independent of repetition.
      const positions = Array.from({ length: ply }, (_, i) => ({ ...p, hWalls: [i + 500] }))
      positions.push(p)
      const evaluate = vi.fn(uniform)
      await searchNeural({ positions, simulations: 1, evaluate })
      const horizon = ply < 100 ? 100 : 200
      expect(evaluate.mock.calls[0]).toEqual([p, 1, (horizon - ply) / horizon])
      expect(evaluate.mock.calls.length).toBe(ply === 100 ? 2 : 1)
    }
  })
  it('rejects finished games and invalid model outputs', async () => {
    const p = start()
    await expect(searchNeural({ positions: [p, p, p], simulations: 1, evaluate: uniform })).rejects.toThrow('finished')
    await expect(searchNeural({ positions: [p], simulations: 0, evaluate: uniform })).rejects.toThrow('budget')
    await expect(searchNeural({ positions: [p], simulations: 1, evaluate: async () => ({ logits: new Float32Array(209).fill(NaN), value: 0 }) })).rejects.toThrow('invalid policy')
  })
  it('normalizes repetition keys and reports completed work', async () => {
    expect(positionKey({ ...start(), hWalls: [8, 1] })).toBe(positionKey({ ...start(), hWalls: [1, 8] }))
    const progress: number[] = []
    const result = await searchNeural({ positions: [start()], simulations: 3, evaluate: uniform, onProgress: count => progress.push(count) })
    expect(progress).toEqual([1, 2, 3]); expect(result.simulations).toBe(3)
    expect(Board.initial().legalMoves()).toContain(result.move)
  })
})
