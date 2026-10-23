import { describe, expect, it } from 'vitest'
import { Board, MAX_MOVES, moveToString, wallMove } from './board'
import type { Position } from './types'

const pos = (p: Partial<Position>): Position => ({
  pawns: [76, 4],
  wallsLeft: [10, 10],
  hWalls: [],
  vWalls: [],
  turn: 0,
  ...p,
})

const cell = (x: number, y: number) => y * 9 + x

describe('Board', () => {
  it('starts with 3 pawn moves and 128 walls', () => {
    const b = Board.initial()
    const moves = b.legalMoves()
    expect(moves.filter((m) => m < 81).sort()).toEqual([cell(3, 8), cell(5, 8), cell(4, 7)].sort())
    expect(moves.length).toBe(3 + 128)
    expect(b.distance(0)).toBe(8)
    expect(b.distance(1)).toBe(8)
  })

  it('jumps straight over the opponent', () => {
    const b = Board.from(pos({ pawns: [cell(4, 5), cell(4, 4)] }))
    expect(b.legalMoves().filter((m) => m < 81)).toContain(cell(4, 3))
    expect(b.legalMoves().filter((m) => m < 81)).not.toContain(cell(4, 4))
  })

  it('jumps diagonally when a wall is behind the opponent', () => {
    const b = Board.from(pos({ pawns: [cell(4, 5), cell(4, 4)], hWalls: [3 * 8 + 4] }))
    const pawnMoves = b.legalMoves().filter((m) => m < 81).sort((a, c) => a - c)
    expect(pawnMoves).toEqual([cell(3, 4), cell(5, 4), cell(3, 5), cell(5, 5), cell(4, 6)].sort((a, c) => a - c))
  })

  it('rejects overlapping and crossing walls', () => {
    const b = Board.from(pos({ hWalls: [3 * 8 + 3] }))
    expect(b.wallFits(false, 3, 3)).toBe(false)
    expect(b.wallFits(false, 2, 3)).toBe(false)
    expect(b.wallFits(false, 4, 3)).toBe(false)
    expect(b.wallFits(true, 3, 3)).toBe(false)
    expect(b.wallFits(false, 5, 3)).toBe(true)
    expect(b.wallFits(true, 3, 2)).toBe(true)
  })

  it('never allows a wall that seals a pawn in', () => {
    const b = Board.from(pos({ pawns: [cell(0, 8), cell(4, 0)], hWalls: [7 * 8 + 0], turn: 1 }))
    expect(b.wallFits(true, 1, 7)).toBe(true)
    expect(b.wallLegal(true, 1, 7)).toBe(false)
    expect(b.wallLegal(true, 1, 6)).toBe(true)
  })

  it('keeps hash and state consistent through make/unmake', () => {
    const b = Board.initial()
    const before = JSON.stringify(b.toPosition())
    const h = [b.hash1, b.hash2]
    const buf = new Int16Array(MAX_MOVES)
    let seed = 7
    const played: number[] = []
    for (let i = 0; i < 40 && b.winner() === -1; i++) {
      const n = b.generate(buf, false)
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      const m = buf[seed % n]
      b.make(m)
      played.push(m)
      expect(b.checkHash()).toBe(true)
      expect(b.distance(0)).toBeGreaterThanOrEqual(0)
      expect(b.distance(1)).toBeGreaterThanOrEqual(0)
    }
    for (let i = played.length - 1; i >= 0; i--) b.unmake(played[i])
    expect(JSON.stringify(b.toPosition())).toBe(before)
    expect([b.hash1, b.hash2]).toEqual(h)
  })

  it('distance flood fill matches BFS distance map', () => {
    const b = Board.from(pos({ hWalls: [3, 4 * 8 + 0, 4 * 8 + 2, 4 * 8 + 4], vWalls: [5 * 8 + 5, 2 * 8 + 1] }))
    const map = new Int8Array(81)
    b.distanceMap(0, map)
    expect(b.distance(0)).toBe(map[b.pawn[0]])
    b.distanceMap(1, map)
    expect(b.distance(1)).toBe(map[b.pawn[1]])
  })

  it('formats moves', () => {
    expect(moveToString(cell(4, 8))).toBe('e1')
    expect(moveToString(wallMove(true, 0, 0))).toBe('a9v')
  })
})
