import { Board } from '../engine/board'
import type { Position } from '../engine/types'

export const SIMPLIFIED_PLANES = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13] as const

/** Real <-> side-to-move action mapping; applying it twice restores the action. */
export function canonicalMove(move: number, turn: number): number {
  if (turn === 0) return move
  const base = move < 81 ? 0 : move < 145 ? 81 : 145
  const width = base === 0 ? 9 : 8
  const i = move - base
  return base + (width - 1 - Math.floor(i / width)) * width + i % width
}

function fromPawn(board: Board, start: number): Int8Array {
  const distances = new Int8Array(81).fill(-1)
  const queue = new Int16Array(81)
  let head = 0, tail = 0
  distances[start] = 0
  queue[tail++] = start
  const steps = [-9, 1, 9, -1]
  while (head < tail) {
    const cell = queue[head++]
    for (let direction = 0; direction < 4; direction++) {
      if (!board.canStep(cell % 9, Math.floor(cell / 9), direction)) continue
      const next = cell + steps[direction]
      if (distances[next] >= 0) continue
      distances[next] = distances[cell] + 1
      queue[tail++] = next
    }
  }
  return distances
}

/** Native AZ NCHW encoder, without training-time mirror augmentation. */
export function encodeFeatures(position: Position, repetitions = 1, remaining = 1, planes = 19): Float32Array {
  if (planes !== 19 && planes !== 10) throw new Error(`Unsupported feature plane count: ${planes}`)
  const board = Board.from(position)
  const me = position.turn, opponent = me ^ 1
  const full = new Float32Array(19 * 81)
  const set = (plane: number, cell: number, value: number) => { full[plane * 81 + cell] = value }
  set(0, canonicalMove(board.pawn[me], me), 1)
  set(1, canonicalMove(board.pawn[opponent], me), 1)
  for (const [plane, walls, base] of [[2, position.hWalls, 81], [3, position.vWalls, 145]] as const) {
    for (const wall of walls) {
      const anchor = canonicalMove(base + wall, me) - base
      set(plane, Math.floor(anchor / 8) * 9 + anchor % 8, 1)
    }
  }
  const goal = [new Int8Array(81), new Int8Array(81)]
  board.distanceMap(0, goal[0]); board.distanceMap(1, goal[1])
  const from = [fromPawn(board, board.pawn[0]), fromPawn(board, board.pawn[1])]
  const mine = goal[me][board.pawn[me]], theirs = goal[opponent][board.pawn[opponent]]
  for (let cell = 0; cell < 81; cell++) {
    const target = canonicalMove(cell, me)
    const directions = me === 1 ? [2, 0, 3, 1] : [0, 2, 3, 1]
    for (let d = 0; d < 4; d++) set(4 + d, target, Number(!board.canStep(cell % 9, Math.floor(cell / 9), directions[d])))
    set(8, target, goal[me][cell] < 0 ? 1 : goal[me][cell] / 16)
    set(9, target, goal[opponent][cell] < 0 ? 1 : goal[opponent][cell] / 16)
    set(10, target, Number(from[me][cell] >= 0 && from[me][cell] + goal[me][cell] === mine))
    set(11, target, Number(from[opponent][cell] >= 0 && from[opponent][cell] + goal[opponent][cell] === theirs))
  }
  const constants = [board.wallsLeft[me] / 10, board.wallsLeft[opponent] / 10,
    Math.max(-1, Math.min(1, (theirs - mine) / 10)), me, Math.min(repetitions, 3) / 3, remaining, 1]
  constants.forEach((value, index) => full.fill(value, (12 + index) * 81, (13 + index) * 81))
  if (planes === 19) return full
  const compact = new Float32Array(10 * 81)
  SIMPLIFIED_PLANES.forEach((plane, index) => compact.set(full.subarray(plane * 81, (plane + 1) * 81), index * 81))
  return compact
}
