import { Board } from '../src/engine/board'
import type { Position } from '../src/engine/types'

export interface Rules {
  legal(position: Position): number[]
  step(position: Position, move: number): Position
}
export const winner = (p: Position): number => p.pawns[0] < 9 ? 0 : p.pawns[1] >= 72 ? 1 : -1
export const typescriptRules: Rules = {
  legal: position => winner(position) < 0 ? Board.from(position).legalMoves() : [],
  step: (position, move) => {
    const board = Board.from(position)
    board.make(move)
    return board.toPosition()
  },
}
