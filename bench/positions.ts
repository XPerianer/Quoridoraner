import { Board } from '../src/engine/board'
import { boundedDistance, fusedDistance, queueDistance, scalarDistance } from '../src/engine/distance'
import type { Position } from '../src/engine/types'

export const variants = {
  scalar: (p: Position) => attach(p, scalarDistance),
  baseline: (p: Position) => {
    const b = Board.from(p)
    b.distance = b.distanceReference
    return b
  },
  production: (p: Position) => Board.from(p),
  queue: (p: Position) => attach(p, queueDistance()),
  fused: (p: Position) => attach(p, fusedDistance()),
  bounded: (p: Position) => attach(p, boundedDistance()),
}
function attach(p: Position, kernel: ReturnType<typeof fusedDistance>): Board {
  const b = Board.from(p)
  b.distance = (player) => kernel(b, player)
  return b
}
export function corpus(): Position[] {
  const positions: Position[] = [Board.initial().toPosition()]
  let seed = 173
  for (let game = 0; game < 8; game++) {
    const b = Board.initial()
    for (let ply = 0; ply < 70 && b.winner() < 0; ply++) {
      const moves = b.legalMoves()
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5
      b.make(moves[(seed >>> 0) % moves.length])
      if (ply % 7 === 6 && b.winner() < 0) positions.push(b.toPosition())
    }
  }
  return positions
}
