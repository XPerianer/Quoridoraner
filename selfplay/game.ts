import { Board, MAX_MOVES } from '../src/engine/board'
import { TsSearcher } from '../src/engine/search'
import type { Move, Player } from '../src/engine/types'
import type { EngineParams } from './config'

export type Termination = 'goal' | 'maxPlies' | 'repetition'

export interface MoveRecord {
  move: Move
  depth: number
  nodes: number
  score: number
  ms: number
}

export interface GameTask {
  id: number
  pair: number
  matchup: string
  /** Engine names and parameters, index = player (0 = White, moves first). */
  names: [string, string]
  params: [EngineParams, EngineParams]
  opening: Move[]
  maxPlies: number
}

export interface GameRecord {
  id: number
  pair: number
  matchup: string
  names: [string, string]
  opening: Move[]
  moves: MoveRecord[]
  winner: Player | -1
  termination: Termination
  plies: number
  ms: number
}

/** Seeded PRNG so openings are reproducible across runs. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function randomOpenings(count: number, plies: number, wallProb: number, seed: number): Move[][] {
  const rand = mulberry32(seed)
  const seen = new Set<string>()
  const out: Move[][] = []
  const buf = new Int16Array(MAX_MOVES)
  for (let attempts = 0; out.length < count; attempts++) {
    if (attempts > count * 1000) throw new Error(`could only find ${out.length} distinct openings of ${plies} plies`)
    const b = Board.initial()
    const line: Move[] = []
    while (line.length < plies && b.winner() === -1) {
      const n = b.generate(buf, false)
      const moves = [...buf.subarray(0, n)]
      const pawn = moves.filter((m) => m < 81)
      const walls = moves.filter((m) => m >= 81)
      const pool = walls.length > 0 && rand() < wallProb ? walls : pawn
      const m = pool[Math.floor(rand() * pool.length)]
      b.make(m)
      line.push(m)
    }
    const key = `${b.hash1}:${b.hash2}`
    if (b.winner() !== -1 || seen.has(key)) continue
    seen.add(key)
    out.push(line)
  }
  return out
}

export function playGame(task: GameTask): GameRecord {
  const start = performance.now()
  const board = Board.initial()
  for (const m of task.opening) board.make(m)
  const engines = [new TsSearcher(), new TsSearcher()]
  const seen = new Map<string, number>()
  const moves: MoveRecord[] = []
  let termination: Termination = 'maxPlies'
  while (task.opening.length + moves.length < task.maxPlies) {
    if (board.winner() !== -1) {
      termination = 'goal'
      break
    }
    const key = `${board.hash1}:${board.hash2}`
    const count = (seen.get(key) ?? 0) + 1
    if (count >= 3) {
      termination = 'repetition'
      break
    }
    seen.set(key, count)
    const t0 = performance.now()
    const p = board.turn
    const s = engines[p].search(board.toPosition(), { ...task.params[p], reportEveryMs: Number.MAX_SAFE_INTEGER }, () => {})
    moves.push({ move: s.bestMove, depth: s.iterations.length, nodes: s.nodes, score: s.score, ms: performance.now() - t0 })
    board.make(s.bestMove)
  }
  if (board.winner() !== -1) termination = 'goal'
  return {
    id: task.id,
    pair: task.pair,
    matchup: task.matchup,
    names: task.names,
    opening: task.opening,
    moves,
    winner: board.winner(),
    termination,
    plies: task.opening.length + moves.length,
    ms: performance.now() - start,
  }
}
