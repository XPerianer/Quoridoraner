/**
 * Plain-data contract between the UI and any engine backend (TypeScript today, WebAssembly later).
 * Everything here is structured-clone friendly so it can cross a Worker boundary unchanged.
 */

export type Player = 0 | 1

/** 0..80: pawn to cell (y*9+x); 81..144: horizontal wall at anchor (y*8+x); 145..208: vertical wall. */
export type Move = number

export interface Position {
  pawns: [number, number]
  wallsLeft: [number, number]
  hWalls: number[]
  vWalls: number[]
  turn: Player
}

export type MoveGen = 'likely' | 'all'

export interface SearchOptions {
  maxDepth: number
  timeMs: number
  /** Stop after roughly this many nodes; deterministic alternative to timeMs. */
  maxNodes: number
  moveGen: MoveGen
  /** Search every root move with a full window so each gets an exact score (the original C++ behaviour). */
  exactRoot: boolean
  /** Eval points per wall in hand difference (one path step = 10). */
  wallWeight: number
  /** Eval points for being the side to move. */
  tempo: number
  reportEveryMs: number
}

/** exact: true value; upper: true value <= score; lower: true value >= score. */
export type Bound = 'exact' | 'upper' | 'lower'

export interface ReplyStat {
  move: Move
  /** From the root mover's perspective. */
  score: number
  bound: Bound
  nodes: number
}

export interface RootMoveStat {
  move: Move
  score: number
  bound: Bound
  nodes: number
  depth: number
  replies: ReplyStat[]
  replyCount: number
  repliesSearched: number
}

export interface IterationStat {
  depth: number
  nodes: number
  timeMs: number
  bestMove: Move
  score: number
  pv: Move[]
}

export interface SearchSnapshot {
  done: boolean
  backend: string
  depth: number
  nodes: number
  elapsedMs: number
  bestMove: Move
  score: number
  pv: Move[]
  currentMove: Move
  currentIndex: number
  root: RootMoveStat[]
  iterations: IterationStat[]
  plyNodes: number[]
  plyCutoffs: number[]
  ttProbes: number
  ttHits: number
  /** Per player, how often the search moved that pawn onto each cell. */
  pawnHeat: [number[], number[]]
  /** How often each wall anchor was tried, [horizontal, vertical]. */
  wallHeat: [number[], number[]]
}

export interface EngineBackend {
  readonly name: string
  search(position: Position, options: SearchOptions, onProgress: (s: SearchSnapshot) => void): SearchSnapshot
}

export type WorkerRequest = { id: number; position: Position; options: SearchOptions }
export type WorkerResponse = { id: number; type: 'progress' | 'done'; snapshot: SearchSnapshot }
