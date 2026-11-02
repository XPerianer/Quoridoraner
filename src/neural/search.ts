import { Board } from '../engine/board'
import type { Position } from '../engine/types'
import { canonicalMove } from './features'

export interface NeuralEvaluation { logits: Float32Array; value: number }
export interface NeuralSearchOptions {
  positions: Position[]
  simulations: number
  maxTimeMs?: number
  evaluate: (position: Position, repetitions: number, remaining: number) => Promise<NeuralEvaluation>
  onProgress?: (completed: number) => void
}
interface Edge { move: number; prior: number; child?: Node }
interface Node { position: Position; visits: number; total: number; edges?: Edge[] }
const node = (position: Position): Node => ({ position, visits: 0, total: 0 })

/** Ignore property insertion order and wall-list ordering in repetition detection. */
export function positionKey(p: Position): string {
  return JSON.stringify([p.pawns, p.wallsLeft, [...p.hWalls].sort((a, b) => a - b), [...p.vWalls].sort((a, b) => a - b), p.turn])
}

/** Deterministic PUCT, cpuct=1.5 and FPU=0.25 as in the native AZ engine.
 * Search trees are request-local, so no stale position or model can reuse visits.
 * Native race-solving and Gumbel search are intentionally not part of this backend.
 */
export async function searchNeural({ positions, simulations, maxTimeMs, evaluate, onProgress }: NeuralSearchOptions): Promise<{ move: number; simulations: number }> {
  if (!positions.length) throw new Error('Search requires a position history')
  if (!Number.isInteger(simulations) || simulations < 1 || simulations > 2000) throw new Error('Simulation budget must be between 1 and 2000')
  if (maxTimeMs !== undefined && (!Number.isFinite(maxTimeMs) || maxTimeMs <= 0)) throw new Error('Invalid search time budget')
  const ply = positions.length - 1, maxPlies = ply < 100 ? 100 : 200
  const root = node(positions[ply])
  const history = new Map<string, number>()
  for (const p of positions) { const key = positionKey(p); history.set(key, (history.get(key) ?? 0) + 1) }
  const rootBoard = Board.from(root.position)
  if (rootBoard.winner() !== -1 || ply >= 200 || (history.get(positionKey(root.position)) ?? 0) >= 3) throw new Error('Cannot search a finished game')
  const started = performance.now()
  async function expand(current: Node, repetitions: number, depth: number): Promise<number> {
    const legal = Board.from(current.position).legalMoves()
    if (!legal.length) throw new Error('Position has no legal moves')
    const result = await evaluate(current.position, repetitions, (maxPlies - ply - depth) / maxPlies)
    if (result.logits.length !== 209 || !Number.isFinite(result.value) || !result.logits.every(Number.isFinite)) throw new Error('Model returned invalid policy or value')
    const logits = legal.map(move => result.logits[canonicalMove(move, current.position.turn)])
    const high = Math.max(...logits)
    const weights = logits.map(logit => Math.exp(logit - high))
    const sum = weights.reduce((a, b) => a + b, 0)
    current.edges = legal.map((move, i) => ({ move, prior: weights[i] / sum }))
    return Math.max(-1, Math.min(1, result.value))
  }
  root.total = await expand(root, history.get(positionKey(root.position)) ?? 1, 0)
  root.visits = 1 // Native root expansion is separate from the simulation budget.
  let completed = 0
  for (; completed < simulations;) {
    const seen = new Map(history)
    const path = [root]
    let current = root, depth = 0, value = 0
    while (true) {
      const board = Board.from(current.position)
      const winner = board.winner()
      const repetitions = seen.get(positionKey(current.position)) ?? 0
      if (winner !== -1) { value = winner === current.position.turn ? 1 : -1; break }
      if (repetitions >= 3 || ply + depth >= maxPlies) break
      if (!current.edges) { value = await expand(current, repetitions, depth); break }
      const urgency = current.total / current.visits - 0.25
      const scale = 1.5 * Math.sqrt(Math.max(1, current.visits))
      let best = -Infinity, selected: Edge | undefined
      for (const edge of current.edges) {
        const visits = edge.child?.visits ?? 0
        const q = visits ? -edge.child!.total / visits : urgency
        const score = q + scale * edge.prior / (1 + visits)
        if (score > best) { best = score; selected = edge }
      }
      if (!selected) throw new Error('Empty legal policy')
      if (!selected.child) { board.make(selected.move); selected.child = node(board.toPosition()) }
      current = selected.child
      depth++
      const key = positionKey(current.position)
      seen.set(key, (seen.get(key) ?? 0) + 1)
      path.push(current)
    }
    for (let i = path.length - 1; i >= 0; i--) { path[i].visits++; path[i].total += value; value = -value }
    completed++
    onProgress?.(completed)
    if (maxTimeMs !== undefined && performance.now() - started >= maxTimeMs) break
  }
  // First legal action wins ties, matching argmax over the native policy vector.
  const edges = [...root.edges!].sort((a, b) => a.move - b.move)
  const best = edges.reduce((a, b) => (b.child?.visits ?? 0) > (a.child?.visits ?? 0) ? b : a)
  return { move: best.move, simulations: completed }
}
