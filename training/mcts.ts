import { typescriptRules, winner } from './rules'
import type { Rules } from './rules'
import type { Position } from '../src/engine/types'

export function positionKey(p: Position): string {
  // Same canonical representation as Python core.key().
  return JSON.stringify({ hWalls: p.hWalls, pawns: p.pawns, turn: p.turn, vWalls: p.vWalls, wallsLeft: p.wallsLeft })
}

interface TreeNode {
  position: Position
  winner: number
  priors: [number, number][] | null
  children: Record<string, TreeNode>
  visits: number
  total: number
}

function node(position: Position): TreeNode {
  return { position, winner: winner(position), priors: null, children: {}, visits: 0, total: 0 }
}

export interface SearchRequest {
  id: string
  position: Position
  history: Record<string, number>
  ply: number
  maxPlies: number
  simulations: number
  noise?: number[]
}

export interface Evaluation {
  id: string
  policy: number[]
  value: number
}

interface Leaf {
  id: string
  position: Position
  legal: number[]
  repetitions: number
  remaining: number
}

interface Job {
  request: SearchRequest
  root: TreeNode
  completed: number
  pending: { node: TreeNode; path: TreeNode[]; legal: number[] } | null
  initialVisits: number
}

export class MctsPool {
  private readonly rules: Rules
  constructor(rules: Rules = typescriptRules) { this.rules = rules }
  private roots = new Map<string, TreeNode>()
  private jobs: Job[] = []

  clear(): void { this.roots.clear(); this.jobs = [] }
  drop(ids: string[]): void { for (const id of ids) this.roots.delete(id) }
  snapshot(): [string, TreeNode][] {
    if (this.jobs.some(j => j.pending || j.completed < j.request.simulations)) throw new Error('Search still in progress')
    return [...this.roots]
  }
  restore(snapshot: [string, TreeNode][]): void { this.roots = new Map(snapshot); this.jobs = [] }

  advance(id: string, position: Position, move: number): Position {
    if (winner(position) !== -1 || !this.rules.legal(position).includes(move)) throw new Error('Illegal move')
    const next = this.rules.step(position, move)
    const root = this.roots.get(id)
    this.roots.set(id, root && positionKey(root.position) === positionKey(position)
      ? root.children[move] ?? node(next) : node(next))
    return next
  }

  begin(requests: SearchRequest[]): ReturnType<MctsPool['step']> {
    if (new Set(requests.map(r => r.id)).size !== requests.length) throw new Error('Duplicate search id')
    this.jobs = requests.map(request => {
      let root = this.roots.get(request.id)
      if (!root || positionKey(root.position) !== positionKey(request.position)) root = node(request.position)
      if (root.winner !== -1 || request.ply >= request.maxPlies || (request.history[positionKey(root.position)] ?? 0) >= 3) {
        throw new Error('Cannot search a finished game')
      }
      if (!Number.isInteger(request.simulations) || request.simulations < 1) throw new Error('Invalid simulation budget')
      this.roots.set(request.id, root)
      return { request, root, completed: 0, pending: null, initialVisits: root.visits }
    })
    return this.step([])
  }

  step(evaluations: Evaluation[]): { leaves: Leaf[]; done: boolean; policies: Record<string, number[]>; reused: number } {
    const incoming = new Map(evaluations.map(e => [e.id, e]))
    for (const job of this.jobs) {
      if (!job.pending) continue
      const result = incoming.get(job.request.id)
      if (!result) throw new Error('Missing neural evaluation')
      const { node: leaf, path, legal } = job.pending
      if (result.policy.length !== legal.length || !Number.isFinite(result.value)
          || result.policy.some(p => !Number.isFinite(p) || p < 0)) throw new Error('Invalid neural evaluation')
      const sum = result.policy.reduce((a, b) => a + b, 0)
      if (sum <= 0) throw new Error('Empty neural policy')
      leaf.priors = legal.map((move, i) => [move, result.policy[i] / sum])
      if (path.length) this.backup(job, path, result.value)
      job.pending = null
    }
    const leaves: Leaf[] = []
    for (const job of this.jobs) {
      if (job.root.priors === null) {
        leaves.push(this.leaf(job, job.root, [], job.request.history, 0))
        continue
      }
      while (job.completed < job.request.simulations) {
        let current = job.root
        const path = [current]
        // Only path-local repetitions need writes. Inherit the immutable game
        // history instead of copying up to maxPlies entries for every visit.
        const counts: Record<string, number> = Object.create(job.request.history)
        let depth = 0
        while (true) {
          const repetition = counts[positionKey(current.position)] ?? 1
          if (current.winner !== -1) {
            this.backup(job, path, current.winner === current.position.turn ? 1 : -1)
            break
          }
          if (repetition >= 3 || job.request.ply + depth >= job.request.maxPlies) {
            this.backup(job, path, 0)
            break
          }
          if (current.priors === null) {
            leaves.push(this.leaf(job, current, path, counts, depth))
            break
          }
          let best = -Infinity
          let chosen = current.priors[0]![0]
          for (let i = 0; i < current.priors.length; i++) {
            const [move, base] = current.priors[i]!
            const prior = current === job.root && job.request.noise ? 0.75 * base + 0.25 * job.request.noise[i]! : base
            const child = current.children[move]
            const visits = child?.visits ?? 0
            const q = visits ? -child!.total / visits : 0
            const score = q + 1.5 * prior * Math.sqrt(Math.max(1, current.visits)) / (1 + visits)
            if (score > best) { best = score; chosen = move }
          }
          let child = current.children[chosen]
          if (!child) {
            child = node(this.rules.step(current.position, chosen))
            current.children[chosen] = child
          }
          current = child
          path.push(current)
          depth++
          const key = positionKey(current.position)
          counts[key] = (counts[key] ?? 0) + 1
        }
        if (job.pending) break
      }
    }
    const done = this.jobs.every(j => j.completed === j.request.simulations)
    const policies: Record<string, number[]> = {}
    if (done) for (const job of this.jobs) {
      const policy = Array<number>(209).fill(0)
      const sum = Object.values(job.root.children).reduce((a, child) => a + child.visits, 0)
      for (const [move, child] of Object.entries(job.root.children)) policy[Number(move)] = child.visits / sum
      policies[job.request.id] = policy
    }
    return { leaves, done, policies, reused: this.jobs.reduce((n, j) => n + j.initialVisits, 0) }
  }

  private leaf(job: Job, current: TreeNode, path: TreeNode[], counts: Record<string, number>, depth: number): Leaf {
    const legal = this.rules.legal(current.position)
    job.pending = { node: current, path, legal }
    return { id: job.request.id, position: current.position, legal,
      repetitions: counts[positionKey(current.position)] ?? 1,
      remaining: (job.request.maxPlies - job.request.ply - depth) / job.request.maxPlies }
  }

  private backup(job: Job, path: TreeNode[], value: number): void {
    for (let i = path.length - 1; i >= 0; i--) {
      path[i]!.visits++
      path[i]!.total += value
      value = -value
    }
    job.completed++
  }
}
