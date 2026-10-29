/** Batched neural-search bridge; board traversal stays entirely in TypeScript. */
import { createInterface } from 'node:readline'
import { Board } from '../src/engine/board'
import { DEFAULT_OPTIONS, TsSearcher } from '../src/engine/search'
import type { Position } from '../src/engine/types'
import { MctsPool } from './mcts'
import { typescriptRules, winner } from './rules'
import { loadNative, nativeRules } from './native'
import type { Evaluation, SearchRequest } from './mcts'

const backend = process.env.QUORIDOR_ENGINE ?? 'typescript'
if (!['typescript', 'native', 'native-negamax'].includes(backend)) throw new Error('Unknown engine backend')
const native = backend !== 'typescript' ? loadNative() : null
// Benchmarks show native legal-move calls lose to TypeScript; keep MCTS rules native only for full native trials.
const rules = backend === 'native' ? nativeRules(native!) : typescriptRules
const pool = new MctsPool(rules)
const engine = native ? null : new TsSearcher()
function describe(position: Position) {
  return { position, winner: winner(position), legal: rules.legal(position) }
}
function negamax(position: Position, nodes = 2000) {
  if (native) return native.search(position, nodes)
  engine!.clearCache()
  const result = engine!.search(position, { ...DEFAULT_OPTIONS, maxNodes: nodes, maxDepth: 64,
    timeMs: 1e9, reportEveryMs: 1e9 }, () => {})
  return { move: result.bestMove, nodes: result.nodes, score: result.score }
}

for await (const line of createInterface({ input: process.stdin })) {
  try {
    const req = JSON.parse(line) as {
      position?: Position; move?: number; op?: string; nodes?: number
      requests: SearchRequest[]; evaluations: Evaluation[]; ids: string[]
      snapshot: Parameters<MctsPool['restore']>[0]
      moves: { id: string; position: Position; move: number }[]
      positions: Position[]
    }
    let result: unknown
    switch (req.op) {
      case 'mcts_begin': result = pool.begin(req.requests); break
      case 'mcts_step': result = pool.step(req.evaluations); break
      case 'snapshot': result = { trees: pool.snapshot() }; break
      case 'restore': pool.restore(req.snapshot); result = {}; break
      case 'clear': pool.clear(); result = {}; break
      case 'drop': pool.drop(req.ids); result = {}; break
      case 'advance': result = { states: req.moves.map(m => describe(pool.advance(m.id, m.position, m.move))) }; break
      case 'negamax_batch': result = { results: req.positions.map(p => negamax(p, req.nodes)) }; break
      default: {
        let position = req.position ?? Board.initial().toPosition()
        if (req.move !== undefined) {
          if (winner(position) !== -1 || !rules.legal(position).includes(req.move)) throw new Error('Illegal move')
          position = rules.step(position, req.move)
        }
        result = req.op === 'negamax' ? negamax(position, req.nodes) : describe(position)
      }
    }
    process.stdout.write(JSON.stringify(result) + '\n')
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: String(error) }) + '\n')
  }
}
