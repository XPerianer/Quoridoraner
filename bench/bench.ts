import { Board, moveToString } from '../src/engine/board'
import { DEFAULT_OPTIONS, TsSearcher } from '../src/engine/search'
import type { MoveGen } from '../src/engine/types'

const moveGen = (process.argv[2] ?? 'likely') as MoveGen
const maxDepth = Number(process.argv[3] ?? 4)
const s = new TsSearcher().search(
  Board.initial().toPosition(),
  { ...DEFAULT_OPTIONS, maxDepth, timeMs: 60_000, moveGen, reportEveryMs: 1e9 },
  () => {},
)
for (const it of s.iterations) {
  console.log(
    `depth ${it.depth}  nodes ${it.nodes.toString().padStart(9)}  ${it.timeMs.toFixed(0).padStart(6)} ms  ` +
      `${((it.nodes / it.timeMs) | 0).toString().padStart(6)} kN/s  best ${moveToString(it.bestMove)}  score ${it.score}  pv ${it.pv.map(moveToString).join(' ')}`,
  )
}
