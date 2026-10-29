/** Deterministic same-position, same-budget comparison, including Node-API crossings. */
import { writeFileSync } from 'node:fs'
import { corpus } from '../bench/positions'
import { DEFAULT_OPTIONS, TsSearcher } from '../src/engine/search'
import { loadNative } from './native'
import { typescriptRules } from './rules'

const native = loadNative()
const positions = corpus()
const engine = new TsSearcher()
const repeats = 5
const median = (values: number[]) => [...values].sort((a,b) => a-b)[Math.floor(values.length/2)]!
const rows: { operation: string; typescriptMs: number; nativeMs: number; speedup: number; checksum: number }[] = []
function compare(operation: string, ts: () => number, cpp: () => number) {
  const a: number[] = [], b: number[] = []
  let checksum = 0
  for (let repeat = 0; repeat <= repeats; repeat++) {
    const run = (fn: () => number, samples: number[]) => {
      const start = performance.now()
      const value = fn()
      if (repeat) samples.push(performance.now()-start)
      return value
    }
    // Alternate ordering to reduce systematic temperature/warmup bias.
    const values = repeat % 2 ? [run(ts,a), run(cpp,b)] : [run(cpp,b), run(ts,a)]
    if (values[0] !== values[1]) throw new Error(`${operation}: result mismatch`)
    checksum = values[0]!
  }
  const row = { operation, typescriptMs: median(a), nativeMs: median(b), speedup: median(a)/median(b), checksum }
  rows.push(row)
  console.log(`${operation}: TS ${row.typescriptMs.toFixed(2)} ms / native ${row.nativeMs.toFixed(2)} ms = ${row.speedup.toFixed(2)}x`)
}
compare('legal moves (81 positions × 100)', () => {
  let sum = 0
  for (let i = 0; i < 100; i++) for (const p of positions) sum += typescriptRules.legal(p).length
  return sum
}, () => {
  let sum = 0
  for (let i = 0; i < 100; i++) for (const p of positions) sum += native.legal(p).length
  return sum
})
for (const maxNodes of [2000,50000]) compare(`negamax ${maxNodes} nodes (${positions.length} positions)`, () => {
  let sum = 0
  for (const p of positions) {
    engine.clearCache()
    const result = engine.search(p,{ ...DEFAULT_OPTIONS, maxNodes, maxDepth: 64, timeMs: 1e9, reportEveryMs: 1e9 },() => {})
    sum += result.bestMove + result.score + result.nodes
  }
  return sum
}, () => {
  let sum = 0
  for (const p of positions) { const r = native.search(p,maxNodes); sum += r.move + r.score + r.nodes }
  return sum
})
const output = process.argv[2]
if (output) writeFileSync(output, JSON.stringify({ positions: positions.length, repeats, rows },null,2)+'\n')
