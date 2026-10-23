import { writeFileSync } from 'node:fs'
import os from 'node:os'
import { Board, MAX_MOVES } from '../src/engine/board'
import { corpus, variants } from './positions'
import { DEFAULT_OPTIONS, TsSearcher } from '../src/engine/search'

const positions = corpus()
writeFileSync('bench/positions.json', JSON.stringify(positions.map(p => ({ ...p, expected: [Board.from(p).distanceReference(0), Board.from(p).distanceReference(1)] }))))
const results: Record<string, unknown> = {}
let checksum = 0
function time(fn: () => void): number {
  const t = performance.now(); fn(); return performance.now() - t
}
function median(a: number[]): number { return a.sort((x, y) => x - y)[a.length >> 1] }
const samples: Record<string, Record<string, number[]>> = {}
const factories = Object.entries(variants)
for (const [name] of factories) samples[name] = { distance: [], generate: [], search: [] }
// Rotate order; discard first two rounds for JIT warmup. Each search uses a fresh TT.
for (let round = 0; round < 7; round++) {
  for (let k = 0; k < factories.length; k++) {
    const [name, factory] = factories[(k + round) % factories.length]
    const boards = positions.map(factory), out = new Int16Array(MAX_MOVES)
    const distance = time(() => { for (let i = 0; i < 300; i++) for (const b of boards) checksum += b.distance(0) + b.distance(1) })
    const generate = time(() => { for (let i = 0; i < 10; i++) for (const b of boards) checksum += b.generate(out, false) })
    const search = time(() => {
      for (const index of [0, 2, 5, 9]) {
        const s = new TsSearcher(factory).search(positions[index], {
          ...DEFAULT_OPTIONS, maxDepth: 4, timeMs: 1e9, reportEveryMs: 1e9,
        }, () => {})
        checksum += s.nodes + s.score
      }
    })
    if (round >= 2) for (const [metric, value] of Object.entries({ distance, generate, search })) samples[name][metric].push(value)
  }
}
for (const [name, metrics] of Object.entries(samples)) {
  results[name] = Object.fromEntries(Object.entries(metrics).map(([metric, values]) => [metric, { medianMs: median([...values]), samplesMs: values }]))
}
const report = { date: new Date().toISOString(), cpu: os.cpus()[0].model, node: process.version, positions: positions.length, distanceCalls: positions.length * 600, generateCalls: positions.length * 10, searchDepth: 4, results, checksum }
writeFileSync('bench/kernel-results.json', JSON.stringify(report, null, 2) + '\n')
console.log(JSON.stringify(report, null, 2))
