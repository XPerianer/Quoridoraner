import { execSync } from 'node:child_process'
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { availableParallelism } from 'node:os'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import { Worker } from 'node:worker_threads'
import { loadConfig, matchups, resolveParams } from './config'
import { randomOpenings, type GameRecord, type GameTask } from './game'
import { formatStatsLine, writeReport } from './report'
import { matchStats } from './stats'

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    pairs: { type: 'string' },
    concurrency: { type: 'string', default: String(Math.max(1, availableParallelism() - 1)) },
    out: { type: 'string', default: 'experiments/selfplay' },
  },
})
if (positionals.length !== 1) {
  console.error('usage: npm run selfplay -- <experiment.json> [--pairs N] [--concurrency N] [--out DIR]')
  process.exit(1)
}

const cfg = loadConfig(positionals[0])
if (values.pairs) cfg.pairs = Number(values.pairs)
const concurrency = Number(values.concurrency)
const plan = matchups(cfg)
const names = new Set([cfg.baseline.name, ...plan.map((m) => m.candidate.name)])
if (names.size !== plan.length + 1) throw new Error('engine names must be unique (baseline and every candidate)')

const openings = randomOpenings(cfg.pairs, cfg.openingPlies, cfg.openingWallProb, cfg.seed)
const baseline = resolveParams(cfg.base, cfg.baseline)
const tasks: GameTask[] = []
for (const m of plan) {
  const cand = resolveParams(cfg.base, m.candidate)
  openings.forEach((opening, pair) => {
    for (const candidateWhite of [true, false]) {
      tasks.push({
        id: tasks.length,
        pair,
        matchup: m.candidate.name,
        names: candidateWhite ? [m.candidate.name, cfg.baseline.name] : [cfg.baseline.name, m.candidate.name],
        params: candidateWhite ? [cand, baseline] : [baseline, cand],
        opening,
        maxPlies: cfg.maxPlies,
      })
    }
  })
}

const git = (cmd: string) => execSync(`git ${cmd}`, { encoding: 'utf8' }).trim()
const started = new Date()
const stamp = started.toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15)
const dir = join(values.out!, `${stamp}-${cfg.name}`)
mkdirSync(dir, { recursive: true })
writeFileSync(
  join(dir, 'config.json'),
  JSON.stringify(
    {
      config: cfg,
      startedAt: started.toISOString(),
      commit: git('rev-parse --short HEAD'),
      dirty: git('status --porcelain') !== '',
      concurrency,
      baseline: { name: cfg.baseline.name, params: baseline },
      matchups: plan.map((m) => ({ name: m.candidate.name, varied: m.varied, params: resolveParams(cfg.base, m.candidate) })),
    },
    null,
    2,
  ),
)
const gamesPath = join(dir, 'games.jsonl')
writeFileSync(gamesPath, '')

console.log(`${cfg.name}: ${plan.length} candidate(s) vs ${cfg.baseline.name}, ${cfg.pairs} pairs each, ${tasks.length} games on ${concurrency} threads`)
console.log(`writing to ${dir}\n`)

const results = new Map<string, GameRecord[]>(plan.map((m) => [m.candidate.name, []]))
const perMatchup = cfg.pairs * 2
let done = 0
let next = 0
const t0 = performance.now()

function progress(): void {
  if (!process.stdout.isTTY) return
  const width = 30
  const filled = Math.round((width * done) / tasks.length)
  const elapsed = (performance.now() - t0) / 1000
  const eta = done ? (elapsed / done) * (tasks.length - done) : 0
  process.stdout.write(`\r[${'#'.repeat(filled)}${'.'.repeat(width - filled)}] ${done}/${tasks.length} games  ${fmtTime(elapsed)} elapsed  eta ${fmtTime(eta)}  `)
}

const fmtTime = (s: number) => (s < 60 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, '0')}s`)

await new Promise<void>((resolve, reject) => {
  let active = 0
  const workers = Array.from({ length: Math.min(concurrency, tasks.length) }, () => new Worker(new URL('./worker.ts', import.meta.url)))
  const feed = (w: Worker) => {
    if (next >= tasks.length) {
      void w.terminate()
      if (--active === 0) resolve()
      return
    }
    w.postMessage(tasks[next++])
  }
  for (const w of workers) {
    active++
    w.on('error', reject)
    w.on('message', (g: GameRecord) => {
      appendFileSync(gamesPath, JSON.stringify(g) + '\n')
      const list = results.get(g.matchup)!
      list.push(g)
      done++
      if (list.length === perMatchup) {
        if (process.stdout.isTTY) process.stdout.write('\r\x1b[K')
        console.log(formatStatsLine(matchStats(g.matchup, g.matchup, list)))
      }
      progress()
      feed(w)
    })
    feed(w)
  }
})

if (process.stdout.isTTY) process.stdout.write('\r\x1b[K')
const report = writeReport(dir)
console.log(`\ndone in ${fmtTime((performance.now() - t0) / 1000)}`)
console.log(`games:   ${gamesPath}`)
console.log(`report:  ${report}`)
