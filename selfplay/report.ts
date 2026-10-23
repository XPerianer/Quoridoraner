import { readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { moveToString } from '../src/engine/board'
import type { EngineParams, ExperimentConfig, ParamOverrides } from './config'
import type { GameRecord } from './game'
import { matchStats, type MatchStats } from './stats'

interface RunMeta {
  config: ExperimentConfig
  startedAt: string
  commit: string
  dirty: boolean
  concurrency: number
  baseline: { name: string; params: EngineParams }
  matchups: { name: string; varied: ParamOverrides; params: EngineParams }[]
}

const signed = (x: number) => `${x >= 0 ? '+' : '−'}${Number.isFinite(x) ? Math.abs(x).toFixed(0) : '∞'}`
const ELO_AXIS = 800
/** Charts cannot draw infinite interval ends; they are pinned to the axis edge and flagged in the tooltip. */
const clampElo = (x: number) => Math.max(-ELO_AXIS, Math.min(ELO_AXIS, x))

export function formatStatsLine(s: MatchStats): string {
  return (
    `${s.candidate.padEnd(28)} ${String(s.games).padStart(4)} games  +${s.wins} =${s.draws} -${s.losses}  ` +
    `score ${(100 * s.score).toFixed(1)}%  Elo ${signed(s.elo)} [${signed(s.eloLow)}, ${signed(s.eloHigh)}]  LOS ${(100 * s.los).toFixed(0)}%  ` +
    `depth ${s.candidateDepth.toFixed(1)} vs ${s.baselineDepth.toFixed(1)}`
  )
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const json = (x: unknown) => JSON.stringify(x).replace(/</g, '\\u003c')

export function writeReport(dir: string): string {
  const meta = JSON.parse(readFileSync(join(dir, 'config.json'), 'utf8')) as RunMeta
  const games = readFileSync(join(dir, 'games.jsonl'), 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l) as GameRecord)
  const stats = meta.matchups
    .map((m) => ({ meta: m, games: games.filter((g) => g.matchup === m.name) }))
    .filter((m) => m.games.length > 0)
    .map((m) => ({ ...matchStats(m.meta.name, m.meta.name, m.games), varied: m.meta.varied }))
  writeFileSync(join(dir, 'summary.json'), JSON.stringify(stats, null, 2))

  const cfg = meta.config
  const expected = meta.matchups.length * cfg.pairs * 2
  const sorted = [...stats].sort((a, b) => b.elo - a.elo)
  const sweepKeys = Object.keys(cfg.sweep)
  const sweepKey = sweepKeys.length === 1 && cfg.sweep[sweepKeys[0] as keyof EngineParams]!.every((v) => typeof v === 'number') ? sweepKeys[0] : null
  const sweepRows = sweepKey
    ? stats
        .filter((s) => sweepKey in s.varied)
        .map((s) => ({ value: (s.varied as Record<string, number>)[sweepKey], elo: clampElo(s.elo), lo: clampElo(s.eloLow), hi: clampElo(s.eloHigh), ci: `[${signed(s.eloLow)}, ${signed(s.eloHigh)}]`, games: s.games }))
    : []

  const eloSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    width: 'container',
    height: Math.max(80, 28 * stats.length),
    data: {
      values: sorted.map((s) => ({ name: s.candidate, elo: clampElo(s.elo), lo: clampElo(s.eloLow), hi: clampElo(s.eloHigh), ci: `[${signed(s.eloLow)}, ${signed(s.eloHigh)}]`, games: s.games, los: s.los })),
    },
    encoding: { y: { field: 'name', type: 'nominal', sort: null, title: null } },
    layer: [
      { mark: { type: 'rule', color: '#c3c2b7' }, encoding: { x: { datum: 0 } } },
      {
        mark: { type: 'rule', strokeWidth: 2, color: '#2a78d6' },
        encoding: { x: { field: 'lo', type: 'quantitative', title: `Elo vs ${meta.baseline.name} (95% CI)` }, x2: { field: 'hi' } },
      },
      {
        mark: { type: 'point', filled: true, size: 80, color: '#2a78d6' },
        encoding: {
          x: { field: 'elo', type: 'quantitative' },
          tooltip: [
            { field: 'name' },
            { field: 'elo', format: '+.0f' },
            { field: 'ci', title: '95% CI' },
            { field: 'los', format: '.0%', title: 'LOS' },
            { field: 'games' },
          ],
        },
      },
    ],
  }

  const sweepSpec = sweepKey && {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    width: 'container',
    height: 260,
    data: { values: sweepRows },
    encoding: { x: { field: 'value', type: 'quantitative', title: sweepKey } },
    layer: [
      { mark: { type: 'rule', color: '#c3c2b7' }, encoding: { y: { datum: 0 } } },
      { mark: { type: 'area', opacity: 0.18, color: '#2a78d6' }, encoding: { y: { field: 'lo', type: 'quantitative', title: 'Elo vs baseline' }, y2: { field: 'hi' } } },
      { mark: { type: 'line', strokeWidth: 2, color: '#2a78d6', point: { size: 64, filled: true } }, encoding: { y: { field: 'elo', type: 'quantitative' }, tooltip: [{ field: 'value', title: sweepKey }, { field: 'elo', format: '+.0f' }, { field: 'ci', title: '95% CI' }, { field: 'games' }] } },
    ],
  }

  const depthSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    width: 'container',
    height: Math.max(80, 36 * stats.length),
    data: {
      values: sorted.flatMap((s) => [
        { name: s.candidate, engine: 'candidate', depth: s.candidateDepth, ms: s.candidateMsPerMove, nodes: s.candidateNodes },
        { name: s.candidate, engine: meta.baseline.name, depth: s.baselineDepth, ms: s.baselineMsPerMove, nodes: s.baselineNodes },
      ]),
    },
    mark: { type: 'bar', cornerRadiusEnd: 4, height: { band: 0.8 } },
    encoding: {
      y: { field: 'name', type: 'nominal', sort: null, title: null },
      yOffset: { field: 'engine' },
      x: { field: 'depth', type: 'quantitative', title: 'Mean completed search depth per move' },
      color: { field: 'engine', type: 'nominal', scale: { range: ['#2a78d6', '#eb6834'] }, title: null, legend: { orient: 'top' } },
      tooltip: [{ field: 'name' }, { field: 'engine' }, { field: 'depth', format: '.2f' }, { field: 'nodes', format: ',.0f', title: 'nodes/move' }, { field: 'ms', format: '.0f', title: 'ms/move' }],
    },
  }

  const paramRows = (p: ParamOverrides) =>
    Object.entries(p)
      .map(([k, v]) => `<code>${esc(k)}=${esc(String(v))}</code>`)
      .join(' ') || '<span class="has-text-grey">none</span>'

  const resultText = (g: GameRecord) => (g.winner === -1 ? '½–½' : g.winner === 0 ? '1–0' : '0–1')
  const gameRows = games
    .map(
      (g) =>
        `<tr><td>${g.id}</td><td>${esc(g.matchup)}</td><td>${g.pair}</td><td>${esc(g.names[0])}</td><td>${esc(g.names[1])}</td><td>${resultText(g)}</td><td>${g.termination}</td><td>${g.plies}</td><td class="is-family-monospace is-size-7">${g.opening.map(moveToString).join(' ')} | ${g.moves.map((m) => moveToString(m.move)).join(' ')}</td></tr>`,
    )
    .join('\n')

  const html = `<!doctype html>
<html data-theme="light" lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Self-play: ${esc(cfg.name)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css" />
<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.21.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0"></script>
<style>.chart { width: 100%; } td, th { font-variant-numeric: tabular-nums; } .table-container { max-height: 70vh; overflow: auto; }</style>
</head>
<body>
<section class="section">
<div class="container">
  <h1 class="title">Self-play: ${esc(cfg.name)}</h1>
  <p class="subtitle is-6">
    ${esc(meta.startedAt)} · commit <code>${esc(meta.commit)}</code>${meta.dirty ? ' <span class="tag is-warning">uncommitted changes</span>' : ''}
    · ${games.length}${games.length < expected ? ` of ${expected}` : ''} games${games.length < expected ? ' <span class="tag is-danger">incomplete run</span>' : ''}
  </p>
  <div class="box">
    <div class="columns is-multiline is-size-7">
      <div class="column is-half"><b>Baseline</b> (${esc(meta.baseline.name)}): ${paramRows(meta.baseline.params)}</div>
      <div class="column is-half"><b>Experiment</b>: ${cfg.pairs} opening pairs per candidate (colours swapped), ${cfg.openingPlies} random opening plies (wall prob ${cfg.openingWallProb}), seed ${cfg.seed}, draw after ${cfg.maxPlies} plies or threefold repetition</div>
      <div class="column is-full"><b>Shared overrides</b>: ${paramRows(cfg.base)}${sweepKeys.length ? ` · <b>sweep</b>: ${sweepKeys.map((k) => `<code>${esc(k)}</code> ∈ {${cfg.sweep[k as keyof EngineParams]!.join(', ')}}`).join(', ')}` : ''}</div>
    </div>
  </div>

  <h2 class="title is-5">Results vs ${esc(meta.baseline.name)}</h2>
  <div class="table-container">
  <table class="table is-fullwidth is-striped is-narrow is-size-7">
    <thead><tr><th>candidate</th><th>changes</th><th>games</th><th>W / D / L</th><th>score</th><th>Elo [95% CI]</th><th>LOS</th><th>depth cand / base</th><th>nodes/move cand / base</th><th>ms/move cand / base</th><th>mean plies</th><th>White score</th></tr></thead>
    <tbody>
    ${sorted
      .map(
        (s) =>
          `<tr><td><b>${esc(s.candidate)}</b></td><td>${paramRows(s.varied)}</td><td>${s.games}</td><td>${s.wins} / ${s.draws} / ${s.losses}</td><td>${(100 * s.score).toFixed(1)}%</td><td>${signed(s.elo)} [${signed(s.eloLow)}, ${signed(s.eloHigh)}]</td><td>${(100 * s.los).toFixed(0)}%</td><td>${s.candidateDepth.toFixed(1)} / ${s.baselineDepth.toFixed(1)}</td><td>${s.candidateNodes.toFixed(0)} / ${s.baselineNodes.toFixed(0)}</td><td>${s.candidateMsPerMove.toFixed(0)} / ${s.baselineMsPerMove.toFixed(0)}</td><td>${s.meanPlies.toFixed(1)}</td><td>${(100 * s.whiteScore).toFixed(0)}%</td></tr>`,
      )
      .join('\n')}
    </tbody>
  </table>
  </div>
  <p class="is-size-7 has-text-grey mb-5">Elo and its confidence interval are computed from colour-swapped game pairs (n = pairs per candidate). LOS = likelihood that the candidate is stronger than the baseline.</p>

  <div class="columns">
    <div class="column"><h3 class="title is-6">Elo difference</h3><div id="elo" class="chart"></div></div>
    <div class="column"><h3 class="title is-6">Search depth reached</h3><div id="depth" class="chart"></div></div>
  </div>
  ${sweepKey ? `<h3 class="title is-6">Sweep over <code>${esc(sweepKey)}</code></h3><div id="sweep" class="chart"></div>` : `<p class="has-text-grey is-size-7">Sweep plot: shown only for a sweep over a single numeric parameter${sweepKeys.length ? ` (this run sweeps ${sweepKeys.length} parameters)` : ' (this run has no sweep)'}.</p>`}

  <details class="mt-5"><summary>All ${games.length} games (table: players, result, opening | engine moves)</summary>
  <div class="table-container"><table class="table is-narrow is-size-7 is-fullwidth">
  <thead><tr><th>id</th><th>matchup</th><th>pair</th><th>White</th><th>Black</th><th>result</th><th>end</th><th>plies</th><th>moves</th></tr></thead>
  <tbody>${gameRows}</tbody></table></div>
  </details>
  <details class="mt-3"><summary>Resolved configuration (JSON, ${meta.matchups.length + 1} engines)</summary>
  <pre class="is-size-7">${esc(JSON.stringify(meta, null, 2))}</pre>
  </details>
</div>
</section>
<script>
const opts = { actions: false, renderer: 'svg' };
vegaEmbed('#elo', ${json(eloSpec)}, opts);
vegaEmbed('#depth', ${json(depthSpec)}, opts);
${sweepSpec ? `vegaEmbed('#sweep', ${json(sweepSpec)}, opts);` : ''}
</script>
</body>
</html>
`
  const path = join(dir, 'report.html')
  writeFileSync(path, html)
  return path
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 3) {
    console.error('usage: npm run selfplay:report -- <results-dir>')
    process.exit(1)
  }
  const dir = resolve(process.argv[2])
  const path = writeReport(dir)
  const summary = JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8')) as MatchStats[]
  summary.forEach((s) => console.log(formatStatsLine(s)))
  console.log(`report: ${path}`)
}
