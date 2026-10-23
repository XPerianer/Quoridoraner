import { readFileSync } from 'node:fs'
import { DEFAULT_OPTIONS } from '../src/engine/search'
import type { SearchOptions } from '../src/engine/types'

export type EngineParams = Omit<SearchOptions, 'reportEveryMs'>
export type ParamOverrides = Partial<EngineParams>

export interface EngineSpec {
  name: string
  params: ParamOverrides
}

export interface ExperimentConfig {
  name: string
  /** Each opening is played twice with colours swapped. */
  pairs: number
  /** Random plies played from the start position to diversify games. */
  openingPlies: number
  /** Probability that a random opening ply is a wall instead of a pawn move. */
  openingWallProb: number
  seed: number
  /** Game is adjudicated a draw after this many plies. */
  maxPlies: number
  /** Shared by every engine before its own overrides. */
  base: ParamOverrides
  baseline: EngineSpec
  candidates: EngineSpec[]
  /** Each combination of values becomes a candidate, e.g. { "wallWeight": [0, 5, 10] }. */
  sweep: Partial<Record<keyof EngineParams, (number | string | boolean)[]>>
}

export interface Matchup {
  candidate: EngineSpec
  /** Only the parameters this candidate changes, for plotting sweeps. */
  varied: ParamOverrides
}

const DEFAULTS: Omit<ExperimentConfig, 'name'> = {
  pairs: 50,
  openingPlies: 4,
  openingWallProb: 0.15,
  seed: 1,
  maxPlies: 200,
  base: { maxNodes: 50_000, timeMs: 1e9, maxDepth: 64 },
  baseline: { name: 'baseline', params: {} },
  candidates: [],
  sweep: {},
}

const PARAM_KEYS = new Set(Object.keys(DEFAULT_OPTIONS).filter((k) => k !== 'reportEveryMs'))

function checkParams(where: string, params: object): void {
  for (const k of Object.keys(params)) {
    if (!PARAM_KEYS.has(k)) throw new Error(`${where}: unknown engine parameter "${k}" (known: ${[...PARAM_KEYS].join(', ')})`)
  }
}

export function loadConfig(path: string): ExperimentConfig {
  const raw = JSON.parse(readFileSync(path, 'utf8')) as Partial<ExperimentConfig> & { name: string }
  if (!raw.name) throw new Error(`${path}: "name" is required`)
  const cfg: ExperimentConfig = { ...DEFAULTS, ...raw, base: { ...DEFAULTS.base, ...raw.base } }
  checkParams('base', cfg.base)
  checkParams('baseline', cfg.baseline.params)
  cfg.candidates.forEach((c) => checkParams(`candidate ${c.name}`, c.params))
  checkParams('sweep', cfg.sweep)
  if (cfg.candidates.length === 0 && Object.keys(cfg.sweep).length === 0) throw new Error(`${path}: needs "candidates" or "sweep"`)
  return cfg
}

export function resolveParams(base: ParamOverrides, spec: EngineSpec): EngineParams {
  const { reportEveryMs: _, ...defaults } = DEFAULT_OPTIONS
  return { ...defaults, ...base, ...spec.params }
}

export function matchups(cfg: ExperimentConfig): Matchup[] {
  const explicit = cfg.candidates.map((c) => ({ candidate: c, varied: c.params }))
  const keys = Object.keys(cfg.sweep) as (keyof EngineParams)[]
  if (keys.length === 0) return explicit
  let combos: ParamOverrides[] = [{}]
  for (const k of keys) combos = combos.flatMap((c) => cfg.sweep[k]!.map((v) => ({ ...c, [k]: v })))
  const swept = combos.map((varied) => ({
    candidate: {
      name: Object.entries(varied)
        .map(([k, v]) => `${k}=${v}`)
        .join(','),
      params: { ...cfg.baseline.params, ...varied },
    },
    varied,
  }))
  return [...explicit, ...swept]
}
