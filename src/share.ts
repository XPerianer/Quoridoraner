import { Board } from './engine/board'
import { DEFAULT_OPTIONS } from './engine/search'
import type { Move, SearchOptions } from './engine/types'
import type { WebBot } from './neural/types'
import { DEFAULT_SIMULATIONS, DEPTHS, isStrength, matchPreset, NODE_LIMITS, presetSide, SIMULATIONS, THINK_TIMES, type SideConfig } from './presets'

// Link format (all parameters optional):
//   white, black   easy|medium|hard|expert, or a player menu value (human, engine, browser:<id>, dashboard run JSON)
//   sims           simulations for both sides (older links); wsims / bsims override it per side
//   wengine, bengine  non-default negamax settings, e.g. depth:4,time:500,gen:likely
//   moves          comma-separated move IDs, replayed before play continues
//   dashboard=1    kept so links made in the training dashboard open there

const MAX_PLIES = 200
const SIDES = [{ name: 'white', prefix: 'w', fallback: 'human' }, { name: 'black', prefix: 'b', fallback: 'engine' }] as const

type EngineField = { key: keyof SearchOptions; parse: (value: string) => SearchOptions[keyof SearchOptions] | undefined }
const fromList = (list: number[]) => (value: string) => list.includes(Number(value)) ? Number(value) : undefined
const evalPoints = (value: string) => /^-?\d{1,4}$/.test(value) ? Number(value) : undefined
const ENGINE_FIELDS: Record<string, EngineField> = {
  depth: { key: 'maxDepth', parse: fromList(DEPTHS) },
  time: { key: 'timeMs', parse: fromList(THINK_TIMES) },
  nodes: { key: 'maxNodes', parse: fromList(NODE_LIMITS) },
  gen: { key: 'moveGen', parse: value => value === 'all' || value === 'likely' ? value : undefined },
  wall: { key: 'wallWeight', parse: evalPoints },
  tempo: { key: 'tempo', parse: evalPoints },
  exact: { key: 'exactRoot', parse: value => value === '1' ? true : value === '0' ? false : undefined },
}

export function encodeEngine(options: SearchOptions): string {
  return Object.entries(ENGINE_FIELDS)
    .filter(([, field]) => options[field.key] !== DEFAULT_OPTIONS[field.key])
    .map(([name, field]) => `${name}:${typeof options[field.key] === 'boolean' ? Number(options[field.key]) : options[field.key]}`)
    .join(',')
}

/** Unknown keys and values outside the menu choices are ignored. */
export function decodeEngine(value: string | null): SearchOptions {
  const options: Record<string, unknown> = { ...DEFAULT_OPTIONS }
  for (const part of value?.split(',') ?? []) {
    const [name, raw = ''] = part.split(':')
    const field = Object.hasOwn(ENGINE_FIELDS, name) ? ENGINE_FIELDS[name] : undefined
    const parsed = field?.parse(raw)
    if (field && parsed !== undefined) options[field.key] = parsed
  }
  return options as unknown as SearchOptions
}

export function parseMoves(value: string | null): Move[] {
  const board = Board.initial()
  const moves = value?.split(',').filter(Boolean).map(Number) ?? []
  if (moves.length > MAX_PLIES) return []
  for (const move of moves) {
    if (board.winner() !== -1 || !Number.isInteger(move) || !board.legalMoves().includes(move)) return []
    board.make(move)
  }
  return moves
}

function parseController(value: string | null, bots: WebBot[], publicArena: boolean): string | null {
  if (value === 'human' || value === 'engine') return value
  if (value?.startsWith('browser:') && bots.some(bot => `browser:${bot.id}` === value)) return value
  if (!publicArena && value) {
    try {
      const parsed: unknown = JSON.parse(value)
      if (Array.isArray(parsed) && parsed.length === 2 && parsed.every(part => typeof part === 'string')) return value
    } catch { /* Invalid URL selections use the default player. */ }
  }
  return null
}

export interface SharedSetup {
  sides: [SideConfig, SideConfig]
  moves: Move[]
  /** Whether the link chose players or a game, so the start screen can be skipped. */
  fromLink: boolean
}

export function readShare(search: string, bots: WebBot[], publicArena: boolean): SharedSetup {
  const query = new URLSearchParams(search)
  const simulations = (value: string | null) => SIMULATIONS.includes(Number(value)) ? Number(value) : undefined
  const [white, black] = SIDES.map(({ name, prefix, fallback }): SideConfig => {
    const value = query.get(name)
    if (isStrength(value)) return presetSide(value, bots)
    const controller = parseController(value, bots, publicArena) ?? fallback
    const bot = bots.find(item => `browser:${item.id}` === controller)
    return {
      controller,
      simulations: simulations(query.get(`${prefix}sims`)) ?? simulations(query.get('sims')) ?? bot?.simulations ?? DEFAULT_SIMULATIONS,
      engine: decodeEngine(query.get(`${prefix}engine`)),
    }
  })
  return { sides: [white, black], moves: parseMoves(query.get('moves')), fromLink: ['white', 'black', 'moves'].some(key => query.has(key)) }
}

type Place = Pick<Location, 'origin' | 'pathname' | 'search'>

/** The page's own address without any shared setup. */
export function baseUrl(place: Place): string {
  return new URLSearchParams(place.search).get('dashboard') === '1' ? `${place.origin}${place.pathname}?dashboard=1` : `${place.origin}${place.pathname}`
}

export function shareUrl(place: Place, sides: [SideConfig, SideConfig], bots: WebBot[], moves: Move[] = []): string {
  const query = new URLSearchParams()
  if (new URLSearchParams(place.search).get('dashboard') === '1') query.set('dashboard', '1')
  SIDES.forEach(({ name, prefix }, p) => {
    const side = sides[p]
    const strength = side.controller === 'human' ? null : matchPreset(side, bots)
    if (strength) { query.set(name, strength); return }
    query.set(name, side.controller)
    if (side.controller === 'engine') {
      const engine = encodeEngine(side.engine)
      if (engine) query.set(`${prefix}engine`, engine)
    } else if (side.controller !== 'human') {
      query.set(`${prefix}sims`, String(side.simulations))
    }
  })
  if (moves.length) query.set('moves', moves.join(','))
  // Commas and colons are valid in a query; leaving them unescaped keeps links readable.
  return `${place.origin}${place.pathname}?${query.toString().replaceAll('%2C', ',').replaceAll('%3A', ':')}`
}
