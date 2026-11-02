import { DEFAULT_OPTIONS } from './engine/search'
import type { SearchOptions } from './engine/types'
import type { WebBot } from './neural/types'

export const SIMULATIONS = [8, 16, 32, 64, 128, 256, 400, 512]
export const DEFAULT_SIMULATIONS = 64
export const THINK_TIMES = [250, 500, 1000, 2000, 5000, 10000, 30000]
export const DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
export const NODE_LIMITS = [10_000, 50_000, 200_000, 1_000_000, 5_000_000, DEFAULT_OPTIONS.maxNodes]

/** Everything that decides how one side plays: the controller (as in the player menus) and its budgets. */
export interface SideConfig {
  controller: string
  simulations: number
  engine: SearchOptions
}

export const STRENGTHS = ['easy', 'medium', 'hard', 'expert'] as const
export type Strength = (typeof STRENGTHS)[number]
export const STRENGTH_NAMES: Record<Strength, string> = { easy: 'Easy', medium: 'Medium', hard: 'Hard', expert: 'Expert' }
export const isStrength = (value: string | null): value is Strength => STRENGTHS.includes(value as Strength)

// The ordering is by search budget and model size, not by measured strength; adjust this table, not the code.
// `engine` is the negamax fallback used when the preset's bot is missing from the catalogue.
const PRESETS: Record<Strength, { bot?: string; simulations?: number; engine: Partial<SearchOptions> }> = {
  easy: { engine: { maxDepth: 2, timeMs: 250 } },
  medium: { bot: 'resnet-small', simulations: 16, engine: { maxDepth: 4, timeMs: 500 } },
  hard: { bot: 'resnet-rl', simulations: 64, engine: { maxDepth: 8, timeMs: 1000 } },
  expert: { bot: 'resnet-rl', simulations: 256, engine: { maxDepth: 12, timeMs: 2000 } },
}

const ENGINE_KEYS = ['maxDepth', 'timeMs', 'maxNodes', 'moveGen', 'exactRoot', 'wallWeight', 'tempo'] as const
export const sameEngine = (a: SearchOptions, b: SearchOptions) => ENGINE_KEYS.every(key => a[key] === b[key])

export const humanSide = (): SideConfig => ({ controller: 'human', simulations: DEFAULT_SIMULATIONS, engine: DEFAULT_OPTIONS })

export function presetSide(strength: Strength, bots: WebBot[]): SideConfig {
  const preset = PRESETS[strength]
  const bot = bots.find(item => item.id === preset.bot)
  if (bot && preset.simulations) return { controller: `browser:${bot.id}`, simulations: preset.simulations, engine: DEFAULT_OPTIONS }
  return { controller: 'engine', simulations: DEFAULT_SIMULATIONS, engine: { ...DEFAULT_OPTIONS, ...preset.engine } }
}

/** The preset a side's settings correspond to, or null for custom settings. Budgets the controller ignores are ignored. */
export function matchPreset(side: SideConfig, bots: WebBot[]): Strength | null {
  return STRENGTHS.find(strength => {
    const preset = presetSide(strength, bots)
    if (preset.controller !== side.controller) return false
    return side.controller === 'engine' ? sameEngine(preset.engine, side.engine) : preset.simulations === side.simulations
  }) ?? null
}

export function controllerName(controller: string, bots: WebBot[]): string {
  if (controller === 'human') return 'Human'
  if (controller === 'engine') return 'Built-in negamax'
  if (controller.startsWith('browser:')) return bots.find(bot => `browser:${bot.id}` === controller)?.name ?? controller.slice(8)
  try {
    const [run, mode] = JSON.parse(controller) as [string, string]
    return run ? `${run} / ${mode}` : 'Reference negamax'
  } catch {
    return controller
  }
}

export function describeSide(side: SideConfig, bots: WebBot[]): string {
  if (side.controller === 'engine') return `Built-in negamax · depth ${side.engine.maxDepth} · ${side.engine.timeMs / 1000}s`
  return `${controllerName(side.controller, bots)} · ${side.simulations} simulations`
}
