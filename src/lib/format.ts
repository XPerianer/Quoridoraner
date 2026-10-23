import { isWinScore, STEP, WIN } from '../engine/search'
import type { Bound } from '../engine/types'

export function formatScore(s: number): string {
  if (isWinScore(s)) return `${s > 0 ? '+' : '−'}#${Math.ceil((WIN - Math.abs(s)) / 2)}`
  const v = s / STEP
  return `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(1)}`
}

export const BOUND_PREFIX: Record<Bound, string> = { exact: '', upper: '≤', lower: '≥' }

export const formatBounded = (s: number, b: Bound) => BOUND_PREFIX[b] + formatScore(s)

/** Maps a score to [-1, 1] for color: four path steps of advantage saturate. */
export const scoreToUnit = (s: number) => (isWinScore(s) ? Math.sign(s) : Math.max(-1, Math.min(1, s / (4 * STEP))))

export function compact(n: number): string {
  if (n < 1000) return String(Math.round(n))
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e4 ? 1 : 0)}k`
  return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`
}
