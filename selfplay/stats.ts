import { cumulativeStdNormalProbability, mean as ssMean, sampleVariance } from 'simple-statistics'
import type { GameRecord } from './game'

export interface MatchStats {
  matchup: string
  candidate: string
  games: number
  pairs: number
  wins: number
  draws: number
  losses: number
  /** Candidate's mean score per game in [0, 1]. */
  score: number
  elo: number
  eloLow: number
  eloHigh: number
  /** Likelihood of superiority: P(candidate is stronger). */
  los: number
  whiteScore: number
  meanPlies: number
  candidateDepth: number
  baselineDepth: number
  candidateNodes: number
  baselineNodes: number
  candidateMsPerMove: number
  baselineMsPerMove: number
}

const Z95 = 1.959964

/** ±Infinity when the score (or a CI bound) leaves (0, 1), i.e. the data cannot bound the Elo difference. */
export const eloFromScore = (s: number) => (s >= 1 ? Infinity : s <= 0 ? -Infinity : -400 * Math.log10(1 / s - 1))

const mean = (xs: number[]) => (xs.length ? ssMean(xs) : 0)

/** Scores the candidate (the engine named `candidate`) against the other engine, using colour-swapped pairs as the unit of variance. */
export function matchStats(matchup: string, candidate: string, games: GameRecord[]): MatchStats {
  const result = (g: GameRecord) => {
    if (g.winner === -1) return 0.5
    return g.names[g.winner] === candidate ? 1 : 0
  }
  const byPair = new Map<number, number[]>()
  for (const g of games) byPair.set(g.pair, [...(byPair.get(g.pair) ?? []), result(g)])
  const pairScores = [...byPair.values()].map(mean)
  const s = mean(pairScores)
  const variance = pairScores.length > 1 ? sampleVariance(pairScores) : 0.25
  const se = Math.sqrt(variance / Math.max(1, pairScores.length))

  const side = (g: GameRecord, mine: boolean) => {
    const idx = (g.names[0] === candidate) === mine ? 0 : 1
    const offset = g.opening.length % 2
    return g.moves.filter((_, i) => (i + offset) % 2 === idx)
  }
  const perMove = (mine: boolean, f: (m: GameRecord['moves'][number]) => number) => mean(games.flatMap((g) => side(g, mine).map(f)))

  return {
    matchup,
    candidate,
    games: games.length,
    pairs: byPair.size,
    wins: games.filter((g) => result(g) === 1).length,
    draws: games.filter((g) => g.winner === -1).length,
    losses: games.filter((g) => result(g) === 0).length,
    score: s,
    elo: eloFromScore(s),
    eloLow: eloFromScore(s - Z95 * se),
    eloHigh: eloFromScore(s + Z95 * se),
    los: se > 0 ? cumulativeStdNormalProbability((s - 0.5) / se) : s > 0.5 ? 1 : s < 0.5 ? 0 : 0.5,
    whiteScore: mean(games.map((g) => (g.winner === -1 ? 0.5 : g.winner === 0 ? 1 : 0))),
    meanPlies: mean(games.map((g) => g.plies)),
    candidateDepth: perMove(true, (m) => m.depth),
    baselineDepth: perMove(false, (m) => m.depth),
    candidateNodes: perMove(true, (m) => m.nodes),
    baselineNodes: perMove(false, (m) => m.nodes),
    candidateMsPerMove: perMove(true, (m) => m.ms),
    baselineMsPerMove: perMove(false, (m) => m.ms),
  }
}
