import { describe, expect, it } from 'vitest'
import { corpus, variants } from '../../bench/positions'
import { DEFAULT_OPTIONS, TsSearcher } from './search'

const positions = corpus()
describe('distance kernels', () => {
  for (const [name, factory] of Object.entries(variants)) {
    it(`${name}: distances, complete legality, and reversible state match reference`, () => {
      for (const position of positions) {
        const reference = variants.baseline(position), b = factory(position)
        const map = new Int8Array(81)
        for (const p of [0, 1] as const) {
          reference.distanceMap(p, map)
          expect(b.distance(p)).toBe(map[b.pawn[p]])
        }
        expect(b.legalMoves()).toEqual(reference.legalMoves())
        for (const move of reference.legalMoves()) {
          b.make(move); reference.make(move)
          for (const p of [0, 1] as const) expect(b.distance(p)).toBe(reference.distance(p))
          b.unmake(move); reference.unmake(move)
        }
        expect(b.toPosition()).toEqual(position)
        expect(b.checkHash()).toBe(true)
      }
      // Deliberately unreachable goal, including a non-goal isolated pawn.
      const blocked = factory({ pawns: [76, 4], wallsLeft: [0, 0], hWalls: [], vWalls: [], turn: 0 })
      blocked.blockDown.fill(511)
      expect(blocked.distance(0)).toBe(-1)
      expect(blocked.distance(1)).toBe(-1)
    })
    it(`${name}: identical complete search trees and scores`, () => {
      for (const position of positions.slice(0, 4)) {
        const options = { ...DEFAULT_OPTIONS, maxDepth: 3, timeMs: 1e9, reportEveryMs: 1e9 }
        const reference = new TsSearcher(variants.baseline).search(position, options, () => {})
        const actual = new TsSearcher(factory).search(position, options, () => {})
        expect([actual.bestMove, actual.score, actual.nodes, actual.pv]).toEqual([reference.bestMove, reference.score, reference.nodes, reference.pv])
      }
    })
  }
})
