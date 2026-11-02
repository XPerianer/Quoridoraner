import { describe, expect, it } from 'vitest'
import { Board } from './engine/board'
import { DEFAULT_OPTIONS } from './engine/search'
import type { WebBot } from './neural/types'
import { humanSide, matchPreset, presetSide, STRENGTHS, type SideConfig } from './presets'
import { decodeEngine, encodeEngine, readShare, shareUrl } from './share'

const bot = (id: string): WebBot => ({ id, name: id, url: `models/${id}.onnx`, sha256: 'a'.repeat(64), bytes: 1000, planes: 19, simulations: 32 })
const BOTS = [bot('resnet-small'), bot('resnet-rl')]
const place = (search = '') => ({ origin: 'https://example.org', pathname: '/arena/', search })
const query = (url: string) => new URL(url).search

describe('presets', () => {
  it('recognises every preset, with and without the neural bots', () => {
    for (const bots of [BOTS, []]) {
      const sides = STRENGTHS.map(strength => presetSide(strength, bots))
      expect(sides.map(side => matchPreset(side, bots))).toEqual(STRENGTHS)
      expect(new Set(sides.map(side => JSON.stringify(side))).size).toBe(STRENGTHS.length)
    }
    expect(presetSide('hard', []).controller).toBe('engine')
  })

  it('treats edited settings as custom', () => {
    const hard = presetSide('hard', BOTS)
    expect(matchPreset({ ...hard, simulations: 128 }, BOTS)).toBeNull()
    expect(matchPreset({ ...presetSide('easy', BOTS), engine: { ...presetSide('easy', BOTS).engine, tempo: 5 } }, BOTS)).toBeNull()
  })
})

describe('share links', () => {
  it('round-trips presets and the moves played', () => {
    const board = Board.initial()
    const moves = Array.from({ length: 4 }, () => { const move = board.legalMoves()[0]; board.make(move); return move })
    const sides: [SideConfig, SideConfig] = [humanSide(), presetSide('hard', BOTS)]
    const url = shareUrl(place(), sides, BOTS, moves)
    expect(url).toBe(`https://example.org/arena/?white=human&black=hard&moves=${moves.join(',')}`)
    const shared = readShare(query(url), BOTS, true)
    expect(shared.sides).toEqual(sides)
    expect(shared.moves).toEqual(moves)
    expect(shared.fromLink).toBe(true)
  })

  it('round-trips custom bot and negamax settings', () => {
    const engine = { ...DEFAULT_OPTIONS, maxDepth: 4, timeMs: 500, moveGen: 'likely' as const, exactRoot: true, tempo: -3 }
    const sides: [SideConfig, SideConfig] = [{ controller: 'engine', simulations: 64, engine }, { controller: 'browser:resnet-rl', simulations: 400, engine: DEFAULT_OPTIONS }]
    const url = shareUrl(place(), sides, BOTS)
    expect(url).toContain('wengine=depth:4,time:500,gen:likely,tempo:-3,exact:1')
    expect(url).toContain('bsims=400')
    expect(url).not.toContain('moves=')
    expect(readShare(query(url), BOTS, true).sides).toEqual(sides)
  })

  it('ignores invalid negamax settings and unknown bots', () => {
    expect(decodeEngine('depth:7,gen:bogus,time:250,toString:1')).toEqual({ ...DEFAULT_OPTIONS, timeMs: 250 })
    expect(encodeEngine(DEFAULT_OPTIONS)).toBe('')
    const shared = readShare('?white=browser:gone&black=expert&sims=8', [], true)
    expect(shared.sides[0].controller).toBe('human')
    expect(shared.sides[1]).toEqual(presetSide('expert', []))
  })

  it('applies the older shared sims parameter to both sides', () => {
    const shared = readShare('?white=browser:resnet-small&black=browser:resnet-rl&sims=8&bsims=16', BOTS, true)
    expect(shared.sides.map(side => side.simulations)).toEqual([8, 16])
  })

  it('keeps dashboard links in the dashboard and their run players', () => {
    const run = JSON.stringify(['run-1', 'selfplay'])
    const url = shareUrl(place('?dashboard=1&white=human'), [humanSide(), { controller: run, simulations: 128, engine: DEFAULT_OPTIONS }], [])
    expect(url.startsWith('https://example.org/arena/?dashboard=1&')).toBe(true)
    expect(readShare(query(url), [], false).sides[1].controller).toBe(run)
    expect(readShare(query(url), [], true).sides[1].controller).toBe('engine')
  })
})
