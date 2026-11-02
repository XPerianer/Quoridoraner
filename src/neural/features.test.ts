import { describe, expect, it } from 'vitest'
import { Board } from '../engine/board'
import { canonicalMove, encodeFeatures } from './features'

// SHA256 over float32 bytes from Engine.features in training/az.py, with
// repetitions=2, remaining=.375 and mirror=false. These cover both perspectives,
// both wall orientations and nontrivial shortest-path geometry.
const nativeFixtures: [number[], string, string][] = [
  [[], 'e42a4b57064a8da10aa98133753ec41e498e5af3390b3c7ca97792491de45e77', '203c06f6be7b4e3a4992d57a0d3b8b3b5bb0a42305281c6f1bd412e99d40edc1'],
  [[67], 'fbb1cf974613a77f302fe4ed342815aae8631106132b0e0b1d3a31ccd7754985', 'd428a038a21bed02579b59bca5a98e2444a44684b425b1e0454198983c4231bc'],
  [[81], '7d3d11baa6b1f77a8032668f8ffa5e1b12c4b172961c11eb45384c533e22b420', '29b535b7707d2d8490779d6f75c561a6f2729d054768ab8a9ae86f1ff3ec6ead'],
  [[81, 164], '606d3ff163278b5517c1ed56bcef026ff7e18710ffa05e2bc87219c2a9399d04', 'ec1e2dd7e7631441a631589f540edee7f52e30969fe44255395795ccf48b13ae'],
  [[67, 13, 58, 22], '3396405249b307ba06c3afc6000f5d2ed3e4a1c0401d2681926242e92a2f7a3a', '0fb54c74643ec12397584807d4f24087e73e06fd37a1fd0890de2ad8a0d956af'],
  [[117, 170, 67], '4f56421a7d1fa54c5e918bed6680c2d05f7edc4b3d5cc822b8dbb27243f1614f', 'aec04f9b59834503ead711fb14575f1e56100ac303a85d8822ca836d3d70d933'],
]
describe('AZ browser features', () => {
  for (const [moves, fullHash, compactHash] of nativeFixtures) {
    it(`matches native float32 features after ${JSON.stringify(moves)}`, async () => {
      const board = Board.initial()
      moves.forEach(move => board.make(move))
      for (const [planes, hash] of [[19, fullHash], [10, compactHash]] as const) {
        const result = encodeFeatures(board.toPosition(), 2, .375, planes)
        const digest = await crypto.subtle.digest('SHA-256', result.buffer as ArrayBuffer)
        expect([...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')).toBe(hash)
      }
    })
  }
  it('maps all actions involutively including wall anchor rows', () => {
    for (let move = 0; move < 209; move++) {
      expect(canonicalMove(move, 0)).toBe(move)
      expect(canonicalMove(canonicalMove(move, 1), 1)).toBe(move)
    }
    expect(canonicalMove(76, 1)).toBe(4)
    expect(canonicalMove(81, 1)).toBe(137)
    expect(canonicalMove(145, 1)).toBe(201)
  })
  it('rejects unsupported model inputs', () => {
    expect(() => encodeFeatures(Board.initial().toPosition(), 1, 1, 12)).toThrow('Unsupported')
  })
})
