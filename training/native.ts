import { createRequire } from 'node:module'
import type { Position } from '../src/engine/types'
import type { Rules } from './rules'

export interface NativeEngine {
  legal(position: Position): number[]
  step(position: Position, move: number, validate?: boolean): Position
  search(position: Position, nodes: number, maxDepth?: number): { move: number; score: number; nodes: number }
  inspect(position: Position): { legal: number[]; winner: number; distances: number[]; hash1: number; hash2: number }
}
export function loadNative(): NativeEngine {
  try {
    return createRequire(import.meta.url)('./native/build/engine.node') as NativeEngine
  } catch (error) {
    throw new Error('Native engine is unavailable. Run npm run train:native:build first.', { cause: error })
  }
}
export function nativeRules(engine: NativeEngine): Rules {
  return { legal: p => engine.legal(p), step: (p, m) => engine.step(p, m, false) }
}
