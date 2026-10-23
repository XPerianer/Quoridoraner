import { TsSearcher } from './search'
import type { EngineBackend } from './types'

/** Swap point for a WebAssembly build: return any object implementing EngineBackend. */
export function createBackend(): EngineBackend {
  return new TsSearcher()
}
