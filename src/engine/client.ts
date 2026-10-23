import type { Position, SearchOptions, SearchSnapshot, WorkerResponse } from './types'

/**
 * Runs searches in a dedicated worker. A running search is synchronous inside the worker,
 * so stopping it means terminating the worker and keeping the last reported snapshot.
 */
export class EngineClient {
  private worker: Worker | null = null
  private nextId = 0
  private last: SearchSnapshot | null = null
  private settle: ((s: SearchSnapshot | null) => void) | null = null

  search(position: Position, options: SearchOptions, onProgress: (s: SearchSnapshot) => void): Promise<SearchSnapshot | null> {
    this.cancel()
    const id = ++this.nextId
    this.last = null
    this.worker ??= this.spawn()
    return new Promise((resolve) => {
      this.settle = resolve
      this.worker!.onmessage = (e: MessageEvent<WorkerResponse>) => {
        if (e.data.id !== id) return
        this.last = e.data.snapshot
        onProgress(e.data.snapshot)
        if (e.data.type === 'done') this.finish(e.data.snapshot)
      }
      this.worker!.postMessage({ id, position, options })
    })
  }

  /** Stop now and resolve with the best result found so far. */
  stop(): void {
    if (!this.settle) return
    this.kill()
    this.finish(this.last && { ...this.last, done: true })
  }

  /** Stop now and discard the result. */
  cancel(): void {
    if (!this.settle) return
    this.kill()
    this.finish(null)
  }

  dispose(): void {
    this.cancel()
    this.kill()
  }

  private finish(s: SearchSnapshot | null): void {
    const settle = this.settle
    this.settle = null
    settle?.(s)
  }

  private kill(): void {
    this.worker?.terminate()
    this.worker = null
  }

  private spawn(): Worker {
    return new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  }
}
