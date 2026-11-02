import type { Position } from '../engine/types'
import type { NeuralRequest, NeuralResponse, WebBot } from './types'

/** One persistent worker shares loaded sessions between both players. */
export class NeuralClient {
  private worker: Worker | null = null
  private nextId = 0
  private reject: ((error: Error) => void) | null = null

  search(bot: WebBot, positions: Position[], simulations: number, onStatus: (status: string) => void): Promise<{ move: number; simulations: number }> {
    this.cancel()
    const id = ++this.nextId
    this.worker ??= new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    const worker = this.worker
    return new Promise((resolve, reject) => {
      this.reject = reject
      this.worker!.onerror = event => {
        if (this.worker !== worker || id !== this.nextId || !this.reject) return
        this.reject = null
        this.worker?.terminate()
        this.worker = null
        reject(new Error(event.message || 'Browser bot worker failed'))
      }
      this.worker!.onmessage = (event: MessageEvent<NeuralResponse>) => {
        if (this.worker !== worker || id !== this.nextId || !this.reject) return
        const message = event.data
        if (message.id !== id) return
        if (message.type === 'status') onStatus(message.status)
        else {
          this.reject = null
          if (message.type === 'error') reject(new Error(message.error))
          else resolve({ move: message.move, simulations: message.simulations })
        }
      }
      this.worker!.postMessage({ id, bot, positions, simulations, baseUrl: new URL(import.meta.env.BASE_URL, location.href).href } satisfies NeuralRequest)
    })
  }

  cancel(): void {
    if (!this.reject) return
    this.worker?.terminate()
    this.worker = null
    const reject = this.reject
    this.reject = null
    reject(new DOMException('Search cancelled', 'AbortError'))
  }

  dispose(): void {
    this.cancel()
    this.worker?.terminate()
    this.worker = null
  }
}
