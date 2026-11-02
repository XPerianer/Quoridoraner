import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Board } from '../engine/board'
import { NeuralClient } from './client'
import type { NeuralRequest, NeuralResponse, WebBot } from './types'

class FakeWorker {
  static instances: FakeWorker[] = []
  onmessage: ((event: MessageEvent<NeuralResponse>) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  postMessage = vi.fn<(request: NeuralRequest) => void>()
  terminate = vi.fn()
  constructor() { FakeWorker.instances.push(this) }
  get request(): NeuralRequest { return this.postMessage.mock.calls.at(-1)![0] }
  send(message: NeuralResponse): void { this.onmessage?.({ data: message } as MessageEvent<NeuralResponse>) }
}

const bot: WebBot = { id: 'resnet', name: 'ResNet', url: 'models/resnet.onnx', sha256: 'a'.repeat(64), bytes: 1000, planes: 19, simulations: 32 }
const positions = [Board.initial().toPosition()]
beforeEach(() => {
  FakeWorker.instances = []
  vi.stubGlobal('Worker', FakeWorker)
  vi.stubGlobal('location', { href: 'https://example.org/arena/' })
})
afterEach(() => vi.unstubAllGlobals())

describe('NeuralClient worker lifecycle', () => {
  it('cancels a pending request, rejects it, and replaces the terminated worker', async () => {
    const client = new NeuralClient()
    const pending = client.search(bot, positions, 32, vi.fn())
    const rejection = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    const oldWorker = FakeWorker.instances[0]
    client.cancel()
    await rejection
    expect(oldWorker.terminate).toHaveBeenCalledOnce()
    const next = client.search(bot, positions, 8, vi.fn())
    const worker = FakeWorker.instances[1]
    worker.send({ id: worker.request.id, type: 'done', move: 67, simulations: 8 })
    await expect(next).resolves.toEqual({ move: 67, simulations: 8 })
    client.dispose()
  })

  it('keeps the worker after success to reuse downloaded model sessions', async () => {
    const client = new NeuralClient()
    const status = vi.fn()
    const first = client.search(bot, positions, 32, status)
    const worker = FakeWorker.instances[0]
    const firstId = worker.request.id
    expect(worker.request.bot).toEqual(bot)
    expect(worker.request.positions).toEqual(positions)
    expect(worker.request.baseUrl).toMatch(/^https:\/\/example\.org\//)
    worker.send({ id: firstId, type: 'status', status: 'Downloading model…' })
    worker.send({ id: firstId, type: 'done', move: 67, simulations: 32 })
    await first
    expect(status).toHaveBeenCalledWith('Downloading model…')
    const second = client.search(bot, positions, 8, status)
    expect(FakeWorker.instances).toHaveLength(1)
    expect(worker.terminate).not.toHaveBeenCalled()
    worker.send({ id: worker.request.id, type: 'done', move: 75, simulations: 8 })
    await expect(second).resolves.toEqual({ move: 75, simulations: 8 })
    client.dispose()
    expect(worker.terminate).toHaveBeenCalledOnce()
  })

  it('ignores replies from an earlier request on the reused worker', async () => {
    const client = new NeuralClient()
    const first = client.search(bot, positions, 32, vi.fn())
    const worker = FakeWorker.instances[0]
    const firstId = worker.request.id
    worker.send({ id: firstId, type: 'done', move: 67, simulations: 32 })
    await first
    const status = vi.fn()
    const second = client.search(bot, positions, 8, status)
    worker.send({ id: firstId, type: 'status', status: 'Stale progress' })
    worker.send({ id: firstId, type: 'done', move: 0, simulations: 32 })
    expect(status).not.toHaveBeenCalled()
    worker.send({ id: worker.request.id, type: 'done', move: 75, simulations: 8 })
    await expect(second).resolves.toEqual({ move: 75, simulations: 8 })
    client.dispose()
  })

  it('ignores late callbacks from a terminated worker without losing the active cancellation', async () => {
    const client = new NeuralClient()
    const firstStatus = vi.fn()
    const first = client.search(bot, positions, 32, firstStatus)
    const oldWorker = FakeWorker.instances[0]
    const oldRequest = oldWorker.request.id
    const firstRejection = expect(first).rejects.toMatchObject({ name: 'AbortError' })
    const second = client.search(bot, positions, 8, vi.fn())
    await firstRejection
    const worker = FakeWorker.instances[1]
    oldWorker.send({ id: oldRequest, type: 'status', status: 'Old progress' })
    oldWorker.send({ id: oldRequest, type: 'done', move: 0, simulations: 32 })
    oldWorker.onerror?.({ message: 'Old worker failure' } as ErrorEvent)
    expect(firstStatus).not.toHaveBeenCalled()
    expect(worker.terminate).not.toHaveBeenCalled()
    const secondRejection = expect(second).rejects.toMatchObject({ name: 'AbortError' })
    client.cancel()
    await secondRejection
    expect(worker.terminate).toHaveBeenCalledOnce()
  })

  it('reports worker failures and permits a clean retry', async () => {
    const client = new NeuralClient()
    const first = client.search(bot, positions, 32, vi.fn())
    const rejection = expect(first).rejects.toThrow('GPU unavailable')
    const worker = FakeWorker.instances[0]
    worker.onerror?.({ message: 'GPU unavailable' } as ErrorEvent)
    await rejection
    expect(worker.terminate).toHaveBeenCalledOnce()
    const retry = client.search(bot, positions, 8, vi.fn())
    const next = FakeWorker.instances[1]
    next.send({ id: next.request.id, type: 'done', move: 67, simulations: 8 })
    await expect(retry).resolves.toEqual({ move: 67, simulations: 8 })
    client.dispose()
  })
})
