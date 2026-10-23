import { createBackend } from './backend'
import type { WorkerRequest, WorkerResponse } from './types'

const backend = createBackend()

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const { id, position, options } = e.data
  const post = (msg: WorkerResponse) => self.postMessage(msg)
  const snapshot = backend.search(position, options, (s) => post({ id, type: 'progress', snapshot: s }))
  post({ id, type: 'done', snapshot })
}
