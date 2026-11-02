import { loadBrowserModel, evaluateBrowserModel } from './runtime'
import { searchNeural } from './search'
import type { NeuralRequest, NeuralResponse } from './types'

const scope = self as unknown as { onmessage: ((event: MessageEvent<NeuralRequest>) => void) | null; postMessage: (message: NeuralResponse) => void }
scope.onmessage = event => {
  const { id, bot, positions, simulations, baseUrl } = event.data
  const status = (message: string) => scope.postMessage({ id, type: 'status', status: message })
  void (async () => {
    const model = await loadBrowserModel(bot, baseUrl, status)
    status(`Thinking with ${model.provider}…`)
    const result = await searchNeural({ positions, simulations, maxTimeMs: 15_000,
      evaluate: (position, repetitions, remaining) => evaluateBrowserModel(model.session, position, repetitions, remaining),
      onProgress: completed => { if (completed % 8 === 0) status(`${model.provider} · ${completed}/${simulations} simulations`) },
    })
    scope.postMessage({ id, type: 'done', ...result })
  })().catch(error => scope.postMessage({ id, type: 'error', error: error instanceof Error ? error.message : String(error) }))
}
