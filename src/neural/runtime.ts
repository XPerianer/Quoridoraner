import * as ort from 'onnxruntime-web/webgpu'
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'
import { encodeFeatures } from './features'
import type { Position } from '../engine/types'
import type { WebBot } from './types'

// The 28 MB runtime exceeds Cloudflare Pages' 25 MiB file limit, so it comes from jsDelivr,
// pinned to the bundled onnxruntime-web version that the local .mjs loader must match.
const wasmUrl = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/ort-wasm-simd-threaded.jsep.wasm`

ort.env.wasm.numThreads = 1
ort.env.wasm.proxy = false
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl }

interface LoadedModel { session: ort.InferenceSession; provider: string }
const sessions = new Map<string, LoadedModel>()

async function modelBytes(bot: WebBot, baseUrl: string, status: (message: string) => void): Promise<Uint8Array> {
  const url = new URL(bot.url, baseUrl).href
  let cache: Cache | undefined
  try { cache = await caches.open('quoridor-models-v1') } catch { /* Storage may be unavailable. */ }
  const cached = await cache?.match(url)
  let response = cached
  if (!response) {
    status(`Downloading ${bot.name} (${(bot.bytes / 1e6).toFixed(1)} MB)…`)
    response = await fetch(url)
  } else status(`Loading cached ${bot.name}…`)
  if (!response?.ok) throw new Error(`Model download failed (${response?.status ?? 'unavailable'})`)
  const reader = response.body?.getReader()
  const chunks: Uint8Array[] = []
  let size = 0
  if (reader) {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value); size += value.length
      if (!cached) status(`Downloading ${bot.name}: ${Math.min(100, Math.round(size / bot.bytes * 100))}%`)
    }
  } else { const bytes = new Uint8Array(await response.arrayBuffer()); chunks.push(bytes); size = bytes.length }
  const bytes = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length }
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  const hash = [...new Uint8Array(digest)].map(n => n.toString(16).padStart(2, '0')).join('')
  if (size !== bot.bytes || hash !== bot.sha256.toLowerCase()) {
    await cache?.delete(url)
    throw new Error('Model checksum failed. Retry to download a fresh copy.')
  }
  if (!cached) { try { await cache?.put(url, new Response(bytes)) } catch { /* Inference can continue without cache. */ } }
  return bytes
}

export async function loadBrowserModel(bot: WebBot, baseUrl: string, status: (message: string) => void, preferGpu = true): Promise<LoadedModel> {
  const key = `${bot.sha256}:${preferGpu}`
  const existing = sessions.get(key)
  if (existing) return existing
  const bytes = await modelBytes(bot, baseUrl, status)
  let session: ort.InferenceSession | undefined
  let provider = 'WebAssembly CPU'
  if (preferGpu && 'gpu' in navigator) {
    try {
      status('Preparing WebGPU…')
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' })
      // Compilation/operator errors can surface only on the first inference.
      const warmup = await session.run({ features: new ort.Tensor('float32', new Float32Array(19 * 81), [1, 19, 9, 9]) })
      Object.values(warmup).forEach(tensor => tensor.dispose())
      provider = 'WebGPU'
    } catch {
      await session?.release()
      session = undefined
      status('WebGPU unavailable; preparing CPU…')
    }
  }
  session ??= await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
  if (sessions.size >= 2) {
    const oldest = sessions.keys().next().value!
    await sessions.get(oldest)!.session.release()
    sessions.delete(oldest)
  }
  const loaded = { session, provider }
  sessions.set(key, loaded)
  return loaded
}

export async function evaluateBrowserModel(session: ort.InferenceSession, position: Position, repetitions: number, remaining: number) {
  const input = new ort.Tensor('float32', encodeFeatures(position, repetitions, remaining), [1, 19, 9, 9])
  try {
    const output = await session.run({ features: input })
    try { return { logits: Float32Array.from(output.logits.data as Float32Array), value: Number(output.value.data[0]) } }
    finally { Object.values(output).forEach(tensor => tensor.dispose()) }
  } finally { input.dispose() }
}
