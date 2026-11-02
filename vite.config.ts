import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

// The ONNX Runtime .wasm files are loaded from jsDelivr (see src/neural/runtime.ts). onnxruntime-web
// still references its default build in dead code, so Vite would copy a 27 MB file that is never fetched.
const dropOrtWasm = (): Plugin => ({
  name: 'drop-ort-wasm',
  apply: 'build',
  generateBundle(_, bundle) {
    for (const name of Object.keys(bundle)) if (/ort-wasm-.*\.wasm$/.test(name)) delete bundle[name]
  },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), dropOrtWasm()],
  worker: { plugins: () => [dropOrtWasm()] },
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
})
