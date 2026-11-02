// Fails the build when a file in dist/ is too large for Cloudflare Pages (25 MiB per file).
import { readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const LIMIT = 25 * 1024 * 1024
const root = new URL('../dist/', import.meta.url).pathname
const files = readdirSync(root, { recursive: true }).map(name => join(root, name)).filter(path => statSync(path).isFile())
const sizes = files.map(path => ({ path: relative(root, path), bytes: statSync(path).size })).sort((a, b) => b.bytes - a.bytes)
const mb = bytes => `${(bytes / 1024 / 1024).toFixed(1)} MiB`

const tooLarge = sizes.filter(file => file.bytes > LIMIT)
for (const file of tooLarge) console.error(`dist/${file.path} is ${mb(file.bytes)}, over the 25 MiB Cloudflare Pages limit`)
if (tooLarge.length) process.exit(1)
const total = sizes.reduce((sum, file) => sum + file.bytes, 0)
console.log(`dist/: ${files.length} files, ${mb(total)}; largest dist/${sizes[0].path} (${mb(sizes[0].bytes)})`)
