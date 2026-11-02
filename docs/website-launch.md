# Public arena launch status

## Implemented

The default React entry point now opens a static public arena. It reuses the
existing board and play/watch controls, with independently selectable Human,
Built-in negamax, pretrained ResNet and self-play ResNet controllers. Local
training tools remain available separately at `?dashboard=1`.

Public neural play runs on the visitor's device and needs no Python service:

- `public/bots.json` lists curated, versioned model assets with sizes, checksums,
  architecture configuration and default search budgets.
- `training/export_web.py` exports evaluation-mode AZ ResNets to ONNX, including
  compact 10-plane models and dropped-plane masks. The browser contract is
  float32 `[1,19,9,9]` input, 209 policy logits and one side-to-move value.
- `src/neural/features.ts` ports native AZ feature construction and canonical
  actions. `src/neural/search.ts` implements legal-action PUCT with value sign,
  repetition and horizon handling. The older `training/mcts.ts` is not used.
- Browser workers keep neural inference and search off the rendering thread.
  Pending work is cancelled on control changes, and obsolete replies are ignored.
- ONNX Runtime Web tries WebGPU, including a warm-up inference, then falls back
  to WebAssembly CPU when initialization fails. The current WASM setting uses
  one thread, avoiding a cross-origin isolation requirement.
- Model loading reports progress, validates SHA-256 checksums, and uses browser
  Cache Storage when available. Up to two inference sessions are retained.
  The built-in engine remains available when model loading fails.

The shipped pretrained model is 2.1 MB and the self-play model is 12.8 MB before
HTTP compression. The ONNX WASM runtime is roughly 7 MB compressed, downloaded
separately from the selected weights. Host compression and cache policy matter
for the first visit. The names describe provenance, not measured strength.

## Validation and limits

Both shipped models passed native PyTorch versus ONNX Runtime CPU comparisons
on sixteen deterministic legal positions, including both players and walls.
Maximum absolute logit errors were approximately 0.000006 for the smaller model
and 0.000044 for the self-play model. Export regression tests cover both native
19-plane and compact 10-plane checkpoints. Native-reference browser fixtures and
source/model checksums are included alongside the model reports.

Chrome browser CPU inference passed all 16 fixtures for each model. Maximum
absolute logit errors were 0.0000061 and 0.000044; maximum value error was below
0.0000005. Warm fixture evaluation averaged roughly 3.2 ms and 16.8 ms respectively
on the development machine; these are inference timings, not whole-move budgets.
The GPU-preferred path fell back to CPU in headless Chrome, so WebGPU execution
itself remains unverified.

A browser match between the two neural bots passed pause, restart and resume
checks without API requests or browser errors. An unavailable model download
displayed an error and recovered by selecting the built-in engine. Worker tests
also cover cancellation, session-worker reuse and late callbacks.
That establishes end-to-end browser execution; it does not establish performance
on every device or full match strength parity with the native training engine.
Search accepts the selected simulation budget and checks a 15-second deadline
between simulations. Initialization/download time is additional, and one slow
inference can exceed the deadline.

## Production deployment

Run `npm run build`, inspect with `npm run preview`, then deploy all of `dist/`
to an HTTPS static host. Public play does not require `/api` routes. Keep the
training dashboard local; it still uses the Python service for experiment data
and experiment-model moves.

Serve app and model assets with correct MIME types and paths. Give content-hashed
model and build assets long cache lifetimes while allowing `bots.json` to refresh.
For a subpath deployment, use the appropriate Vite base when building. The model
weights are intentionally public downloads once deployed.

The 28 MB ONNX Runtime `.wasm` is not part of `dist/`: it is loaded from jsDelivr,
pinned to the bundled `onnxruntime-web` version, because static hosts such as
Cloudflare Pages reject files over 25 MiB. `npm run build` fails if any file in
`dist/` exceeds that limit. If jsDelivr is unreachable, neural bots report a load
error and the built-in engine still plays.

### Cloudflare Pages (quoridor.dominikmeier.io)

The site is hosted on Cloudflare Pages, separate from the GitHub Pages homepage
on the apex domain, so its traffic cannot affect the homepage. The free plan has
no bandwidth limit for static assets.

The Pages project `quoridoraner` uses direct upload, not a Git connection, so
pushing does not deploy. `npm run deploy` builds and uploads `dist/` as the
production deployment; wrangler must be logged in to the Cloudflare account. It
was created with `wrangler pages project create quoridoraner --production-branch
main --force`, where `--force` keeps it a Pages project instead of a Workers one
(Workers custom domains need the DNS zone on Cloudflare).

The custom domain `quoridor.dominikmeier.io` is attached to the project. DNS
stays at Namecheap, with a `CNAME` record for host `quoridor` pointing to
`quoridoraner.pages.dev`; Cloudflare issues the certificate once it resolves.

`public/_headers` gives `/assets/*` and the hashed `/models/*.onnx` a one-year
immutable cache; everything else keeps Cloudflare's revalidating default.

## Remaining release checks

- Benchmark cold loading, warm inference, complete move latency and peak memory
  at several simulation budgets on desktop and a representative phone. Pick
  final defaults from those measurements.
- Test the actual HTTPS deployment with no Python service running, including
  WebGPU initialization and CPU fallback on the intended browser/device matrix.
- Exercise human/bot and bot/bot games through completion, pause/resume, undo,
  restart, draw adjudication and cancellation during download and search.
- Test unavailable or corrupt model assets, storage restrictions, refresh after
  a catalog update, and recovery using retry or the built-in engine.
- Preserve export commands, checksums, parity reports and browser measurements
  for each subsequent model release. Evaluate bots before publishing strength
  rankings or promising fixed move times.

## Repeatable browser checks

After `npm run build`, run `npm run preview` in one terminal, then
`npx playwright install chromium` once and `npm run test:browser` in another.
An installed Chrome can be used instead by setting `CHROME_PATH` to its binary.
Set `WEB_URL` for another production URL. The smoke test checks neural matches,
pause/restart/resume, absence of API requests and failed-download recovery.

For output parity checks as well, start `npm run dev` and run
`WEB_DEV_URL=http://127.0.0.1:5173/ npm run test:browser`. This compares both models
against every native fixture on CPU and the GPU-preferred path, reporting which
provider actually ran. GPU fallback does not count as a successful GPU test.

## References

- [Model export workflow](../training/WEB_EXPORT.md)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [Browser execution-provider constraints](https://onnxruntime.ai/docs/tutorials/web/)
- [WebGPU integration](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
