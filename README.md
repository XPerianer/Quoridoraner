# Quoridoraner

Play Quoridor against bots or watch them play each other. The public arena runs
the built-in negamax engine and trained ResNet bots on the visitor's own device,
with no Python service required. Negamax includes a live view of its search space.
Written in TypeScript (Vite + React). The original wxWidgets/C++ version lives in [`legacy-cpp/`](legacy-cpp/).

## Running

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # engine unit tests (vitest)
npm run bench    # search benchmark in Node: npm run bench -- all 5
npm run build    # static site in dist/
npm run selfplay -- selfplay/experiments/wall-weight.json   # engine-vs-engine tuning, see selfplay/README.md
```

The default page is the public arena. Choose to play against a bot or watch two
bots, pick a strength preset, and start. Advanced bot settings and the analysis
engine are in collapsible sections. The share button copies a link to the current
game or to the bot settings.

## Browser neural bots and deployment

The arena includes a pretrained ResNet (2.1 MB) and a self-play ResNet (12.8 MB).
ONNX Runtime Web runs their inference in a worker, using WebGPU when available
and falling back to WebAssembly CPU. The AZ feature encoder and PUCT search also
run in the worker. Models download only when selected, are checked against their
SHA-256 hashes, and are cached locally when browser storage is available. The
runtime retains up to two loaded model sessions.

Neural simulations set the search budget; a 15-second search limit is checked
between simulations. Initial runtime/model loading takes additional time. The
ONNX runtime WASM asset (roughly 7 MB compressed) loads from jsDelivr, separate
from the model downloads. Device speed affects move times.
The bot labels describe training provenance, not a measured strength ranking.

```sh
npm run build
npm run preview  # review the production build locally
```

Deploy the complete `dist/` directory to a static HTTPS host, including its
`assets/`, `models/` and `bots.json` files. The public site is on Cloudflare Pages
at quoridor.dominikmeier.io. No Python inference server or `/api`
routes are needed for public play. HTTPS enables browser GPU and cryptographic
APIs; localhost also works for development. The current single-threaded WASM
configuration does not require cross-origin isolation headers. For hosting under
a subpath, build with the corresponding Vite `--base` option.

See [model export instructions](training/WEB_EXPORT.md) for adding trusted
checkpoints and [launch status and remaining checks](docs/website-launch.md).

## Train neural bots

An AlphaZero-style policy/value network and MCTS trainer learns two independent
bots: one through self-play, one against the fixed negamax engine. It includes a
live terminal dashboard, timed resumable checkpoints, batched inference, and CLI play/evaluation.

```sh
.venv/bin/python -m pip install -r training/requirements.txt
.venv/bin/python training/train.py --out experiments/training/quick
# Ctrl-C saves and pauses; continue with:
.venv/bin/python training/train.py --resume experiments/training/quick
```

See [the training guide](training/README.md) for setup, settings, and how to play
against exported bots. The default experiment is intentionally small; measured
strength improvement needs substantially more training and evaluation games.

## Playing

- Click a highlighted square to move. Jumps over the opponent, and diagonal jumps when a wall is behind them, follow the standard rules.
- To place a wall, click a gap between squares. The wall is centred on the pointer, and a preview shows whether it is legal.
- White (bottom) moves first. Each side can be a human or the engine, so engine vs engine works too.

## Engine (`src/engine/`)

| File | Contents |
| --- | --- |
| `types.ts` | Plain-data contract (positions, options, search snapshots) shared by UI and backends |
| `board.ts` | Board state as typed arrays and per-row bitmasks: move generation, wall legality, bit-parallel flood-fill distance, Zobrist hashing, make/unmake |
| `search.ts` | Iterative-deepening negamax with alpha-beta, a transposition table, history/distance move ordering and search telemetry |
| `backend.ts` | Chooses the `EngineBackend` implementation |
| `worker.ts`, `client.ts` | Run the search in a Web Worker and stream snapshots back to the UI |

The evaluation is the difference in shortest path to goal, as in the C++ version, plus a bonus for walls in hand (`wallWeight`, default 10 = one path step, tuned with the self-play harness). Every engine parameter can be changed in the settings panel.

### Moving to WebAssembly later

The UI talks only to `EngineBackend` (`types.ts`) and gets structured-clone data back. The engine code uses integer moves, typed arrays and no allocations in the hot path, so it maps directly onto AssemblyScript, or onto Rust/C++ compiled to WASM. To switch, implement `EngineBackend.search` on top of the WASM module and return it from `createBackend()`. The UI shows which backend is in use.

## Search visualisation

While the engine thinks, the right-hand panel updates about every 80 ms:

- **Stat tiles**: completed and current depth, eval, nodes and speed, time, TT hit rate, effective branching factor.
- **Progress** through the root moves at the current depth, with the move being searched outlined on the board.
- **Principal variation**, also drawn on the board as numbered markers.
- **Search tree icicle**: candidate moves on top and opponent replies below. Width is proportional to nodes searched, colour is the score, and faded segments are alpha-beta bounds rather than exact scores.
- **Nodes per iteration** and **nodes per ply** on log scales, with cutoff rates in the tooltips.
- **Candidate table** with score bars and each move's share of the nodes.
- **Board overlays**: candidate scores for every root move, or a heatmap of where the search moved pawns and tried walls.

Turn on "Exact score for every candidate" to search every root move with a full window, as the C++ version did. Every move then gets a true score instead of a bound, and the search runs slower.

### Fast engine for algorithm experiments

The default engine now uses scalar bitboard pathfinding, measured at **2.59× the
full-search throughput** of the original implementation on an M1 Pro. Run
`npm run bench:kernels` to compare five pathfinding approaches. See
[benchmark results and engine API examples](bench/README.md) for methodology,
reproducible timings, and the optional PyTorch/MPS batch experiment.

## Training dashboard

```sh
npm run dashboard
```

Open **http://127.0.0.1:8765/?dashboard=1**. This builds the frontend and starts a local Python
server using the existing `.venv` training dependencies. Training runs under
`experiments/training` are discovered automatically, including nested run directories.
Select runs to compare evaluation scores, goal rates, policy/value losses, and
saved configuration/performance. Saved metrics refresh every 15 seconds; older
runs without `summary.json` are summarized from `metrics.json`.

In **Play & watch**, select Human, Built-in negamax, or either exported model
from any run for White and Black. Choose two bots to watch a match, and use
Pause/Resume, Undo, and New game to control it. Neural simulations control MCTS
strength; think time and other engine settings apply to built-in negamax.
Neural inference runs on CPU and caches up to four exported models, reloading
when their files change. Exports reflect the latest pause/completion, while
training charts reflect the latest saved checkpoint. Matches adjudicate threefold
repetition and 200 plies as draws. These are artificial match limits.

For dashboard frontend development, run `.venv/bin/python training/dashboard.py`
and `npm run dev` in separate terminals, then open
**http://localhost:5173/?dashboard=1**; Vite proxies `/api` to port 8765.
The dashboard's experiment-model controllers use its local Python server; the
default public arena uses the separate static model catalog and browser inference.
