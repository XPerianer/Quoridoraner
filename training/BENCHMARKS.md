# Training speed measurements

For model capacity, residual depth, and equal-time playing-strength comparisons,
see [the depth comparison](DEPTH_COMPARISON.md).

## Long-run profiling, rolling batches, and randomized budgets

Profiled one additional iteration from the completed 200-iteration
`both-values-full-curriculum` checkpoint, preserving its 52,120 historical game
records and full replay buffers. Both arms, 64 simulations, 128 games per arm,
32 updates per arm, final monitoring and held-out evaluation were included.
The source checkpoint was not modified. The stopped user trainer did not run
during these measurements.

| Profiled component | Before | After reporting fix |
| --- | ---: | ---: |
| Complete `run()` call, including final save/exports | 23.27 s | 19.45 s |
| Dashboard reporting | 5.28 s | 0.053 s |
| Final checkpoint/save function | 4.71 s | 4.77 s |

Plain reporting previously scanned the entire history to construct sections
that were never printed. It now exits before that work. The interactive
dashboard incrementally indexes new records and keeps only the last 128
training records per arm for recent-game statistics. Both profiles performed
exactly 16,109 inference batches and evaluated 226,407 positions. Complete game
records, losses, and both exported models matched exactly. cProfile adds
overhead, and its cumulative component times overlap; this single comparison
is not an unprofiled throughput estimate. Final serialization remains material.
See [profile measurements](measurements/long-run-profile.json).

Separate, **unprofiled** comparisons used five iterations, 256 games per arm
per iteration, 128 concurrent games, 32 updates at batch 256, the race
curriculum, MPS hybrid inference, and four native workers. Each seed compared
the same game/update counts and full evaluation budgets. The randomized variant
uses rolling scheduling with 64 new simulations on 25% of neural training moves
and 16 otherwise; only full searches supply policy targets. Evaluation always
uses 64. The runs include startup, baseline/final monitoring, final held-out
evaluation, checkpointing, and exports.

| Variant | Seed 1 wall | Seed 2 wall | Mean inference batch, seeds 1 / 2 |
| --- | ---: | ---: | ---: |
| Fixed budget, cohort scheduler | 59.13 s | 52.72 s | 16.5 / 14.8 |
| Fixed budget, rolling scheduler | 36.34 s | 42.96 s | 25.7 / 24.6 |
| Randomized budget, rolling scheduler | 32.09 s | 29.83 s | 13.7 / 15.6 |

Rolling scheduling reduced wall time by 19–39%; adding randomized budgets
reduced it by 43–46% relative to fixed cohorts (12–31% relative to fixed rolling).
The randomized runs actually requested about 28.1 new simulations per neural
training move. Mixed budgets shrink batches after shallow searches finish,
which helps explain why fewer simulations do not yield proportional savings.
With equal total/concurrent game counts there are no spare games to refill;
rolling scheduling alone cannot fix that final tail.

These are two seeds, not a confidence interval. Different schedules/budgets
change game trajectories and game lengths, so this is equal configured work,
not identical positions or model weights. All variants had zero goal wins in
the final full-game monitoring and held-out suites (20 games per arm per
suite per seed). This speed test therefore does **not** establish preserved
playing strength. Fixed budgets remain the default; randomization is opt-in.
The first seed's cohort/rolling measurements overlapped the regression suite
and were discarded, then repeated sequentially without tests. Variant order
was reversed for seed 2. Raw commands, timings, game counts, and evaluations:
[search budget measurements](measurements/search-budget-speed.json).

Reproduce with `training/benchmark_speed.py`; use `training/profile_segment.py`
for mature-checkpoint profiles. See the commands and semantics in
[the training README](README.md#speed-and-batching). The full regression suite
passed 58 tests with one hardware-gated skip, including exact CPU interruption
and resume with rolling scheduling, randomized budgets, and changed worker
counts. Tests also cover unique refill indices, unequal native search budgets,
full-budget evaluation, and value-only minibatches.

### Learning check on zero-wall races

A second comparison used 20 iterations, zero walls, no curriculum, 32 games
per arm per iteration, 32 concurrent games, 32 updates at batch 128, and the
same 64 versus randomized 16/64 budgets. Both variants use rolling scheduling;
with only 32 games, refilling cannot affect this experiment. Evaluation uses
64 simulations, 20 monitoring games every five iterations, and 20 final held-out
games per arm. Runs were sequential and the order was reversed for seed 2.

| Training budget | Seed | Wall time | Selfplay held-out wins | Negamax held-out wins |
| --- | ---: | ---: | ---: | ---: |
| Fixed 64 | 1 | 79.15 s | 4/20 | 2/20 |
| Randomized 16/64 | 1 | 54.83 s | 0/20 | 3/20 |
| Fixed 64 | 2 | 85.93 s | 3/20 | 3/20 |
| Randomized 16/64 | 2 | 51.08 s | 1/20 | 6/20 |

Randomization took 31–41% less time at equal iteration budgets, but pooled
selfplay held-out wins dropped from 7/40 to 1/40. Negamax wins increased from
5/40 to 9/40; these small samples do not establish a reliable improvement.
This is evidence against enabling this 25%-full-search recipe by default for
selfplay. Keep fixed budgets for normal runs and treat randomization as an
experiment. These are simpler zero-wall games, not full-wall strength, and the
comparison is not at equal wall-clock budgets. The full results include
monitoring trajectories and cutoff counts:
[learning measurements](measurements/search-budget-learning.json).

Reproduce this control with:

```sh
.venv/bin/python training/benchmark_speed.py --out experiments/training/budget-learning \
  --variants rolling randomized --seeds 1 2 --iterations 20 \
  --games 32 --parallel-games 32 --batch-size 128 --walls 0 --curriculum none
```

## MPS pipeline rewrite: measured before implementation

Measured on the local 10-core Apple Silicon Mac, macOS 26.6.2, PyTorch 2.14.1,
float32, four CPU search workers, seed 1. Each iteration generates 128 games
per learning arm with 16 new simulations per neural move, then performs eight
updates per arm at batch size 256; evaluation uses two games per arm. Network,
optimizer, budgets, openings and sampling settings are unchanged.

Before implementing the rewrite, the original four-process MPS pipeline took
14.40 seconds for one iteration. Its main components were:

| Component | Seconds |
| --- | ---: |
| Neural forward and host/device transfers | 5.35 |
| MCTS bridge: search, JSON serialization and pipes | 4.87 |
| Per-leaf policy preparation | 1.52 |
| Negamax bridge | 0.64 |
| Feature encoding in Python | 0.57 |
| Gradient updates | 0.27 |

A separate cProfile pass attributed about 3.1 seconds to JSON serialization
across bridge workers and the parent process. Profiling adds overhead and
thread-inclusive cumulative totals overlap, so those times are not added to
the unprofiled table. Inference microbenchmarks showed almost constant MPS
round-trip latency across batches 1–128 (roughly 0.8–1.1 ms after warmup).
Separately synchronized transfers cost roughly 0.23 ms into MPS and 0.26 ms
back; these forced synchronization measurements describe latency and should
not be summed as a throughput prediction.

FP16 and channels-last layouts did not consistently help this network.
`torch.compile` measured 1.26 ms at batch 32 and 1.72 ms at batch 128, versus
1.11 ms and 0.98 ms eager, excluding compilation time. Frozen tracing produced
small gains but is deprecated in this PyTorch/Python combination. None of
these alternatives was adopted. Asynchronous input copies subsequently
measured 0.87 ms versus 1.16 ms blocking at batch 32, and 1.09 ms versus 1.55 ms
at batch 128; bucketed/hybrid modes now use them, retaining the host buffer
until the blocking result copy completes.

### Rewrite and end-to-end results

The new `direct` backend keeps the complete MCTS pool and game engine in a
local C++ library, uses persistent CPU threads, fills contiguous float32
feature buffers directly, and consumes a batched policy tensor. This removes
JSON/pipes and per-leaf Python dispatch. It is an MPS-oriented data-path rewrite;
the branch-heavy game engine remains native CPU code. Padding to a few stable
batch sizes reduces MPS graph setup. Hybrid mode additionally runs batches of
16 or fewer positions on a synchronized CPU copy of the network; larger
batches and all gradient updates use MPS. It does not reduce simulation counts
or change model precision.

Initial repeated one-iteration comparisons (before asynchronous input copies):

| Pipeline | Loop, repeat 1 | Loop, repeat 2 | Full wall, repeat 1 | Full wall, repeat 2 |
| --- | ---: | ---: | ---: | ---: |
| Original bridge, MPS | 14.85 s | 14.62 s | 17.22 s | 16.75 s |
| Direct, eager MPS | 6.06 s | 6.83 s | 8.29 s | 9.06 s |
| Direct, bucketed MPS | 5.34 s | 5.58 s | 7.53 s | 7.95 s |
| Direct, hybrid MPS/CPU | 4.22 s | 4.38 s | 6.45 s | 6.68 s |

Final sustained comparison: three iterations, including the asynchronous-copy
optimization, growing replay buffers and repeated model updates:

| Pipeline | Training loop | Full process wall time |
| --- | ---: | ---: |
| Original bridge, MPS | 39.21 s | 41.78 s |
| Direct, bucketed MPS | 14.74 s | 17.26 s |
| Direct, hybrid MPS/CPU | **10.81 s** | **13.53 s** |

That is **3.63× loop throughput and 3.09× full-process throughput** for hybrid,
and 2.66× / 2.42× for GPU-only bucketed inference. Every comparison above
matched complete game records, loss histories and both exported models
**exactly on this machine**. The three-iteration runs each evaluated 301,963
neural positions in 7,646 waves; hybrid routed 4,225 small waves to CPU and
3,421 to MPS. This exact match is evidence for these workloads, not a promise
of bitwise equality across devices or future PyTorch versions.

The remaining hybrid costs over three iterations were 6.01 seconds in neural
forward/transfers, 1.20 seconds in native MCTS plus encoding, 0.59 seconds in
negamax, 0.55 seconds in gradient updates, and 0.41 seconds in policy
preparation. Neural dispatch/compute is now the largest component. These
measurements do not claim a theoretical maximum or improved playing strength.

Full wall time includes interpreter/model startup, saves and exports. The
training-loop timer excludes the final save and exports; periodic saves are
included. The original three-iteration run made two periodic saves plus its
final save (1.54 seconds combined), while the faster runs only made the final
save (about 0.63 seconds). Thus some loop gain is due to avoided periodic
checkpoint work. Comparisons were sequential, not competing for CPU/GPU.
The native library was built before timed runs; first-time builds add setup.

Raw results: [baseline](measurements/mps-pipeline-before.json),
[layouts](measurements/mps-layout-before.json),
[compilation](measurements/mps-compile-before.json),
[transfers](measurements/mps-transfers.json),
[repeated pipeline runs](measurements/mps-pipeline-rewrite.json),
[three-iteration comparison](measurements/mps-pipeline-sustained.json).

```sh
npm run train:bench:pipeline -- --out experiments/training/mps-reproduce --device mps
npm run train:bench:pipeline -- --out experiments/training/mps-sustained-reproduce \
  --device mps --iterations 3 --repeats 1 --variants bridge bucketed hybrid
```

## CPU worker pool: October 23

The same 128-game, one-iteration CPU workload below was run sequentially with
1, 2 and 4 engine processes, using one shared neural model per learning arm.
No simulation, update, game-count or batch-size settings changed.

| Component | 1 worker | 2 workers | 4 workers | 4 workers repeat |
| --- | ---: | ---: | ---: | ---: |
| Total training loop | 16.33 s | 14.09 s | 13.50 s | 13.01 s |
| MCTS bridge | 6.70 s | 5.42 s | 5.00 s | 4.83 s |
| Negamax bridge | 1.92 s | 1.06 s | 0.64 s | 0.63 s |
| Neural inference and preparation | 5.86 s | 6.11 s | 6.21 s | 6.09 s |

Four workers reduced loop time by **17–20%**, or **1.21–1.25× throughput**,
relative to one worker with the earlier feature-encoding optimization. The
search/negamax work now executes in separate CPU processes; Python pipe I/O
uses threads to keep requests in flight simultaneously. All workers' pending
leaves are evaluated in one ordered batch. CPU search and inference still
alternate rather than overlap.

All configurations evaluated 126,571 neural positions in 3,148 batches.
Before/after game records, loss histories and model weights matched exactly.
Four-worker runs finished before the 15-second periodic-checkpoint threshold,
so part of the observed total gain comes from avoiding one periodic save.
The final checkpoint is excluded from the loop total in all configurations.
These are small local samples, not an A100 or MPS performance claim.

Reproduce using the command below with `--workers 1`, `--workers 2`, or
`--workers 4` and a distinct output directory per run. Existing checkpoints
can also resume with `--workers 4`; their trees are redistributed automatically.

## 128-game workload: October 23 optimization

Measured one complete iteration matching the latest run's settings: 128 training
games per arm, 128 parallel games, batch size 256, 16 simulations, 8 updates per
arm, two evaluation games per arm per evaluation, and the TypeScript engine.
Device was explicitly CPU. Runs were sequential, with seed 1 and plain output.
This includes initial/final evaluation and both learning arms.

| Training-loop component | Before | After | After repeat |
| --- | ---: | ---: | ---: |
| Total loop | 20.44 s | 16.59 s | 16.51 s |
| Inference including encoding and policy preparation | 9.29 s | 5.90 s | 5.87 s |
| MCTS bridge including engine work and JSON transport | 7.07 s | 6.76 s | 6.73 s |
| Fixed negamax opponent | 1.92 s | 1.91 s | 1.92 s |
| Gradient updates | 0.46 s | 0.51 s | 0.46 s |

This is about **19% less loop time (1.23× throughput)** for identical work:
126,571 neural positions in 3,148 inference batches. The before/after complete
game records, losses, and both final models were checked for exact equality.
These few runs establish a local speed improvement, not a general hardware or
playing-strength result. MPS was not benchmarked in this comparison.

A separate baseline cProfile run attributed 4.30 seconds to 134,776 calls of
the scalar tensor feature encoder. The optimized encoder builds one float32
NumPy batch and shares it with PyTorch; inference encoding now takes about
0.45 seconds. This is a component comparison across profiled/unprofiled runs,
so use the unprofiled total-loop table for the overall gain. MCTS also uses
path-local repetition counts inheriting game history instead of copying the
entire history for each simulation. Neither change alters search budgets,
model architecture, learning targets, or optimizer settings.

New `performance.json` fields split inference into encoding, forward pass with
device transfers, and legal-policy preparation. The remaining largest cost is
the search bridge; gradient updates are only about 3% of the loop in this
configuration. Faster game generation helps wall-clock learning throughput;
whether more updates per iteration improve playing strength needs a separate
learning experiment.

Reproduce the workload (choose a fresh output directory each time):

```sh
.venv/bin/python training/train.py --out experiments/training/speed-128 \
  --iterations 1 --games 128 --parallel-games 128 --batch-size 256 \
  --device cpu --plain
```

The loop timer excludes startup, exports, and the final checkpoint; periodic
checkpoints are included. Individual component timers are not all additive
(for example, checkpoint snapshots also count as bridge requests).

## Earlier quick-preset measurements

Measured locally on an Apple Silicon Mac on 2026-10-22, with the small default
network and PyTorch 2.14.1. These are single-run observations, not a rigorous
hardware comparison. Full-run wall time includes Python/Node startup and exports.

Both versions used the default quick experiment: two iterations, two training
games per arm per iteration, two evaluation games per arm at iteration zero and
after each iteration, 16 new simulations per neural move, and a 2,000-node fixed
negamax opponent. Output was line-oriented (`--plain`).

| Runtime | Before | After | Before moves/s | After moves/s |
| --- | ---: | ---: | ---: | ---: |
| CPU | 9.69 s | 3.39 s | 47.0 | 119.0 |
| Apple MPS | 23.80 s | 6.52 s | 19.1 | 62.0 |

Each run completed 20 games. The old runs played 455 plies, the new runs 404.
Retaining search visits and batching games change trajectories, so elapsed time
is not a comparison of identical work. Move throughput improved approximately
2.5× on CPU and 3.2× on MPS. These measurements say nothing about playing strength.

A separate cProfile baseline attributed about 10.0 s of its 14.8 s runtime to
checkpointing, including serialization and log rewrites. Profiling adds overhead,
so these totals are not directly comparable to the unprofiled table above.

In the optimized CPU run, the training loop itself took about 1.64 s:

- Neural inference and feature/output preparation: 0.61 s.
- Search bridge round trips, including TypeScript traversal: 0.48 s.
- Negamax bridge round trips: 0.19 s.
- Gradient updates: 0.15 s.
- Final training-loop checkpoint: 0.018 s.
- 1,008 retained visits were available when searches started on reused roots.

The default two-game phases leave small neural batches (average 1.25 positions
in this run). Increase the number of games and `--parallel-games` to use larger
batches; do not expect the GPU benefit of large batches from a two-game preset.

## Neural inference microbenchmark

Medians after warmup, including host-to-device and device-to-host transfers:

| Batch | CPU | MPS |
| ---: | ---: | ---: |
| 1 | 0.085 ms | 0.807 ms |
| 2 | 0.152 ms | 0.886 ms |
| 8 | 0.381 ms | 0.934 ms |
| 32 | 1.058 ms | 1.144 ms |
| 64 | 2.019 ms | 1.325 ms |

CPU is faster for the quick preset on this machine. MPS overtakes it at batch 64
in this inference-only measurement. Training updates, variable batch sizes,
engine work, and startup costs are not measured by this microbenchmark.

Recheck on your own machine:

```sh
npm run train:bench -- --out experiments/training/devices.json
npm run train -- --out experiments/training/cpu --device cpu --plain
npm run train -- --out experiments/training/mps --device mps --plain
```

The trainer writes `performance.json` on pause/completion, with inference batch
counts, retained visits, and timing categories. `bridge_seconds` is the parent
total of the three bridge categories, so do not add it to those categories.
Training-loop counters exclude startup and compact model export writes.

## C++ engine trial

The native-vs-TypeScript search benchmark uses the same 81 reachable positions
and deterministic node budgets. On this Mac, native C++ search was **3.56× faster
at 2,000 nodes** and **2.85× faster at 50,000 nodes**. Both produced exactly the
same best move, score, and number of nodes on every tested position. C++ legal
move generation through the Node-API binding was still 9% slower, so the native
search can also run with the TypeScript MCTS rules using `--engine native-negamax`.

In a final sequential three-way comparison with identical seed, budgets,
training settings, game records, and 404 total plies, full-run times were 3.29 s
for TypeScript, 3.15 s for `native-negamax`, and 3.18 s for the full native
engine. That is a 3–4% reduction in this run. An earlier single comparison
measured 3.69 s versus 3.35 s, so startup and run variation make the small
whole-run difference noisy. Across those runs, the accelerated component was
clearer: fixed-opponent search took 0.195 s in TypeScript and 0.066 s in native
C++, while the total training-loop time was about 1.6–1.8 s. Legal generation,
neural inference, and state conversion limit the whole-run gain. TypeScript
remains the default; use `--engine native-negamax` for the cleanest native
experiment or `--engine native` to exercise the complete native move/search
engine.

Build and rerun both comparisons on your machine:

```sh
npm run train:native:build
npm run train:native:bench -- experiments/training/native-search.json
npm run train -- --engine typescript --out experiments/training/reference --device cpu
npm run train -- --engine native-negamax --out experiments/training/cpp-opponent --device cpu
npm run train -- --engine native --out experiments/training/cpp-full --device cpu
```
