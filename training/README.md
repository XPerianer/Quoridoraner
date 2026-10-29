# AlphaZero-style Quoridor lab

For the active `az.py` pipeline, see [shared lookup, endgame solving, batching
and VM recovery](AZ_PIPELINE_SPEED.md), including measured speedups and limits.

> **New (2026-10-28): `training/az.py` beats the 2000-node negamax.** It uses
> path-feature inputs, a supervised negamax bootstrap, and then MCTS RL. See
> [AZ_REPORT.md](AZ_REPORT.md) for why the models below never won, what changed,
> and the measured results. The rest of this README describes the original
> `train.py` lab.

Two independent policy/value networks start with identical weights:

- **selfplay** learns from games in which the current network searches for both sides.
- **negamax** learns policy from its own MCTS turns and value from both players' positions against the existing `TsSearcher`, at a fixed node budget. The learner alternates colors. Negamax moves are not imitation targets.

A small convolutional network predicts a probability for each of the 209 move IDs and a value in `[-1, 1]` for the player to move. PUCT Monte Carlo tree search (MCTS) uses those predictions to explore continuations. Root visit counts become policy training targets; completed game outcomes become value targets. Training uses exploration noise and samples early moves. Evaluation uses neither.

The TypeScript bridge calls the existing board and negamax code directly. All legal walls are available. MCTS uses the learned policy/value model on both sides of its hypothetical continuations, including when the real opponent is negamax. This is outcome-based reinforcement learning, not backpropagation through the opponent or an opponent-specific minimax tree.

New runs default to `--negamax-value-turns both`: each pre-move position receives
the final outcome from its side-to-move perspective. Opponent positions have no
policy target. Policy loss is averaged over learner samples only; value loss
uses the entire batch. Use `--negamax-value-turns learner` for the previous
objective in a fresh control run. Older checkpoints retain learner-only value
training on resume; existing replay and in-flight games are preserved. The
new setting is saved in checkpoints and cannot be changed on resume.

## Start a quick experiment

From the repository root:

```sh
npm install
python3 -m venv .venv                         # only if no environment exists yet
.venv/bin/python -m pip install -r training/requirements.txt
.venv/bin/python training/train.py --out training/runs/my-first-run
```

Automatic device selection benchmarks CPU and available Apple MPS/NVIDIA CUDA inference, including host/device transfers, at the configured game batch size. It selects the faster device and saves the measurements. A resumed run retains that choice unless you override `--device cpu`, `--device mps`, or `--device cuda`. The program uses one PyTorch CPU thread. No browser changes or model downloads are needed.

After setup, `npm run train -- --out training/runs/my-first-run` is an equivalent shortcut.

In a terminal, the colored live dashboard is organized into live telemetry,
per-arm learning, training quality, run settings, and loss panels. It shows the
current arm and task, overall run progress with an iteration-time ETA, phase
progress and rate, MCTS inference waves, active games and curriculum tasks, game
ply, replay/session throughput, and evaluation suites. Each arm's row includes
policy/value loss, the change since the previous update, a short loss sparkline,
and its latest score against the fixed negamax baseline. Recent goal,
repetition, and artificial-cutoff rates are color-coded. Redirecting stdout, or
adding `--plain`, produces ordinary log lines.

The loss plots compare self-play (cyan circles) and negamax training (magenta
diamonds), print the latest value for each arm in the panel title, and give each
loss its own labeled vertical scale. The horizontal axis is cumulative optimizer
updates. Curves include saved history after resume and refresh after every
update. The layout adapts to narrow terminals by dropping optional panels rather
than overwriting the live metrics.

New runs default to 2 iterations, 2 training games per model per iteration, **64 new MCTS simulations** per neural move, and 8 optimizer updates per iteration. Monitoring uses **20 games per model** at baseline, every **10 iterations**, and at the final iteration. The final evaluation also includes **40 additional held-out games per model**. For a fast plumbing check, explicitly pass `--simulations 8 --eval-games 2 --heldout-games 0`. Games stop at a goal, threefold repetition, or 100 plies. The fixed negamax opponent has a 2,000-node budget and searches all legal moves. As in the existing engine, the node limit is approximate and depth one always completes.

The default two-game training batch verifies the learning process; it is not enough to claim increased strength. High branching factors and sparse win/loss rewards make Quoridor challenging. Repetition and move-limit cutoffs are artificial training adjudications, **not Quoridor draw rules**. They receive zero value targets and half-score for monitoring, consistently with the search engine. Cutoff-heavy runs can teach a near-zero value function. More games, more simulations, and larger independent evaluation samples are needed to establish improvement.

## Performance-gated curriculum and balanced replay

For a fresh run that advances only after measured performance:

```sh
.venv/bin/python training/train.py \
  --out training/runs/gated-balanced-full \
  --curriculum gated --replay-sampling task --negamax-value-turns both \
  --walls 10 --curriculum-games 40 --curriculum-win-rate 0.4 --curriculum-streak 2 \
  --iterations 200 --games 128 --parallel-games 128 \
  --updates 32 --batch-size 256 --simulations 64 \
  --eval-games 20 --eval-every 10 --heldout-games 40 \
  --engine direct --workers 4 --device mps --mps-inference hybrid \
  --checkpoint-seconds 60 --plain
```

Each arm starts at race-0 and advances independently through race-1, race-2,
and race-3 (wall inventories 0, 2, 5, and the target inventory). Earlier races
remain in the training mixture. Unlike `--curriculum races`, full standard
starts are introduced only at stage 3, as one pair in four. Iteration count
alone never advances a gated run; `--curriculum-interval` does not apply.

At each evaluation, each arm plays 40 additional games against the fixed
`--negamax-nodes` opponent at its **current race difficulty**, without training
noise or move sampling. Pairs share a position and exchange learner colors.
The openings use an independent seed namespace that changes each evaluation.
Only goal wins count: by default, 16/40 wins in two consecutive suites unlocks
the next stage. A failed suite resets the streak, and baseline evaluation
cannot promote. The 40% threshold is a configurable practical starting point:
short races often already favor one player, so paired games can limit even a
strong player's wins to roughly half. It is not a proof of mastery.

Full-game monitoring and final held-out evaluations remain unchanged and do
not drive this gate. `--adaptive-negamax`, if enabled separately, still has
its own promotion suite; curriculum tests always use the fixed opponent.
`curriculum.json` and `latest.pt` save each arm's stage, streak and gate history.
The stage is also shown in the terminal, and `diagnose.py` reports it.

`--replay-sampling task` tags samples by race level or standard game, reserves
equal buffer capacity for each observed task, and allocates equal minibatch
shares to each nonempty task. Remainders are distributed across tasks; sparse
tasks are sampled with replacement when necessary. Each task retains its
newest samples; spare buffer capacity is left unused rather than letting long
games crowd out short tasks. The total buffer remains bounded by `--replay-size`.
Opponent value-only rows remain value-only. Uniform replay is still available
and is the default unless this flag is supplied.

Existing runs retain their scheduled curriculum and uniform replay on resume.
Use a new directory for this experiment; resume restores all gate and sampling
settings without modification. Pause and resume the new run normally:

```sh
.venv/bin/python training/train.py --resume training/runs/gated-balanced-full
```

## Adaptive negamax opponent

For a controlled learning diagnostic, see [the zero/two-wall investigation](DIAGNOSTICS.md).
Audit saved results, including curriculum sample imbalance, with:

```sh
.venv/bin/python training/diagnose.py training/runs/adaptive-64-u32
```

Start a fresh run with a weaker training opponent and win-gated promotions:

```sh
.venv/bin/python training/train.py \
  --out training/runs/adaptive-64-u32 \
  --iterations 200 --games 128 --parallel-games 128 --batch-size 256 \
  --updates 32 --simulations 64 --curriculum races --curriculum-interval 25 \
  --adaptive-negamax --negamax-start-nodes 32 --negamax-nodes 2000 \
  --promotion-win-rate 0.6 --promotion-games 40 --promotion-streak 2 \
  --eval-games 20 --eval-every 10 --heldout-games 40 \
  --engine direct --workers 4 --device mps --mps-inference hybrid \
  --checkpoint-seconds 60 --plain
```

Only the negamax training arm changes opponent strength. At each evaluation,
including baseline and final, it plays an additional 40 paired games on independent
`promotion` openings with the target wall inventory. At least 24 actual goal wins
in each of two consecutive evaluation suites doubles the budget, capped at 2,000.
A failed suite resets the streak; artificial cutoffs count as failures. Easy race
training wins never trigger promotion. Both arms retain their fixed 2,000-node
monitoring and final held-out evaluations.

`latest.pt` saves the current budget, streak, last processed evaluation, and full
promotion history. `opponent.json` exposes that state at each checkpoint, and
`metrics.json` includes each game's opponent node budget. Resume with
`--resume training/runs/adaptive-64-u32`; settings and progression are restored.
Existing fixed-opponent checkpoints remain fixed; use a new directory to enable
adaptation. Node budgets are approximate: depth one always completes, so even
32 nodes can be a capable opponent and adjacent budgets can behave similarly.

## Pause, resume, extend

Press **Ctrl-C** once. Training finishes the current batch of moves or gradient update, writes a checkpoint, exports both bots, and exits. SIGTERM follows the same path.

```sh
.venv/bin/python training/train.py --resume training/runs/my-first-run

# Extend the total iteration budget, including a completed experiment:
.venv/bin/python training/train.py --resume training/runs/my-first-run --iterations 10
```

An atomic `latest.pt` is written at startup, every 15 seconds at a completed move-batch/update boundary, and on pause or completion. Set `--checkpoint-seconds 10` to change the interval. It contains both models and AdamW optimizer states, bounded replay buffers, every active game and its training samples, retained search trees, repetition history, random generator states, phase/counters, results, and configuration. A crash resumes from the last completed checkpoint; work since that checkpoint is recomputed. The interval can be exceeded by the time needed to finish one move batch or update. Graceful Ctrl-C saves all completed work. A process lock prevents two trainers writing to one run. Existing runs cannot be overwritten accidentally.

Resume restores training settings. Only `--iterations` (extend only), `--device`, `--checkpoint-seconds`, `--workers`, `--engine`, `--mps-inference`, and `--plain` can change. Existing v1 checkpoints migrate automatically and keep single-game scheduling; fresh runs use batched scheduling. Migration preserves saved training state but uses the new search algorithm going forward. CPU interrupt/resume is tested for exact equality against an uninterrupted run. GPU behavior and switching devices are not guaranteed bit-for-bit reproducible. MPS/CUDA RNG state is saved when that device is used; checkpoint loading uses PyTorch's restricted `weights_only=True` loader ([PyTorch serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization)).

## Use either bot

```sh
# Play as white (pawn squares e2; wall anchors d5h / d5v; ? lists legal moves).
.venv/bin/python training/bot.py play training/runs/my-first-run/selfplay.pt

# Compare the other model with negamax on paired openings.
.venv/bin/python training/bot.py evaluate training/runs/my-first-run/negamax.pt --games 20

# JSON move ID, notation, and visit-count policy from the initial board.
.venv/bin/python training/bot.py move training/runs/my-first-run/selfplay.pt

# Or provide a Position JSON matching src/engine/types.ts.
.venv/bin/python training/bot.py move training/runs/my-first-run/selfplay.pt --position position.json
```

`bot.py` also accepts `--device`, `--simulations`, `--max-plies`, `--negamax-nodes`, `--seed`, and (for play) `--human black`. The `move` command treats the supplied position as a fresh game with no earlier repetition history. Training and complete CLI games track that history.

## Optional native C++ engine trial

The experimental Node-API engine ports Quoridor moves, walls, shortest-path distance, hashing, and fixed-budget negamax to C++17. It builds locally against the installed Node headers; no package or source download is required.

```sh
npm run train:native:build
npm run train -- --engine native --out training/runs/native
# Keep TypeScript moves for MCTS and accelerate only its fixed negamax opponent:
npm run train -- --engine native-negamax --out training/runs/native-opponent
npm run train:native:bench -- training/runs/native-search.json
```

These Node-based options are `--engine typescript` (the default), `native`, and `native-negamax`. The faster in-process `direct` backend is described below. Bot commands accept the same `--engine` option. The compiled addon is ignored by git and can be rebuilt with `npm run train:native:build`. Existing checkpoints retain their backend on resume; explicitly pass `--engine` to switch. Cross-backend training-resume reproducibility is not guaranteed.

Differential tests compare legal-move ordering, path distances, hashes, transitions, MCTS policies, and negamax move/score/node results. A 100-game Address/UndefinedBehaviorSanitizer run checks make/unmake consistency.

## Speed and batching

See [local before/after measurements](BENCHMARKS.md) for timings and their limitations.

New runs use `--game-scheduler rolling`: finished slots are immediately filled
while unstarted games remain in the current phase. `--game-scheduler cohort`
retains the previous behavior of waiting for all active games to finish.
Older checkpoints retain cohort scheduling. Scheduling is saved and cannot
change on resume; start a separate experiment to compare it. Rolling scheduling
changes RNG consumption order and game trajectories, even with the same seed.
It helps when `--games` exceeds `--parallel-games` (for example 256 and 128).
If both are 128, there are no spare games to fill the final shrinking batch.
The scheduler does not start games from future iterations or use stale weights.

Randomized training search is opt-in:

```sh
.venv/bin/python training/train.py --out training/runs/randomized-budget \
  --iterations 200 --games 256 --parallel-games 128 --game-scheduler rolling \
  --search-budget randomized --simulations 64 --fast-simulations 16 \
  --full-search-probability 0.25 --updates 32 --batch-size 256 \
  --engine direct --workers 4 --device mps --mps-inference hybrid \
  --checkpoint-seconds 60 --plain
```

Each neural training move independently receives 64 new simulations with
probability 0.25, otherwise 16 (an expected 28). Full searches provide policy
and outcome-value targets; shallow searches provide outcome-value targets only.
All evaluation suites still use 64 simulations. Retained tree visits are reused
as before, so these budgets count **new** simulations, not total root visits.
Budget choices use a separate deterministic per-move seed and survive resume.
These settings change the learning objective and cannot be overridden on resume.
Fewer simulations do not guarantee equal strength or proportional speedups:
mixed budgets can leave small inference batches, and fewer rows supervise policy.
The initial two-seed zero-wall control ran faster but reduced selfplay held-out
wins from 7/40 to 1/40. Keep fixed budgets for normal selfplay runs; see
[the measured learning tradeoff](BENCHMARKS.md#learning-check-on-zero-wall-races).

Compare both schedulers and randomized budgets sequentially with two seeds:

```sh
.venv/bin/python training/benchmark_speed.py --out training/runs/speed-comparison
# Profile one additional iteration, including mature history/replay, in a new directory:
.venv/bin/python training/profile_segment.py \
  --checkpoint training/runs/both-values-full-curriculum/latest.pt \
  --out training/runs/profile-segment --device mps
```

The profiler requires a completed checkpoint and never modifies its source.
It preserves history/replay, performs one additional iteration with final
evaluation, and saves cProfile output plus a resumable result. Profiling adds
overhead; compare its breakdown separately from unprofiled benchmark wall times.
`performance.json` now records training search moves, requested simulations,
and full-search moves to audit the realized budget and policy-target frequency.

```sh
# Compare CPU/MPS/CUDA at several inference batch sizes (JSON output is optional).
npm run train:bench -- --out training/runs/devices.json

# More games in flight produce larger inference batches.
npm run train -- --out training/runs/batched --games 32 --eval-games 16 \
  --parallel-games 8 --checkpoint-seconds 15
```

`--parallel-games` defaults to 8 and is capped by the number of games remaining in the phase. Games advance in deterministic waves. One pending leaf from each neural turn is evaluated in a single PyTorch call. Negamax turns are handled by the existing CPU engine; games waiting on those turns do not add neural inference positions. This controls batch size, independently of the CPU worker count configured with `--workers`.

MCTS selection, expansion, legal move generation, and backup run inside the CPU engine workers. Python exchanges batches of neural requests instead of making an engine call for every child position. After each actual move, the matching child becomes the new root. Both its evaluation and previous visits are reused; policy targets normalize **all** retained plus new child visits. Exploration noise is applied only at the current root and never overwrites its network priors. Trees are dropped when games finish or model weights change.

Each worker's negamax opponent reuses one searcher's preallocated buffers, invalidating its transposition table before every move so its decisions remain independent of earlier games. Live loss plots refresh during updates; model and replay serialization no longer runs after every move.

The tiny default network often runs faster on CPU at small batch sizes. MPS becomes more competitive with larger batches; the included benchmark measures your machine instead of assuming a GPU is faster. Auto-selection is an inference microbenchmark, not a prediction of the complete training workload. `performance.json` records actual training-loop timing, bridge requests, inference batch sizes, checkpoint timing, and retained visits. For Node backends, bridge time includes engine computation plus transport; it is not pure IPC time. Timings exclude process/model startup and exported-model writes.

Inference timing also includes `encoding_seconds`, `forward_seconds` (including device transfers), and `policy_seconds` for legal-action probabilities. These three are components of `inference_seconds`, so do not add them to that total. Input features are built as one contiguous CPU batch and shared with PyTorch to avoid thousands of scalar tensor operations. With `direct`, encoding happens inside C++ and is included in `search_bridge_seconds`; that timer then measures native search work rather than IPC.

### CPU search workers and shared GPU inference

With the Node backends, use `--workers 4` to run search in four independent Node.js processes. With `direct`, the same flag selects native CPU threads inside the Python process. Each
worker owns a subset of games and retained trees, with its own negamax searcher.
Python collects one pending leaf per game across all workers and runs **one
combined inference batch** on the selected CPU, MPS, or CUDA device. Workers
never load neural models. Searches advance in synchronized waves; CPU search
and inference do not overlap yet. Request/result ordering stays deterministic.

```sh
# Resume an existing large run with four CPU search workers.
.venv/bin/python training/train.py --resume training/runs/20261022-220205 --workers 4

# NVIDIA host with CUDA-enabled PyTorch, Node.js and npm dependencies installed:
python training/train.py --resume training/runs/20261022-220205 --workers 4 --device cuda
```

`--workers` defaults to 1 for compatibility and small workloads. The actual
process count is capped by `--parallel-games` and the largest phase's game
count. Start with 2–4 workers, depending on available CPU cores; more processes
can increase communication overhead. Compare complete runs using
`performance.json`, not just neural inference benchmarks. On this Mac, the
128-game workload benefited from four workers; see [measurements](BENCHMARKS.md).

You can change worker count when resuming, including old single-process
checkpoints. Snapshots keep the existing tree format and are redistributed to
the new workers. CPU tests check exact policies, model updates and interrupted
resume across different worker counts. CUDA execution has a hardware-gated
smoke test but has not been measured on an A100 here. On a hosted runtime,
keep a persistent copy of your run directory so the checkpoint survives the VM.

## Fast MPS training with the direct engine

The `direct` backend rewrites the complete search hot path in C++17: MCTS,
retained trees, repetition checking, legal moves, feature encoding, and the
negamax opponent. Python calls a local shared library through `ctypes`, which
releases the GIL. Persistent native threads perform independent searches; the
same neural model evaluates their combined batch. No Node processes, JSON,
or pipes are used by this backend. Model architecture, float32 precision,
simulation budgets, training targets, and optimizer settings are preserved.

```sh
# Pause an active run with Ctrl-C before resuming it here.
.venv/bin/python training/train.py --resume training/runs/20261022-220205 \
  --iterations 2000 --engine direct --workers 4 --device mps --mps-inference hybrid

# Or start a fresh large run:
.venv/bin/python training/train.py --out training/runs/mps-fast \
  --iterations 1000 --games 128 --parallel-games 128 --batch-size 256 \
  --engine direct --workers 4 --device mps --mps-inference hybrid
```

The shared library builds automatically on first use or when its sources
change, using a local C++17 compiler. On macOS this requires Xcode Command Line
Tools; no Python/Node headers or package downloads are needed. Build explicitly
with `npm run train:direct:build`. Build artifacts are ignored by Git. The
backend also works on CPU and CUDA; the following inference modes affect MPS
only:

- `bucketed` (default): float32 GPU inference padded to powers of two, with a
  minimum batch of eight. Reusing a few shapes avoids repeated graph setup.
  Input transfers are asynchronous; the result transfer synchronizes before
  buffers can be reused.
- `hybrid`: the same GPU path for batches larger than 16; smaller batches use
  a CPU copy of the current model. The copy refreshes after weight updates.
  This was the fastest mode measured locally, because small MPS round trips
  cost more than their CPU computation. Gradient updates remain on MPS.
- `eager`: unpadded, ordinary MPS transfers, for comparisons.

`--mps-inference` can change on resume. CPU search workers remain useful with
all three modes. Hardware-specific gains and the crossover at batch 16 should
be remeasured on a different Mac. Choosing `bucketed` keeps all neural forwards
on the GPU. Float32 kernels and reduction order can differ across devices, so
cross-backend equality on the measured runs is not a general bitwise guarantee.

Old Node-backend checkpoints can resume directly with `--engine direct`, and
the retained trees can also be restored back into the Node backend. The old
checkpoint and export formats are preserved. Tests check this migration,
repetition draws, legal moves, terminal values, multiworker search, and exact
CPU interruption/resume. `performance.json` additionally records CPU/MPS batch
counts in the direct MPS path.

Reproduce complete comparisons, including wall time, checkpoints, game records,
losses and exported weights:

```sh
npm run train:bench:pipeline -- --out training/runs/mps-comparison --device mps
# Include several model updates and growing replay buffers:
npm run train:bench:pipeline -- --out training/runs/mps-sustained \
  --device mps --iterations 3 --repeats 1 --variants bridge bucketed hybrid
# Inspect layouts, precision, compilation, and host/device transfers:
.venv/bin/python training/benchmark_mps_layout.py --out training/runs/mps-layout.json
```

See [BENCHMARKS.md](BENCHMARKS.md) and [raw measurements](measurements/) for the
before/after evidence. These benchmarks measure execution speed, not improved
playing strength.

## Settings and artifacts

Run `training/train.py --help` for all flags. Useful larger-run settings:

```sh
.venv/bin/python training/train.py --out training/runs/larger \
  --iterations 20 --games 20 --simulations 128 --updates 100 --eval-games 20
```

Training and evaluation game counts must be even. New runs use varied, seeded legal pawn/wall openings, with colors swapped within each pair. Monitoring openings are fixed across iterations and both arms. Final-only held-out openings use a separate seed namespace and longer opening sequences; results are reported separately. Once inspected and used to tune settings, those results are no longer held out for subsequent experiments: use `bot.py evaluate --seed 2027 --games 40` for a fresh test. Evaluation always uses the target wall budget and fixed negamax opponent, never curriculum positions. Score is wins plus half of artificial cutoffs divided by games, not a win percentage or an Elo estimate. Old checkpoints keep their original four-pawn-move evaluation and cadence.

Each run directory contains:

| File | Contents |
| --- | --- |
| `latest.pt` | Complete resumable checkpoint, including both models, active games, and retained search trees |
| `selfplay.pt`, `negamax.pt` | Compact bot exports, refreshed on pause or completion |
| `config.json` | Resolved training settings |
| `metrics.json` | Completed training/evaluation games, openings, moves, outcomes and termination reasons |
| `summary.json` | Per-iteration and per-suite wins, losses, artificial cutoffs, repetition/goal rates, average plies, and score |
| `losses.json` | Policy/value loss for every gradient update |
| `performance.json` | Training-loop timing and batching/reuse counters, updated on pause/completion |
| `device-benchmark.json` | Auto-selection CPU/MPS/CUDA timings on new auto-device runs |
| `run.lock` | Advisory process lock; harmless when no trainer is running |

The models have separate optimizers and replay buffers (10,000 positions each by default), and no experience is shared. Input planes encode pawns, walls, wall inventories, turn, current repetition count, and remaining move budget. Both players' values are trained from their own perspective. The default is a compact CNN; optional residual models are described below. Distributed actors, champion gating, and production-scale training infrastructure are not included.

## Validation

```sh
.venv/bin/python -m unittest discover -s training -p 'test_*.py' -v
npm test
npm run build
npm run lint
```

The direct engine also has a standalone memory-safety stress test:

```sh
clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread training/native/direct_sanitize.cpp -o /tmp/quoridor-direct-sanitize
/tmp/quoridor-direct-sanitize
```

Tests cover legal moves, immediate-win search for both players, draw boundaries, outcome perspective, replay bounds, independent training phases, optimizer restoration, batched/single-game policy equivalence, search-tree restoration, cold-cache negamax equivalence, and a real interrupt/resume comparison with uninterrupted batched training.

## Curriculum and learning-progress experiments

New runs use `--curriculum races`. Each stage lasts `--curriculum-interval 25`
iterations. Race tasks start on an empty board, with distinct nonterminal pawn
squares at varied distances from their goals and a seeded side to move:

| Stage | Wall inventory per player | Maximum starting distance to goal |
| --- | ---: | ---: |
| Iterations 1–25 | 0 | 2 rows |
| Iterations 26–50 | 2 | 4 rows |
| Iterations 51–75 | 5 | 6 rows |
| Iterations 76+ | target (default 10) | 8 rows |

One pair in four always starts at the standard initial pawn positions with the
full target wall inventory. Other pairs sample uniformly from all stages unlocked
so far, retaining easier races later in training. Both arms see the same seeded
starting-position pairs. These are synthetic training positions, not a claimed
sample of naturally reached endgames. The curriculum is time-based, not an
adaptive guarantee of mastery; check goal rate and evaluation strength at each stage.

`--walls 2` caps all training tasks and evaluation at two walls per player.
That experiments with a simpler game; evaluation results cannot be compared
directly with ten-wall results. For learning standard Quoridor, leave `--walls 10`
and let the curriculum introduce walls gradually. `--curriculum none` disables
race starts while retaining the chosen wall inventory.

Start a separate experiment, preserving the original run:

```sh
.venv/bin/python training/train.py --out training/runs/curriculum-64 \
  --iterations 200 --games 128 --parallel-games 128 --batch-size 256 \
  --simulations 64 --curriculum races --curriculum-interval 25 --walls 10 \
  --eval-games 20 --eval-every 10 --heldout-games 40 \
  --engine direct --workers 4 --device mps --mps-inference hybrid \
  --checkpoint-seconds 60
```

Resume restores the experiment's exact curriculum and evaluation settings. New
learning settings require a new run; changing defaults never changes an old run.
The dashboard labels artificial cutoffs `C` and shows recent training goal and
repetition rates. `summary.json` separates monitoring and held-out evaluation.
A short test establishes correctness, not improved strength: compare full-wall
win rate, goal rate, repetition rate, and strength gained per wall-clock hour.

## Network depth experiments

See [the depth comparison](DEPTH_COMPARISON.md) for measured target fitting,
inference costs, and playing-strength results.
The residual variants fit saved policies better, but did not outperform the
original model in the two-seed equal-time playing tests. Keep `small` for normal
runs unless deliberately running a longer architecture experiment.

Choose the architecture when creating a new run:

| Flag | Spatial layers | Parameters |
| --- | --- | ---: |
| `--architecture small` (default) | Two 32-channel 3×3 convolutions | 370,866 |
| `--architecture residual32` | 32-channel stem + four two-convolution residual blocks | 435,602 |
| `--architecture residual64` | 64-channel stem + four two-convolution residual blocks | 991,442 |

All variants retain the 128-unit dense layer and the same policy/value heads,
input encoding, loss, and optimizer settings. Residual blocks use
`ReLU(x + 0.5 * conv2(ReLU(conv1(x))))`, without batch normalization or dropout.
The 32-channel variant tests added depth with only 17% more parameters; the
64-channel variant also increases width. The spatial receptive field before
flattening grows from 5×5 to 19×19. The original dense layer already sees the
entire board; this is a change in spatial processing, not access to the board.

Hybrid MPS inference uses architecture-specific CPU batch limits measured on
this Mac: 16 for `small`, 4 for `residual32`, and 2 for `residual64`. Larger
batches go to MPS. The old cutoff of 16 was substantially slower for residual
models. See [crossover measurements](measurements/depth-inference-crossover.json);
these thresholds may need retuning on other hardware.

Add `--architecture residual32` or `--architecture residual64` to a **fresh**
training command with a new output directory. Checkpoints and bot exports save
the architecture. Resume, bot commands, the dashboard's bot loader, and automatic
device selection use the saved model shape. Old checkpoints default to `small`;
architecture cannot change on resume.

Run the controlled comparison:

```sh
.venv/bin/python training/benchmark_depth.py --out training/runs/depth-comparison-new \
  --source training/runs/both-values-full-curriculum/latest.pt \
  --device mps --seeds 1 2 --updates 500 --seconds 120
```

The frozen-replay experiment splits by game pair, removes exact validation
states from training, and fits each model on identical minibatches. The source
must have uniform replay and both-turn value supervision so game boundaries can
be reconstructed and verified. The second experiment trains each architecture
from scratch on standard two-wall games with a 120-second process budget, then
evaluates both arms at 64 simulations against the same 2,000-node opponent on
paired race-1, standard two-wall, and standard ten-wall openings. The process
budget includes initialization and periodic saves; the graceful final save and
post-training evaluation occur afterward. An interrupted move batch can overrun
the nominal deadline slightly. Actual updates/games and timing are recorded.
These two-minute runs are bounded diagnostics, not converged strength estimates.
Use `--phase frozen` or `--phase online` to run just one part.

For paired direct matches against the original architecture, use:

```sh
.venv/bin/python training/benchmark_depth.py --out training/runs/depth-matches \
  --phase matches --trained-runs training/runs/depth-comparison-new \
  --device mps --seeds 1 2 --eval-games 20
```

Each residual model plays the matching seed/arm's small model on standard
two-wall openings, swapping colors within pairs. Both sides use 64 simulations
and a fresh tree per move, preventing reuse of another network's cached values
and priors. Cutoffs are reported separately from wins/losses.


## Board representation experiments

Fresh runs also accept these experimental architecture names:

| Architecture | Representation and network |
| --- | --- |
| `edges` | Original nine planes plus four directed blocked-edge planes, including board boundaries; same two-layer CNN |
| `lattice` | 17×17 grid with pawns on even-coordinate squares and walls spanning the gaps; square-mask plane, stride-two first convolution, then the same CNN head |
| `mlp` | Original nine planes flattened into a 384→128 dense network |
| `small` | Unchanged baseline |

All retain absolute coordinates, 209 policy outputs, outcome targets, and the
same optimizer. Geometry is derived inside the model, so existing replay data,
native search encoders, bot exports, and resume use the same input format.
The edge representation adds wall geometry rather than replacing wall anchors.
The lattice changes both geometry and first-layer sampling; it is not a pure
encoding-only ablation. Parameter counts are reported by the benchmark.
New variants use GPU-only inference in hybrid MPS mode until their crossovers
are measured.

Compare on identical frozen replay minibatches and game-pair validation splits:

```sh
.venv/bin/python training/benchmark_depth.py   --out training/runs/representation-comparison --phase frozen   --architectures small edges lattice mlp --device cpu --seeds 1 2   --updates 500
```

Use `--phase online` for equal-time training followed by playing evaluations,
or `--phase matches --trained-runs PATH` for paired matches against small.
Pass the same architecture list in each phase; matches require a trained small
baseline. Run timing comparisons without another trainer. Frozen target fitting
measures generalization to saved search targets, not playing strength.

## Google Colab and larger CUDA models

Open [colab.ipynb](colab.ipynb) in Colab and select a GPU runtime. From this
workspace run `python3 training/prepare_colab.py`, then upload
`training/runs/colab-source.zip` in the notebook. The bundle includes current
uncommitted training source; no push, Node, or web dependencies are needed for
the direct C++ backend. Keep Colab's installed CUDA-enabled PyTorch rather than
installing the locally pinned `requirements.txt`. The notebook builds the native
engine, mounts Drive, measures models, and runs controlled learning comparisons.

New residual variants retain the existing input and heads, adding width/depth:

| Architecture | Channels | Residual blocks |
| --- | ---: | ---: |
| `residual128` | 128 | 8 |
| `residual256` | 256 | 12 |
| `residual512` | 512 | 16 |

Choose `--precision bf16` for CUDA autocast on a GPU with BF16 support, or keep
`--precision fp32` (default). Weights and optimizer state remain FP32; network
outputs, policy softmax, and losses use FP32. BF16 needs no loss scaler. Precision
is stored in checkpoint/export config and restored on resume; CPU/MPS inference
uses FP32, including for exported BF16 runs. Precision changes can affect search
and learning, so use the same precision across a controlled architecture study.
Existing architectures and old checkpoints retain their shapes and FP32 behavior.

Measure hardware before choosing training batches:

```sh
python training/benchmark_cuda.py --out training/runs/cuda-sweep.json \
  --architectures small residual64 residual128 residual256 residual512 \
  --batches 32 128 512 --precisions fp32 bf16
```

The sweep records GPU/PyTorch details, parameter counts, synchronized inference
round-trip and actual replay-update timings, and peak CUDA allocated/reserved
memory. Each case runs in a fresh process; CUDA OOM and unsupported BF16 are
recorded, and results are saved after each case. Warmup is excluded. Synthetic
inputs/targets are identical across architectures: this is an execution test,
not evidence of stronger play. The reported memory is for **one** model; training
keeps two models and AdamW states resident, plus CPU replay and search trees.
Leave headroom and validate with a short real training run before a long sweep.

MCTS inference batch size is bounded by active concurrent games, not optimizer
batch size. Start with `--games 128 --parallel-games 128 --batch-size 256
--engine direct --workers 2 --device cuda`, then measure. Increase concurrent
games if GPU inference is underfilled, while checking CPU/RAM use. More workers
than available CPU cores can hurt. Large GPUs cannot accelerate native search,
negamax or replay bookkeeping. `performance.json` separates these costs.

For learning comparisons, keep games, seeds, search budgets, updates, minibatch
size, curriculum, precision, and evaluation openings fixed across models.
Compare playing score, goal/cutoff rates, completed games/updates and wall time
for each arm. Use several paired seeds. An equal-work experiment measures the
benefit of extra capacity for a given training workload; an equal-time experiment
measures the benefit under a compute budget. Run both as separate experiments.
Frozen-replay fitting via `benchmark_depth.py --phase frozen` remains available
for all new architectures (FP32), with the existing game-pair split safeguards.

The notebook saves training directly to Drive so completed checkpoints survive
runtime deletion. Checkpoint I/O can be significant for large models and is part
of the observed runtime; keep save cadence/storage consistent. GPU availability
and runtime duration are not guaranteed. See the
[Colab FAQ](https://research.google.com/colaboratory/faq.html) and
[PyTorch AMP guidance](https://docs.pytorch.org/docs/stable/amp.html).

### Longer Colab curriculum and early repetition experiment

Use `colab_curriculum_repetition.ipynb` with a newly generated
`python3 training/prepare_colab.py` bundle. It extends small/residual128/residual256
threefold runs to 200 iterations and creates separate `-repeat2` runs from scratch
at the same seed/settings. Defaults use seed 1; add seeds 2 and 3 for replication.
Training now accepts `--training-repetition-limit 2` (default 3). Two means stop
at the first revisit of the exact position including side to move. Goal wins
still take precedence. Cutoffs receive zero value and are recorded as repetition.
The setting is saved and cannot change on resume. Old checkpoints default to 3.
Evaluation and hypothetical MCTS branches retain threefold; this experiment
changes actual training rollout adjudication only, not the search rule.

In the downloaded completed Colab runs, 13.2–24.8% of selfplay games ended in
repetition, versus at most 0.1% of negamax-training games. This supports testing
savings in selfplay, but does not measure the saved time or establish improved
learning: some second occurrences could otherwise escape rather than repeat
again. Compare full-wall goal wins, cutoff rate, plies, and wall time against
the threefold control. Both variants keep zero-valued cutoff targets, which can
still reinforce draw-seeking behavior. The notebook streams output at most once
every two seconds and forwards notebook interruption as SIGINT for checkpointing.
