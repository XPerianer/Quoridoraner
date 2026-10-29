# Network depth comparison

Three architectures were compared on this Apple Silicon Mac using MPS, float32,
and the direct native engine. The existing `rolling-fixed-01` checkpoint was
not used as an experimental output. Its training settings and architecture
remain unchanged. The original CNN remains the default.

| Architecture | Convolution layers | Channels | Parameters |
| --- | ---: | ---: | ---: |
| `small` | 2 | 32 | 370,866 |
| `residual32` | 9: stem + four residual blocks | 32 | 435,602 |
| `residual64` | 9: stem + four residual blocks | 64 | 991,442 |

All models retain a 128-unit dense layer and the same policy/value heads,
encoding, optimizer, losses, and gradient clipping. Residual branches use a
factor of 0.5 and no normalization or dropout. `residual32` adds depth with 17%
more parameters; `residual64` changes both depth and width. This is not an exact
parameter-matched ablation. The dense layer in the original model already
combines information across the whole board.

## Fixed replay data

Source: the completed 200-iteration `both-values-full-curriculum` checkpoint,
with 10,000 replay rows per arm. The source hash and configuration are saved
in [the frozen-data measurements](measurements/depth-frozen.json).

Game boundaries were reconstructed from uniform replay's append order and
verified against each row's player, remaining-ply feature, and outcome. The
split holds out whole game pairs; paired colors cannot cross the split. Exact
validation-state duplicates were removed from training as an additional check.
This left 7,661 training / 1,889 validation rows for selfplay and 4,362 / 1,875
for negamax. The duplicate removal excluded 450 and 3,763 training rows,
respectively. Validation contains related positions from each held-out game;
its rows are not independent statistical trials.

Each architecture starts from scratch and receives the same 500 minibatches
of 128 rows for each seed and arm. Seeds 1 and 2 were used. The target policies
are the saved MCTS policies, not an external expert or proof of optimal moves.
Thus better validation loss establishes better target fitting, not necessarily
stronger play.

Mean validation metrics across the two initialization seeds (lower is better):

| Architecture | Selfplay policy KL | Selfplay value MSE | Negamax policy KL | Negamax value MSE |
| --- | ---: | ---: | ---: | ---: |
| small | 0.869 | 0.111 | 2.024 | 0.448 |
| residual32 | 0.674 | 0.116 | 1.771 | 0.565 |
| residual64 | 0.624 | 0.099 | 1.458 | 0.440 |

The 32-channel residual model reduces policy KL by 22% on selfplay data and
12% on negamax data. The 64-channel model reduces it by about 28% on both.
Value fitting does not improve uniformly: in particular, residual32 has worse
negamax validation value error. These comparisons match update counts, not
training time. Mean 500-update fitting times per arm were about 5.0–5.4 seconds
for small, 5.6–6.5 for residual32, and 9.4–9.9 for residual64.

## Inference settings

The old hybrid cutoff of 16 positions was tuned for the small network. At
batch 16, CPU inference took about 0.54 ms for small, 3.07 ms for residual32,
and 11.00 ms for residual64; the corresponding MPS measurements were about
0.79, 1.49, and 1.20 ms. Reusing the old cutoff would unfairly penalize depth.

Hybrid inference now sends batches up to 16 / 4 / 2 positions to CPU for
small / residual32 / residual64, respectively. Larger batches use MPS.
The small-model behavior is preserved. See
[the measured crossover](measurements/depth-inference-crossover.json).
Hardware-specific thresholds should be remeasured on another machine.
Reproduce with `.venv/bin/python training/benchmark_depth_inference.py`.

## Equal-time training protocol

Each architecture is trained from scratch for a nominal 120-second process
budget, for each of two seeds. This includes initialization, two baseline
evaluation games per arm, and periodic checkpointing. The trainer then receives
SIGINT, finishes its current move/update transaction, and saves. Final saves
and the subsequent evaluation are outside that budget; reported process wall
seconds include the final save. Completed games and optimizer updates vary.

Training uses standard two-wall starts, no curriculum, 32 games per arm per
iteration, 32 concurrent games, 64 simulations, 32 updates, and minibatches of
128. Both-turn value targets and rolling scheduling are enabled. This is a
controlled two-wall diagnostic, not the same curriculum as the user's full
run and not a converged strength estimate. Networks/learning rates were not
separately tuned.

Evaluation uses 40 paired games per suite and arm, seed 917, 64 simulations,
a 2,000-node fixed negamax opponent, no exploration noise, and no sampled moves.
The suites are race-1, standard two-wall, and standard ten-wall. The ten-wall
suite tests transfer beyond the training wall inventory. Variant order is
reversed on seed 2. All final timed comparisons ran sequentially while the
original trainer was paused; earlier overlapping results were discarded.
The original trainer was resumed during the later fixed-search-budget direct
matches, for which no throughput or elapsed-time comparison is claimed.

### Timed results

Each seed received the same nominal 120-second process budget. Final save and
shutdown brought total process wall times to 121.3–121.6 seconds. The number of
completed optimizer updates **per arm** was:

| Architecture | Seed 1 updates | Seed 2 updates | Mean updates |
| --- | ---: | ---: | ---: |
| small | 704 | 736 | 720 |
| residual32 | 320 | 320 | 320 |
| residual64 | 160 | 224 | 192 |

Goal wins against fixed negamax, pooled across two seeds (80 games per cell):

| Architecture | Selfplay race-1 | Selfplay standard 2 walls | Selfplay standard 10 walls | Negamax race-1 | Negamax standard 2 walls | Negamax standard 10 walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 2/80 | 5/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| residual32 | 2/80 | 4/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| residual64 | 1/80 | 2/80 | 0/80 | 1/80 | 0/80 | 0/80 |

The original selfplay model's standard two-wall wins were 0/40 and 5/40 by
seed; residual32 had 3/40 and 1/40; residual64 had 0/40 and 2/40. The first
residual32 seed looked better, but that did not hold when both seeds were
considered. Sample sizes are small, and these results do not establish a
statistically reliable ordering among all architectures.

Full commands, realized training games/updates, inference counters, individual
evaluation outcomes, and cutoff counts are saved in
[the online measurements](measurements/depth-online.json).

## Direct matches

To avoid relying exclusively on a stronger fixed opponent, each residual model
also played its matching seed/arm's small model. Each matchup used 20 paired
standard two-wall games per seed, 64 simulations for both sides, deterministic
move selection, and a fresh tree each move. Fresh trees prevent one network
from inheriting another network's cached priors and values. These search
semantics differ from the retained-tree fixed-negamax evaluations above.

Counts below are from the residual model's perspective, pooled across two
seeds. Cutoffs are artificial adjudications, not Quoridor draw rules.

| Residual model | Training arm | Wins | Losses | Cutoffs | Score with half-credit cutoffs |
| --- | --- | ---: | ---: | ---: | ---: |
| residual32 | selfplay | 13 | 17 | 10 | 45.0% |
| residual64 | selfplay | 5 | 18 | 17 | 33.75% |
| residual32 | negamax | 0 | 0 | 40 | 50.0% |
| residual64 | negamax | 1 | 0 | 39 | 51.25% |

The negamax-arm matches are almost entirely cutoffs and provide little evidence
of relative strength. The selfplay matches do not support switching to a deeper
model under this compute budget. See [match measurements](measurements/depth-matches.json).
The evaluator's paired-color behavior was additionally checked using identical
networks on an immediate-goal fixture: exactly one win and one loss.

## Conclusion and implementation

**Depth improves policy-target fitting, but did not improve playing strength
per unit training time in this bounded experiment. Keep `small` as the default.**
The deeper models generated less experience and completed fewer updates.
Their lower frozen-replay error did not translate into a clear strength gain.
This does not show that depth can never help: longer budgets, the gated
curriculum, pretraining, or different architecture/optimizer tuning could change
the result. Those combinations were not tested here.

`--architecture residual32` and `--architecture residual64` remain available
for fresh experiments. Existing checkpoints default to small when architecture
metadata is absent. Resume restores the saved architecture; it cannot be changed
in place. Bot exports, bot CLI, dashboard bot loading, automatic device selection,
and the profiling utility use the saved architecture.

The full regression suite ran 62 tests with one CUDA skip before the hybrid
cutoff change. All 17 direct-engine tests passed afterward, including the new
residual CPU/MPS consistency test. Coverage includes legacy checkpoint layout,
residual gradients and exports, and exact CPU interrupted/resumed training.
The residual export also passed a real bot-CLI move smoke check.

Reproduce the fitting and timed tests:

```sh
.venv/bin/python training/benchmark_depth.py \
  --out training/runs/depth-comparison-repeat \
  --source training/runs/both-values-full-curriculum/latest.pt \
  --device mps --seeds 1 2 --updates 500 --seconds 120
```

Then the direct matches:

```sh
.venv/bin/python training/benchmark_depth.py --phase matches \
  --trained-runs training/runs/depth-comparison-repeat \
  --out training/runs/depth-matches-repeat --device mps --seeds 1 2 --eval-games 20
```

The timed-run checkpoints are intentionally paused; their large iteration
limit lets the harness stop by time rather than iteration count.
