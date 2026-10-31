# L4 RL ablation results

## Recommendation for the full L4 run

The best candidate from this screen is **AdamW at constant 3e-4 with randomized
playout caps: 75% at 40 and 25% at 400 simulations per learning move**. Keep the
128×10 network, batch 1,024, replay 600k, reuse 4 and existing opponent mix.
Use `--playout-cap randomized-40-400 --optimizer adamw --schedule none` on the
existing RL command. Low-cap moves supply value targets; high-cap moves supply
policy and value targets. Keep the existing legacy endgame search for now.

This is a provisional choice for the user to make, rather than proof of the best
long-run setup. The evidence favors it when strength against another trained
network is the priority. The existing fixed-200 AdamW baseline remains a reasonable
choice when fixed-negamax performance or conservative extrapolation matters more.

| Requested change | Measured benefit/cost | Decision |
|---|---|---|
| Muon + cosine/warm restarts | Muon costs 5.2% more per optimizer update. Flat, cosine and restart Muon models scored 44.5%, 46.75%, and 44.75% head-to-head against baseline; all intervals include 50%. No demonstrated strength-per-hour gain or 30–40% step reduction. | Keep AdamW. |
| 75% cap 40 / 25% cap 400 | Head-to-head 56.0% [51.0, 61.0] in screening; 55.5% [51.5, 59.5] on 400 fresh-opening games. About 11.3% fewer games per training-hour and 12.7% fewer total updates in the budget. | Best strength candidate, not a speed optimization. |
| Exact zero-wall bitboard tablebase | Isolated zero-wall search was 1.73× faster, but full RL throughput was 64.1% lower, with 12 iterations versus baseline's 33. Head-to-head score was 23.75% [18.25, 29.25]. Solver-on/off score against negamax-32000 was identical in the sampled 200 games. | Do not enable it for the full run in its current form. |

The simpler bounded proof extension was screened out before a full training run:
it reduced neural evaluations but made the isolated search batch 37× slower.
The exact tablebase is a distinct implementation and received the full RL test.

**Important opponent tradeoff.** On fresh seed-2028 openings, baseline scored
91.25% against negamax-32000 and playout randomization scored 88.25%. Thus the
head-to-head advantage does not imply uniform superiority against all opponents.
The confirmation reuses the same trained weights with new openings; it does not
measure variance between training seeds. Neither this short run nor these
intervals establish a long-run learning-speed multiplier. Do not add benefits
from separately tested optimizer, schedule, cap and solver changes.

## What was implemented and verified

The requested cheap-model implementation used `gpt-6-luna`, followed by parent
review and fixes. All changes are opt-in. Muon operates on flattened trunk
convolution matrices with AdamW-comparable RMS scaling; BatchNorm, biases and
heads use AdamW. Cosine and warm-restart schedules advance per update and restore
the live LR on resume. Playout cap draws use checkpointed per-game RNG streams.

The exact tablebase uses bitboard pawn moves including straight and blocked
jumps, a fixed-wall product graph, repetition-aware draw sinks, distance-to-mate
retrograde labels and the true remaining-ply cutoff. It is separate from the
budget-limited proof search. Exhaustive finite-horizon checks cover all 12,960
non-colliding pawn/turn states through horizons 0–100 on open and placed-wall
boards, plus history-aware brute minimax and integrated search comparisons.
The complete 40-test AZ suite passed locally and on the second L4.

Eight configurations received approximately 45 training minutes each: baseline,
AdamW cosine/restarts, Muon flat/cosine/restarts, randomized caps, and exact
tablebase. The bounded proof extension received the rejection microbenchmark.
All trained model artifacts are backed up under
`experiments/training/az/ablations-20261030/models/`. Raw metrics, paired-game
outcomes, bootstrap intervals, source hashes, software environments and replay
fingerprints accompany this report. The replay arrays on the two L4s were
verified identical. One original VM was lost; its missing Muon restart arm was
rerun from the common initial checkpoint on the surviving L4. Cross-VM speed
variation is an additional caveat.


The network is unchanged (128 channels, 10 residual blocks). These are optimizer and search ablations.

Budget: 45 minutes of self-play and training per arm; one training seed (1). Schedules use 15,000 updates. Evaluation uses 100 paired openings.

## Training throughput

| Arm | Iterations | Actual min | Games | Positions | Updates | Games/s | Median iteration s |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 33 | 45.80 | 67,823 | 4,097,915 | 15,991 | 24.68 | 82.0 |
| adamw_cosine | 33 | 45.10 | 67,798 | 4,095,857 | 15,984 | 25.05 | 80.7 |
| adamw_restarts | 34 | 46.13 | 69,875 | 4,174,662 | 16,289 | 25.25 | 80.9 |
| muon_flat | 33 | 45.69 | 67,853 | 4,075,901 | 15,906 | 24.75 | 81.2 |
| muon_cosine | 33 | 46.48 | 67,814 | 4,105,558 | 16,019 | 24.32 | 83.3 |
| muon_restarts | 32 | 46.27 | 65,802 | 4,001,690 | 15,618 | 23.70 | 87.1 |
| playout_40_400 | 29 | 45.33 | 59,550 | 3,576,711 | 13,955 | 21.89 | 94.4 |
| race_solver | rejected on inference cost | | | | | | |
| tablebase | 12 | 46.42 | 24,651 | 1,585,761 | 6,187 | 8.85 | 239.9 |

## Held-out strength

Score = wins + half draws, divided by games. Intervals resample opening pairs, retaining both colors.

| Arm | Network vs N2000 | PUCT128 vs N2000 | PUCT128 vs N32000 | H2H vs baseline |
|---|---:|---:|---:|---:|
| baseline | 0.830 [0.782, 0.877] | 0.925 [0.890, 0.958] | 0.877 [0.835, 0.917] | — |
| adamw_cosine | 0.830 [0.775, 0.880] | 0.925 [0.890, 0.958] | 0.915 [0.877, 0.950] | 0.472 [0.420, 0.525] |
| adamw_restarts | 0.850 [0.805, 0.895] | 0.922 [0.887, 0.955] | 0.897 [0.855, 0.938] | 0.527 [0.477, 0.578] |
| muon_flat | 0.853 [0.805, 0.897] | 0.938 [0.902, 0.970] | 0.885 [0.843, 0.925] | 0.445 [0.388, 0.505] |
| muon_cosine | 0.815 [0.755, 0.870] | 0.922 [0.885, 0.955] | 0.912 [0.875, 0.948] | 0.468 [0.415, 0.520] |
| muon_restarts | 0.840 [0.787, 0.887] | 0.927 [0.890, 0.960] | 0.870 [0.823, 0.912] | 0.448 [0.388, 0.507] |
| playout_40_400 | 0.777 [0.720, 0.833] | 0.930 [0.895, 0.960] | 0.887 [0.843, 0.927] | 0.560 [0.510, 0.610] |
| race_solver | rejected on cost | rejected on cost | rejected on cost | rejected on cost |
| tablebase | 0.757 [0.705, 0.810] | 0.948 [0.915, 0.975] | 0.895 [0.850, 0.935] | 0.237 [0.182, 0.292] |

## Paired change against the stronger fixed opponent

Each difference compares the same openings and colors with baseline PUCT128 vs negamax-32000.

| Arm | Score change | Paired bootstrap 95% interval |
|---|---:|---:|
| adamw_cosine | +3.8 pp | [-0.8, +8.2] pp |
| adamw_restarts | +2.0 pp | [-3.0, +7.0] pp |
| muon_flat | +0.8 pp | [-4.0, +5.5] pp |
| muon_cosine | +3.5 pp | [-1.0, +7.5] pp |
| muon_restarts | -0.8 pp | [-6.0, +4.5] pp |
| playout_40_400 | +1.0 pp | [-4.2, +6.2] pp |
| tablebase | +1.8 pp | [-3.2, +6.8] pp |

## Optimizer overhead

Identical replay batches, fresh initial model, 20 warmup updates per repeat: AdamW 66.57 ms/update; Muon 70.03 ms/update (median, Muon/AdamW = 1.052).
Muon needs 4.9% fewer updates to offset this measured update-time ratio in training alone. This calculation does not establish that it actually needs fewer updates.

## Randomized cap diagnostics

Finished games contain 1,934,546 low-cap and 642,734 high-cap learning moves. Replay additions: 642,734 policy-bearing rows and 2,933,977 value-only rows (the latter also include negamax moves). Retained-tree visits count toward caps; 130 nominal simulations per move is not a measurement of fresh network work.

## Solver inference microbenchmark

128 zero-wall replay positions, 128 simulations each: median legacy 0.146s, extended proof 5.450s. Speed ratio legacy/extended = 0.027.
These isolated queries use the replay row’s current-position repetition count and otherwise fresh history; the full training arm uses complete histories. This measures runtime, not playing strength.

## Exact tablebase inference

Median search batch: legacy 0.148s; tablebase 0.085s. Legacy/tablebase speed ratio 1.732.
This is a separate implementation from the bounded proof extension. It solves the fixed-wall pawn graph with repetition-aware draw sinks and exact distance-to-mate cutoff handling. Source provenance is in `tablebase/source.json`.

## Solver enabled during play

Same tablebase-trained model, PUCT128 vs negamax32000: solver off 0.895, solver on 0.895; paired change +0.0 pp [95% interval +0.0, +0.0 pp].

## Fresh-opening confirmation

Playout randomization vs baseline on seed 2028, 400 games: score 0.555, paired-bootstrap 95% interval [0.515, 0.595]. This uses the same trained weights with fresh openings; it is not a second training seed.
- baseline, PUCT128 vs negamax32000, seed 2028: 0.912.
- playout_40_400, PUCT128 vs negamax32000, seed 2028: 0.882.

## Limits of the estimate

- One training seed per arm: intervals cover opening variation, not variation between training runs.
- Multiple-arm screening creates selection bias. A small apparent winner needs an independent-seed confirmation.
- Short-run results do not establish the best long-run schedule; the final schedule horizon must match the final run.
- Training losses use different replay distributions and cannot substantiate a 30–40% validation-step reduction.
- All network comparisons use the baseline evaluation search. The isolated solver benchmark measures runtime, not playing strength.
- No combined optimizer + randomized-cap + solver arm has been tested; individual benefits cannot simply be added.

Exact arguments, source hashes, checkpoint hash and environment are in `plan.json` and `environment.json`.
