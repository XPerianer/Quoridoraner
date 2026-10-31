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
