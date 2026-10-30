# Profile of the `rl` loop on one L4 (2026-10-29)

Colab L4, 12 vCPUs, torch 2.11. Final Colab checkpoint (128×10). Real `az.py rl`
runs from that checkpoint: 200 simulations, half the games against negamax at
2,000 nodes, batch 1,024, reuse 4, replay 600,000, eval every 3 iterations (40
games). `profile_rl.py` wraps the pipeline functions with timers and samples
`nvidia-smi` every 0.5 s. Iteration 1 includes compilation and is excluded.

| | Before (`before-1024.json`) | After, 1,024 games (`after-1024.json`) | After, 2,048 games (`after-2048.json`) |
|---|---:|---:|---:|
| Games finished per iteration | 1,024 | ~1,030 | ~2,050 |
| Self-play seconds | 76 | 33 | 62 |
| Training seconds | 20 | 16 | 30 |
| Iteration seconds | 96 | 49 | 93 |
| Self-play ms per game | 74 | 32 | 30 |
| GPU busy during self-play | ~45% | — | 61% |
| Eval match (40 games) | 26 s | — | 7.4 s |

Where the old time went: GPU and CPU took turns (GPU ~50% busy mid-iteration),
and the last ~20 s of each iteration ran a few long games with nearly empty
batches.

Changes, with their measurements:

- Two alternating search groups: CPU search overlaps GPU inference.
- Inference copy (`gpu-variants.json`, `compile-modes.json`): BatchNorm folded,
  bf16 channels-last weights and `torch.compile(dynamic=True)` reach 79k
  evaluations/s on mixed batch sizes, against 40k for the previous autocast
  model. Batches above ~512 rows are slower on the L4.
- CUDA graphs (`launch-cost.json`): queueing a batch of 352 rows cost 3.0 ms of
  CPU for the compiled model and 0.5 ms as a graph replay, at equal GPU speed.
- Continuous games across iterations remove the tail.
- Training in channels-last: 85 → 65 s per 1,000 steps. Fused AdamW and
  compiling the training step gave nothing further.

Remaining time per 2,048-game iteration (`after-2048.json`, per iteration):

| | Seconds |
|---|---:|
| Native waves and queueing (`launch`) | 22 |
| Waiting for and applying results (`collect`) | 11 |
| Negamax opponent moves (blocks the loop) | 7 |
| Python per-move bookkeeping (advance, begin, noise, take) | 15 |
| Training | 30 |
| Checkpoint and model save | 3 |
| Eval, every `--eval-every` iterations | 7 |

The self-play loop was then limited by its single Python thread.

## Native bookkeeping and background negamax (`after-native-2048.json`)

- Game histories live in C++ next to the search trees; moves are applied, and
  finished searches read out, in one native call per loop pass. Root noise is
  drawn natively from a per-move seed instead of per-move Python
  `gammavariate` calls.
- Negamax replies run on their own native pool and thread, and are applied on
  the next loop pass, so results stay deterministic.

On a Mac CPU (tiny net), loop time outside the network fell from 79 to 27 µs
per move. On the L4, at 2,048 games:

| | Before (`after-2048.json`) | After (`after-native-2048.json`) |
|---|---:|---:|
| Self-play seconds per iteration | 62 | 51.5 |
| Iteration seconds | 93 | 84 |
| GPU busy during self-play | 61% | 84% |
| Eval match (40 games) | 7.4 s | 6.1 s |

Self-play is now mostly GPU-bound. Training (31 s) is the largest remaining
block; only overlapping it with self-play would hide it.

## Overlapping training with self-play: rejected

Prototype (`overlap-prototype.patch`, against `91e49ac`; not merged): an
`--overlap` flag trained on each iteration's games on a background thread,
its own CUDA stream and its own native feature pool, while the next iteration
played with the previous weights. Weights were swapped at iteration ends, so
games lagged the trained network by one iteration. Runs stayed deterministic
across worker counts and after a resume. CUDA graph capture had to become
thread-local.

Same L4 session, 2,048 games, no eval, 5 iterations each
(`profile_rl.py 2048 200 5 0 <tag> [--overlap]`):

| | Off (`overlap-off-2048.json`) | On (`overlap-on-2048.json`) |
|---|---:|---:|
| Iteration seconds, iterations 2–5 | 81–83 | 82–89 |
| Self-play seconds per iteration | 50 | 81–88 |
| Waiting for training at the iteration end | — | ~0 s |
| GPU busy (nvidia-smi) | 89% playing, 91% training | 99% while both ran |
| Wall time for 5 iterations | 467 s | 464 s |

Training finished well inside each rollout, but slowed the rollout by more
than its own 32 s. The GPU is the bottleneck in both phases: about 73 s of GPU
work per iteration bounds any overlap at ~11% on one L4, and training kernels
competing with search batches delay the searches, which then leave the CPU
waiting. Overlap could pay off only with spare GPU capacity, such as a second
GPU for training.
