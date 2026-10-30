# Native lookup and continuous scheduling on one L4 (2026-10-29)

Colab L4 (24 GB), 12 vCPUs, torch 2.11. Final Colab checkpoint (128×10), pure
self-play from random openings (0–8 plies), Dirichlet noise, temperature for
8 plies, 100-ply limit. `before.json`: source before this change (Python
blake2b lookup, lockstep moves). `native.json`: C++ lookup checked during
selection plus continuous per-game scheduling. Both ran without
`cudnn.benchmark`, the same as `az.py`. Script: `l4_bench.py`.

| Sims/move | Lookup | Before (games/h) | After (games/h) | Speedup |
|---:|---|---:|---:|---:|
| 200 | off | 27.6k | 40.9k | 1.48× |
| 200 | 1M entries | 24.6k | 51.4k | 2.09× |
| 800 | off | 7.3k | 9.1k | 1.25× |
| 800 | 1M entries | 5.9k | 13.8k | 2.33× |

Best configuration versus the best previous one: 1.86× at 200 simulations,
1.89× at 800. The previous Python lookup was slower than none.

2,048 games at 800 simulations: 12.6k games/h (batches above 1,024 are slower
on this GPU). Training: 85 s per 1,000 steps at batch 1,024.

## SigmaQuoridor comparison (reported, not re-measured)

Sigma: 800 simulations, 2,048 games per cycle, ~24k games/h, 105 s per 1,000
steps, on a Ryzen 7800X3D and RTX 5070 Ti. Its README gives ~90k evaluations/s
on that GPU. We measured 42–50k/s for our network on the L4 at batch 1,024.

At 800 simulations we reach 57% of Sigma's games/h. Our run sustains 34k
evaluations/s, about 70–80% of the L4 ceiling. Sigma sustains ~25k/s, roughly
28% of its ceiling. A Sigma-shaped cycle (2,048 games, 1,000 steps) would take
about 10.3 min here, against 6.7 min reported. That is 55 h for Sigma's 321
cycles, against 35.7 h.

Differences remain: 100- versus 200-ply limits, different evaluation-cache
policies, and Sigma's endgame solver.
