# Colab and local playing-strength screening

Reference: `both-values-full-curriculum`, iteration 200. Each candidate plays the matching reference arm (selfplay or negamax). All matches use FP32 on MPS, 64 simulations per move, fresh search trees between network turns, 20 games on ten paired ten-wall openings (seed 917), and a 100-ply limit. Repetition and move-limit endings count as cutoffs, worth half a point. This is an equal-search-budget comparison of existing checkpoints, not equal training compute.

| Candidate | Arm | Export iteration | Wins | Losses | Cutoffs | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual128-bf16-seed1 | selfplay | 20 | 1 | 17 | 2 | 10.0% |
| residual256-bf16-seed1 | selfplay | 20 | 0 | 20 | 0 | 0.0% |
| residual64-bf16-seed1 | selfplay | 20 | 0 | 18 | 2 | 5.0% |
| small-bf16-seed1 | selfplay | 20 | 0 | 19 | 1 | 2.5% |
| rolling-fixed-01 | selfplay | 532 | 0 | 10 | 10 | 25.0% |
| lattice-fixed-01 | selfplay | 379 | 4 | 10 | 6 | 35.0% |
| gated-balanced-full | selfplay | 170 | 0 | 17 | 3 | 7.5% |
| curriculum-64-u32 | selfplay | 200 | 4 | 15 | 1 | 22.5% |
| residual128-bf16-seed1 | negamax | 20 | 1 | 2 | 17 | 47.5% |
| residual256-bf16-seed1 | negamax | 20 | 2 | 6 | 12 | 40.0% |
| residual64-bf16-seed1 | negamax | 20 | 1 | 3 | 16 | 45.0% |
| small-bf16-seed1 | negamax | 20 | 0 | 1 | 19 | 47.5% |
| rolling-fixed-01 | negamax | 532 | 6 | 1 | 13 | 62.5% |
| lattice-fixed-01 | negamax | 379 | 2 | 3 | 15 | 47.5% |
| gated-balanced-full | negamax | 170 | 1 | 5 | 14 | 40.0% |
| curriculum-64-u32 | negamax | 200 | 0 | 3 | 17 | 42.5% |

## Recorded Colab held-out evaluations

All completed runs are seed 1, iteration 20, 128 games and 32 optimizer updates per arm per iteration. Each final held-out suite uses 40 games against the 2,000-node negamax engine. These recorded tests used CUDA BF16; the new matches above use FP32 on both sides.

| Model | Arm | Wins | Losses | Cutoffs |
| --- | --- | ---: | ---: | ---: |
| residual128-bf16-seed1 | negamax | 0 | 40 | 0 |
| residual128-bf16-seed1 | selfplay | 0 | 40 | 0 |
| residual256-bf16-seed1 | negamax | 0 | 39 | 1 |
| residual256-bf16-seed1 | selfplay | 0 | 37 | 3 |
| residual64-bf16-seed1 | negamax | 0 | 36 | 4 |
| residual64-bf16-seed1 | selfplay | 0 | 39 | 1 |
| small-bf16-seed1 | negamax | 0 | 39 | 1 |
| small-bf16-seed1 | selfplay | 0 | 40 | 0 |

## Interpretation

- No completed Colab model won a final held-out game against the fixed negamax engine. Larger capacity has not translated into demonstrated full-game strength in this pilot.
- All Colab runs stopped before the first scheduled curriculum transition at iteration 25. Race-0 positions dominate, with one standard-start pair in four. The local reference received 200 iterations; these results do not isolate architecture quality.
- The downloaded residual512 run reaches iteration 7 and has no selfplay.pt or negamax.pt export. It is excluded from head-to-head comparisons. Only seed 1 was downloaded; there is no multi-seed replication.
- The local reference selfplay bot won the match score against every screened candidate. The rolling negamax bot performed better than the reference negamax bot in this sample, but many games cut off. Neither result proves an overall ranking across all opponents.
- Twenty games contain only ten distinct color-paired openings. Small differences and cutoff-heavy scores need larger independent suites. No Elo is inferred. Match seeds differ from recorded training evaluations.
- Existing zero-wall and two-wall diagnostic scores cannot be ranked directly against ten-wall scores. Recorded evaluations were inventoried in recorded-evaluations.json; selected mature full-wall runs were screened directly.

## Next experiment

Continue a small baseline and residual128/256 through the full curriculum on paired seeds before adding more capacity. Retain equal games, update counts, search budgets and evaluation schedules. Track full-wall goal wins separately from cutoff score. Run larger fresh-opening matches for the local selfplay reference and rolling negamax candidate before selecting a champion.

Source artifacts: matches.json contains all screening game scores/lengths and checkpoint paths; recorded-evaluations.json contains the inventory of saved evaluations. Reproduce with `python training/analyze_colab_matches.py` (requires MPS; completed candidates are skipped).
