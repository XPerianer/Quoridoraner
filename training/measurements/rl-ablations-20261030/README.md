# RL optimizer, playout-cap, and pawn-race ablations

Status: complete. Eight configurations received full L4 training/evaluation runs;
the bounded proof extension was rejected on measured inference cost. All models
and raw results are saved locally. The 40-test AZ suite passed locally and on L4.
See `REPORT.md` for the recommendation, measured benefits, fresh-opening
confirmation and limitations.

Initial findings: Muon costs 5.2% more per update (median 70.03 vs 66.57 ms).
The bounded proof extension is rejected as a speed optimization: 5.45 vs 0.146 s
on the sampled zero-wall search batch despite fewer neural evaluations. A separate
exact tablebase implementation is queued for L4 evaluation; these proof timings do not measure it.
Muon flat scored 0.445 head-to-head against baseline (paired 95% interval
0.3875–0.505), so this screen has not demonstrated a Muon strength advantage.

The cheaper-model implementation agent used `gpt-6-luna`. Parent review corrected
optimizer-group rebinding, channels-last storage handling, LR restoration on
resume, and the solver CLI horizon before the L4 runs.

## Question and controls

Which changes improve strength per L4 training-hour before the full run?
These are optimizer/search changes; the network remains the same 128-channel,
10-block residual model. Every arm starts from
`experiments/training/az/colab/pre-model.pt`, with fresh optimizer state and the
same replay seed, generated from 30,000 negamax games (seed 1).

The screen allocates 45 minutes of self-play plus training per arm. It retains
the existing time-budget convention: stop starting iterations once the budget is
reached; include the final iteration's overshoot in the reported time. Evaluation
is outside the training budget. All arms use 2,048 game lanes, 50% negamax-2000
lanes, 600k replay, batch 1,024, reuse 4, seed 1, and AdamW base LR 3e-4.

| Arm | Change from baseline |
|---|---|
| baseline | PUCT 200, AdamW, constant LR |
| adamw_cosine | Cosine LR |
| adamw_restarts | Cosine LR with warm restarts |
| muon_flat | Muon trunk matrices; AdamW heads and vectors |
| muon_cosine | Muon plus cosine |
| muon_restarts | Muon plus cosine warm restarts |
| playout_40_400 | Per learning move: 75% cap 40, 25% cap 400 |
| race_solver | Opt-in zero-wall exact proofs, network fallback if not proved |

Schedules use a fixed 15,000-gradient-step horizon. The restart schedule begins
with a quarter-horizon cycle and doubles cycle length. Schedule settings and
states must survive checkpoint/resume. Muon updates must use AdamW-comparable
RMS scaling; retaining LR 3e-4 with unscaled Muon would not be a fair comparison.

Randomized caps retain tree reuse, as the existing baseline does. The weighted
nominal cap is 130, but that does **not** establish a 35% compute reduction:
retained visits, batch occupancy, training time, and policy-target density matter.
Low-cap moves contribute value targets only; high-cap moves contribute policy
and value targets. This is an explicitly chosen variant, not an assertion that
it reproduces every detail of KataGo.

The existing baseline already has a depth-4, 256-node pawn-only proof search at
unexpanded leaves. The solver arm must be compared against that baseline, not
against an imaginary baseline with no endgame handling. Position-only shortest
paths are insufficient because of jumps, side to move, threefold repetitions,
and the 100-ply cutoff. Solver outputs must be exact proofs; exhaustion must
return unknown and fall back to neural evaluation.

## Evaluation and interpretation

Each trained network plays 100 held-out opening pairs (200 games), seed 2027:
network-only versus negamax-2000, PUCT 128 versus negamax-2000, PUCT 128 versus
negamax-32000, and PUCT 128 head-to-head versus the baseline. Evaluation uses the
same search for every model to isolate learning effects. Separate solver-on/off
runtime measurements will assess its inference benefit.

Per-game outcomes are retained. Confidence intervals bootstrap opening pairs,
keeping the two colors together. One training seed is a screening experiment;
small differences, multiple arm selection, and a short training horizon limit
claims about the full run. Training loss across different replay distributions
is not a common validation metric. The proposed 30–40% Muon step reduction is a
hypothesis, not an input to the estimate.

`training/run_rl_ablations.py` writes exact arguments, source/checkpoint hashes,
hardware/software information, checkpoints, metrics, and evaluation files.
`training/ablation_eval.py` writes paired-game records and bootstrap intervals.
`training/benchmark_rl_optimizers.py` measures update overhead;
`training/benchmark_race_solver.py` measures isolated endgame search overhead.
`training/summarize_rl_ablations.py` produces `REPORT.md` from actual outputs.

Runtime: Colab session `rl-ablations`, `/content/q`, outputs `/content/ablations`.
The queue runs inside an active kernel supervisor: a detached subprocess alone
did not keep the first Colab VM alive. Results and model-only archives are written
to `/content/ablations-results.zip` and `/content/ablations-models.zip` after each
finished arm/evaluation; full resume checkpoints remain in the arm directories.

## Method references

- [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html):
  matrix restriction, AdamW for other parameters, and update scaling.
- [Microsoft Dion](https://github.com/microsoft/dion): experimental convolution
  support through flattening kernels to matrices.
- [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md):
  full/fast searches and recording full-search training targets.

## Exact tablebase follow-up

A separate `tablebase` search mode now solves the fixed-wall two-pawn graph.
It treats previously twice-seen positions as future draw sinks; winning and
losing distance-to-mate strategies decrease distance, so decisive proofs cannot
create a new threefold repetition. The current root is evaluated separately
when its prior count is two. True remaining plies convert outcomes beyond the
cutoff to draws. Goal precedence is preserved.

Validation includes all 12,960 non-colliding pawn/turn states at horizons 0–100
on open and placed-wall boards, checked against independent finite-horizon DP,
plus history-aware brute minimax samples and integrated MCTS comparison.
The full local AZ suite passes 40 tests. L4 validation and timing remain pending.
The follow-up uses `/content/q-tablebase`, preserving the original queue source;
its source hashes are saved in `tablebase/source.json`.

## Parallel runtime placement

At the user's request, the exact tablebase arm moved to a second L4 session,
`rl-ablations-2`. The original tablebase follow-up watcher was stopped before
launching the second VM's job. Each VM runs one experiment at a time. The
original eight-arm queue stays on `rl-ablations`. The second VM regenerates the
same 30,000-game seed-1 replay dataset and starts from the identical pretrained
checkpoint; its paired matches use a copy of the completed baseline checkpoint.
Cross-VM timing differences remain a screening caveat.


The first VM was lost during Muon warm restarts. That arm was recovered from the
common initial checkpoint on the second L4. Playout randomization was also moved
to the second L4 at the user's request. It completed once there; no completed
first-VM playout run is included. Fresh-opening confirmation (seed 2028, 400 H2H
games) scored 55.5% [51.5, 59.5] for playout randomization against baseline.
