# Gumbel AlphaZero implementation guide

Repository review: 2026-10-29. This is an implementation specification, not an implemented feature or a measured speedup. No experiments were run to prepare it.

**Status (2026-10-30): implemented** in `az.py` and `native/az.cpp` (both milestones: root search and the
`completed-q` interior selector), tested in `test_az_gumbel.py`. Choices made where this guide left room:

- The schedule is ported from mctx `seq_halving.py` at commit `428bfb7` and checked against a Python copy of
  it for many (candidates, budget) pairs. Root scoring, final action, mixed value and Q transform follow
  mctx `action_selection.py`, `policies.py` and `qtransforms.py` at the same commit.
- `--gumbel-rescale-values` defaults to on (the mctx default). The min/max is taken over legal moves only.
- Inference returns FP32 log-softmax over legal moves. Edges keep both the log prior and the probability;
  the evaluation cache stores log priors. PUCT uses `exp(log prior)`, equal to the old softmax up to rounding.
- Noisy (training) searches draw Gumbels from the per-move seed that drives Dirichlet noise under PUCT.
  Searches without a seed (evaluation) use zero Gumbels. Temperature sampling is off in Gumbel mode.
- `az_take` returns the target row, the root value and the chosen move (-1 under PUCT, where Python still
  samples or takes the argmax). `Engine.search(..., moves=True)` exposes the move.
- Tree snapshots are format 2 (raw values, log priors). Format 1 still loads: raw value 0 and
  log prior = log(prior), which PUCT never reads and Gumbel never reuses (it starts a fresh tree each move).
- Runs saved without these settings resume as PUCT. `rl --eval-search` (default `puct`) picks the search
  for in-run evaluations.
- Diagnostics are limited to the mean target entropy (`search_stats.target_entropy / targets`) plus the
  existing cache and evaluation counters.
- `rl --budget-minutes` stops a run once self-play and training (not evaluation) reach a wall-clock budget,
  for budget-matched comparisons (`colab_gumbel.sh`).

Budget-matched L4 comparison (2026-10-30): no advantage over PUCT; the pipeline default stays PUCT.
See `measurements/gumbel-vs-puct-20261030/README.md`.

## Intended result

Add an optional Gumbel search mode to `az.py` and `native/az.cpp`, preserving the current PUCT mode. The goal is useful policy improvement with fewer simulations per training move. Equal simulation counts alone do not establish a performance gain.

Implement root Gumbel search and improved policy targets first, with PUCT below the root. Add the full algorithm's interior selector as a separate second step. Keep the existing network architecture, terminal value targets, optimizer, native batching, inference cache, and two search groups.

The paper distinguishes root-only and full variants. Its improvement guarantees assume appropriate action-value estimates; they do not guarantee improvement for this trained Quoridor agent. [Paper, Algorithms 1–2 and Sections 4–5](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

## 1. Configuration and compatibility

Proposed options, not currently implemented:

```text
--search puct|gumbel                   default puct
--gumbel-interior puct|completed-q     default puct
--gumbel-candidates 16
--gumbel-scale 1.0                    training exploration
--gumbel-value-scale 0.1
--gumbel-maxvisit-init 50
--gumbel-rescale-values               explicit saved boolean
--eval-search puct|gumbel
```

Use 16 candidates as an initial engineering choice, not a Quoridor-tuned optimum. Require at least one simulation; cap candidate count by legal actions and simulation budget. Define all rounding behavior through the reference schedule below.

For deterministic Gumbel evaluation set Gumbel scale to zero. For training use scale one at every searched move initially. Disable Dirichlet root noise and the existing visit-temperature move sampler in Gumbel mode. Do not silently reuse `temperature_plies` as a Gumbel schedule.

Save all choices in settings/checkpoints and include them in evaluation output. Old checkpoints missing these fields must resolve to PUCT. Resume currently restores saved learning settings: document an explicit fork/migration path rather than suggesting CLI flags override an existing run.

## 2. Separate the selected action from the training target

Current flow:

```text
az_take -> visit distribution -> searched(g, pi)
                              -> sample/argmax move
                              -> record same pi
```

Proposed flow:

```text
az_take_v2 -> selected real move ID
          -> improved policy, 209 entries in real move coordinates
          -> root value and optional diagnostics
searched(g, result) -> play selected move
                    -> canonicalize and record improved policy
```

Add a versioned native entry point or extend the signature and every ctypes caller together. Preserve PUCT behavior through the same result abstraction. Update both continuous single-network play and the multi-network evaluation branch, plus `Engine.search`, `Engine.take`, and direct test callers.

The move is the root competition's winner. The target is a distribution over **all legal actions**, including unvisited actions. Never sample the played move from that target in the baseline Gumbel implementation.

## 3. Store immutable network values

Add `raw_value` to `Node`. Set it on neural expansion and cache-hit expansion; do not overwrite it during backup. `Node.total / Node.visits` is a changing search average and cannot substitute for the original network estimate.

The existing cache `Entry.value` already supplies the required raw value. `Cache::find` must install it on the node. Include the new node field in native snapshot serialization and bump the format version. Reject or explicitly migrate incompatible snapshots; old accumulated totals cannot recover the raw estimate.

Use these local conventions:

```text
N(a) = child.visits, or zero if absent
Q(a) = -child.total / child.visits, for visited children
v    = node.raw_value
```

The minus sign is essential: child totals use the child's side-to-move perspective. Existing backup already alternates signs correctly. Exact terminal/proof results remain on the same value scale.

For root selection, preserve policy log probabilities. Prefer transmitting FP32 legal log-softmax values from inference and caching them, with probability priors derived as needed for PUCT. Taking logs of existing positive priors is mathematically sufficient up to a constant, but zeros caused by underflow lose information. A probability floor is a documented approximation, not exact recovery of the logits. Update graph output buffers and cache formats together if using log probabilities.

## 4. Root scheduling

At search initialization, draw one Gumbel value per legal action from the existing per-search RNG seed:

```text
g[a] = -log(-log(u[a])),  0 < u[a] < 1
```

Use an open-interval uniform generator; the existing uniform generator can return zero. Reuse this same vector throughout the search. Candidate ranking begins with `log_prior[a] + g[a]`; later comparisons add the transformed action value. Do not redraw noise at phase boundaries. [Paper, Algorithm 2](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

Use DeepMind's considered-visit schedule as the precise reference rather than inventing phase rounding. Precompute the length-B schedule for each effective candidate count and budget. At simulation t, only actions whose root visit count equals schedule[t] are eligible; select the largest Gumbel-plus-log-prior-plus-transformed-Q score among eligible legal actions. The schedule implicitly eliminates candidates. Its remainder handling matters for budgets such as 200. [Reference schedule](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/seq_halving.py)

Port the reference behavior, including the one-action case and final-selection eligibility. Pin a specific upstream commit when implementing; preserve its Apache license attribution if copying code. Do not combine this implicit schedule with a separately implemented explicit elimination scheme.

At completion, restrict eligibility to legal actions with the maximum root visit count, then maximize `g + log_prior + sigma` within that set. This is the reference's final action rule, not simply visit argmax with action-ID tie-breaking. Use stable action-ID tie-breaking only when the full scores tie. [Reference policy output](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)

Keep job-local Gumbels, schedule/cursor, and search mode in `Job`. In `Job::select`, replace selection only when `current == root` for the first milestone. Root expansion is outside the B simulation budget. A cache hit or a proved/terminal backup still completes one simulation and advances the schedule. A pending neural request does not advance it until backup.

The scheduler must remain entirely native. Each job progresses independently through the existing `az_wave`/`az_evaluate` machinery; do not create a Python or global GPU barrier per halving phase.

## 5. Completed action values and policy target

For visited actions V, let S be the sum of their visits and p their network priors:

```text
q_bar = sum(p[a] * Q[a] for a in V) / sum(p[a] for a in V)
v_mix = (v + S * q_bar) / (1 + S)
q_completed[a] = Q[a] if N[a] > 0 else v_mix
```

When V is empty, use v directly. Handle tiny prior mass safely. Optionally rescale completed legal values to [0,1] with an epsilon-protected range. Then:

```text
sigma[a] = (maxvisit_init + max(N)) * value_scale * q_completed[a]
```

Use the same transformation configuration for root competition, target generation and, later, interior selection. The suggested constants and mixed-value calculation follow [DeepMind's Q-transform implementation](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py). Rescaling is an algorithmic choice; save it explicitly.

Construct `target = softmax(log_prior + sigma)` over legal moves, without Gumbels, and scatter it into a zero-initialized 209-entry row. The existing cross-entropy loss can consume it. Keep final game outcome as the value label. [Paper, Section 4](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

Use stable softmax. Never compute a dense loss containing `0 * -infinity`. Existing training logits can remain finite across all 209 outputs, with zero target mass on illegal moves. Preserve canonicalization and horizontal reflection exactly once each.

## 6. Subtree reuse and recovery

`az_begin` currently initializes completed simulations from retained root visits. That is incompatible with a fresh sequential-halving schedule.

For the first correct implementation, start a fresh tree on every Gumbel move, while retaining the shared neural evaluation cache. Set completed to zero. Leave PUCT reuse unchanged. Record this difference in comparisons: disabling reuse can offset savings from a smaller search budget.

Do not merely zero completed while leaving old child visits intact. They would still alter eligibility, Q scaling and budget accounting. A later reuse implementation needs separate fresh-search edge counts and an explicit definition of whether Q uses old or new evidence; treat that as a new algorithm variant.

Continue checkpointing only after active searches drain to move boundaries. There is then no need to serialize an in-flight halving schedule. Preserve game RNG state and seed consumption so resumption reproduces the next move's Gumbels. If checkpoint boundaries ever move inside searches, serialize the entire job, including its fixed noise vector and pending state.

Do not cache Gumbels, improved targets or history-dependent search Q values in the neural evaluation cache. Model changes must continue to invalidate cached neural outputs.

## 7. Full interior selector, second milestone

At non-root nodes, compute the same improved distribution and select:

```text
argmax_a [improved_policy[a] - N[a] / (1 + sum_b N[b])]
```

No interior Gumbel noise is needed. This deterministically tracks the improved policy. Use actual outgoing child counts, not `node.visits`, whose count includes the node's own expansion. [DeepMind action selection](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py)

Keep this selectable separately from root search. It adds per-node arithmetic, so avoid allocations in the hot loop: reuse scratch buffers and use fixed-size storage where appropriate for the engine's legal-action bound.

## 8. Implementation order and review gates

1. Add settings and the separate action/target result type. PUCT must retain existing behavior.
2. Add raw network values, log-prior handling, and versioned snapshots.
3. Implement pure native helpers for mixed values, transformed Q, schedules and selection. Build a small independent reference fixture set.
4. Integrate root-only Gumbel into jobs, including fresh trees and native result extraction.
5. Wire training, evaluation, canonicalization, resume settings and diagnostics.
6. Add the optional interior selector.

Targeted correctness tests to implement later:

| Case | Required result |
|---|---|
| Constant completed Q | Target equals legal network prior |
| Visited child losing for child | Positive Q for parent |
| Zero visited actions | Mixed value equals raw network value |
| One legal action | Legal action selected, target mass one, exact budget |
| Budgets 1, 2, 7, 16, 63, 200 | Exact backup count; no empty eligible set |
| Non-power-of-two legal counts | Schedule matches reference fixtures |
| Fixed logits, values, Gumbels | Candidate visits and final action match reference |
| Gumbel versus target | Noise changes search choices, never directly enters target logits |
| Cache on/off | Same numerical inputs produce equivalent decisions and targets |
| Reordered groups/worker counts | Per-game seeded decisions remain reproducible |
| Mirror and side-to-move transforms | Legal normalized targets survive coordinate mapping |
| Goal, repetition, ply limit, proof | Existing termination rules and signs preserved |
| Save/resume at move boundary | Next seeded searches match uninterrupted execution |
| Old checkpoint | Explicit PUCT default or clear format-migration error |

Also exercise continuous games, two-network matches, negamax replies and overlapping actor/learner mode. Test numeric tolerance separately from bitwise equality where GPU batch shapes differ.

## 9. Observability and later evaluation

Log search mode, effective candidates, actual backups, unique neural evaluations, cache hits, root target entropy, target/prior divergence, chosen-action prior rank, and root raw/mixed/search values. Count diagnostics per iteration rather than copying every node to Python.

The future comparison should include existing PUCT, root-only Gumbel and full Gumbel. Separate equal-search-budget comparisons from equal-wall-time learning curves. Use the same initialization, opponent mixture, openings and optimizer settings initially; report the loss of subtree reuse explicitly. Report goal wins and cutoffs separately.

Acceptance means correct scheduling and target construction, preserved batching/recovery, and eventually better held-out strength per GPU-hour. A lower simulation count or faster rollout alone is insufficient. No training runs, benchmarks or new experiments are authorized by this guide itself.
