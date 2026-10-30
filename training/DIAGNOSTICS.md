# Learning investigation — 2026-10-26

The evidence points to a learning-signal/exploration bottleneck, rather than a
completely broken optimizer or an obvious search value-sign error. This does
not rule out every implementation bug.

## Controlled experiments

Fresh CPU runs use the same seed (1), network, 9×9 board, standard starts,
64 simulations, 32 games per iteration per arm, 32 updates, batch size 128,
20 iterations, 20 monitoring games and 20 final held-out games. Only wall
inventory differs. Neither experiment changes existing runs.

| Walls per player | Arm | Initial monitoring wins | Final monitoring wins | Held-out wins |
| --- | --- | --- | --- | --- |
| 0 | selfplay | 0/20 | 6/20 | 5/20 |
| 0 | negamax | 0/20 | 0/20 | 0/20 |
| 2 | selfplay | 0/20 | 0/20 | 0/20 |
| 2 | negamax | 0/20 | 0/20 | 0/20 |

The zero-wall selfplay result demonstrates improvement on this seed; it is
not a multi-seed estimate or evidence that the full game is solved. The
held-out openings are separate from monitoring, but share training's seed
configuration and rules. All reported evaluation non-wins were goal losses.

With two walls, all 320 selfplay training games in iterations 11–20 ended
in artificial cutoffs. Simply reducing to two walls does not escape collapse
with these settings.

Reproduce either run in a **new** output directory:

```sh
.venv/bin/python training/train.py --out experiments/training/zero-walls-repeat \
  --iterations 20 --games 32 --parallel-games 32 --batch-size 128 \
  --updates 32 --simulations 64 --walls 0 --curriculum none \
  --eval-games 20 --eval-every 5 --heldout-games 20 \
  --engine direct --workers 4 --device cpu --plain
# Repeat with --walls 2 and a different output directory.

.venv/bin/python training/diagnose.py \
  experiments/training/diagnostic-zero-walls experiments/training/diagnostic-two-walls
```

## Findings in the existing runs

- `adaptive-64-u32` had zero promotion wins in every evaluation through
  iteration 200 and remained at 32 requested nodes. It also had zero wins
  in both arms' 40-game held-out evaluations at iteration 200.
- The low opponent budget is not a random or novice opponent. From the
  standard initial position, requested budgets 1, 32, 64 and 128 all expand
  131 nodes and choose move 67. Depth one always completes and uses the
  handcrafted path evaluator. Lowering this budget further cannot weaken
  that first move.
- Curriculum proportions count games, but replay samples count turns. In
  iterations 192–201 of the adaptive run, standard starts were 320/1280
  selfplay games (25%) but supplied 11,476/15,873 samples (72.3%). The shortest
  race task supplied just 716 samples (4.5%). These are generated samples,
  not measured optimizer sampling frequencies.
- The adaptive negamax checkpoint's replay contained 8,170 losing and 1,830
  winning samples. In the zero-wall control, all 640 negamax training games
  were losses. That network predicted exactly -1.0 for both turns at the
  initial zero-wall position. It has collapsed to a losing value prediction.
  Only learner turns receive training targets in this arm, while MCTS also
  evaluates hypothetical opponent turns. This makes all-loss training
  particularly uninformative; it is not evidence of a reversed backup sign.
- The old `20261022-220205` run's latest 128 selfplay games were all repetition
  cutoffs. Curriculum runs largely eliminate that symptom, but finishing
  selfplay games alone does not establish competitive strength.

## Next training changes to test

1. Use zero walls as a learning acceptance check, then introduce walls in
   stages. Keep evaluation at each difficulty so regression is visible.
2. Balance replay by task, rather than relying on game-count proportions;
   the current short races are underrepresented in optimizer input.
3. For the negamax arm, test an actually weak opponent (for example an
   explicit random-move mixture), and collect value supervision for both
   players. Keep opponent policy moves out of MCTS policy targets unless
   deliberately running an imitation experiment.
4. Repeat with independent seeds before committing to a long training run.

These are proposed controlled changes, not fixes already applied to the
trainer. A physically smaller board would require coordinated rule,
action-space, encoding, network and checkpoint changes: 9×9 and 209 actions
are currently fixed. Zero/two-wall controls isolate difficulty without that
additional implementation risk.

The direct-engine test suite passed (13 tests, one hardware-gated skip),
including immediate wins for either player, terminal/repetition handling,
feature equality, and agreement with the TypeScript implementation.
The full Python suite also passed: 39 tests with two skips, using
`.venv/bin/python -m unittest discover -s training`.

## First intervention: value supervision on both players' turns

New runs now default to `--negamax-value-turns both`. Opponent pre-move
positions are saved with a `None` policy target and the final outcome from
their side-to-move perspective. Policy loss averages only learner rows;
value loss averages all rows. Existing checkpoints retain the old objective
on resume. `--negamax-value-turns learner` allows fresh control experiments.

The matching 20-iteration zero-wall experiment is saved in
`experiments/training/diagnostic-zero-walls-both-values`. Negamax-trained results:

| Objective | Final monitoring wins | Held-out wins |
| --- | --- | --- |
| Learner-only value | 0/20 | 0/20 |
| Both players' value | 4/20 | 2/20 |

The new replay has 4,879 positive and 4,649 negative targets, including
4,879 opponent value-only rows. Initial zero-wall position predictions for
turns 0 and 1 changed from -1/-1 to approximately -0.408/+0.481. These are
network predictions, not assertions about the true game-theoretic values.
The latest 320 training games include 79 learner wins rather than zero.

This is a promising single-seed result, not proof of a general improvement.
Adding rows changes replay composition and random sampling; with the same
batch size, there are fewer policy-supervised rows per update. Both arms
also share a Python random stream, so changing negamax training can change
later selfplay trajectories. Its different score is not an independent
effect of this intervention on selfplay.

The matching two-wall run, `diagnostic-two-walls-both-values`, still has
zero negamax learner wins: 0 wins / 0 cutoffs / 20 losses in final monitoring
and 0 / 1 / 19 held-out. Its latest 320 training games were all losses.
Both-side value supervision helps the zero-wall control but does not by
itself solve learning with walls at this training budget.

Regression tests cover missing policy targets, policy-loss normalization,
value target signs, checkpoint serialization, preservation of the old
objective, and exact interrupted/resumed replay and model weights.
All 45 Python tests passed (two skips) after this change.

## Performance gates and task-balanced replay

Implemented as `--curriculum gated --replay-sampling task`; see README for
the full command and thresholds. This is opt-in and does not alter the
existing `both-values-full-curriculum` run.

The 20-iteration CPU diagnostic `gated-balanced-diagnostic` used 32 games per
arm, 64 simulations, 32 updates, batch size 128, 40 curriculum evaluation
games, and evaluation every five iterations. Both arms scored 20/40 at
race-0 in iterations 5 and 10 and advanced to race-1. They then scored 6/40
and 8/40 at iterations 15 and 20, so neither advanced further. Baseline
performance did not trigger promotion. No harder tasks entered training.

At completion, the selfplay replay held 1,012 race-0 / 775 race-1 samples;
negamax held 862 / 1,028. A 128-row minibatch contained 64 from each task in
both arms. Full-game monitoring and held-out results were still 0/20 wins
each for both arms. This verifies the gating/sampling behavior, not a claim
of stronger full-game play after this small experiment.

The Python suite passed 53 tests with two skips, including gate failures,
consecutive successes, task retention, balanced sampling, evaluation-suite
separation, and deterministic interrupted/resumed training.
