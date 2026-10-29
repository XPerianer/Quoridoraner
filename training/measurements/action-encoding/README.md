# Pawn action encoding: small goal-tactic experiment

Run: `.venv/bin/python training/experiment_actions.py`

This experiment does not support switching the production policy to relative
pawn actions. On this particular task, destination actions generalize better
when diagonal examples are absent. Both representations succeed once a small
number of diagonal examples is included.

## Setup

- Existing `small` convolutional trunk, FP32 CPU, three paired seeds (1/2/3).
- Destination head: 209 outputs. Relative head: 12 pawn displacements plus
  128 wall placements, 140 outputs. Inputs remain absolute in both arms.
- 1,200 constructed training positions; 150 AdamW updates of batch size 64,
  learning rate .001, weight decay .0001. Paired arms use the same positions,
  initial trunk weights, and sampled minibatches.
- Training diagonal fractions: 0%, 5% (60 positions), and 33.3% (400).
  The remainder is split evenly between ordinary and straight-jump finishes.
- Every example has an immediate win. Policy targets are uniform over winning
  moves, with the existing unmasked policy loss and value target +1. These are
  supervised tactical examples, not self-play or existing checkpoint replay.
- Both players, different columns, and 1–8 randomly selected legal walls.
  All wall placements pass the native legality checks. Positions are constructed
  rather than collected from complete games; full historical reachability is
  not asserted.
- Held-out set: 100 positions per finish type, with exact position duplicates
  excluded from all training pools. The same test set is reused across seeds.
- Measure legal-masked policy argmax and fresh native MCTS at 8/32 simulations
  on 30 of the held-out diagonal cases. No root noise or tree reuse.
- A separate experimental immediate-win guard chooses a legal goal-row move
  before consulting the policy. Every held-out guard move is also applied
  through the native engine to verify its terminal winner. An initial-board
  check verifies that the guard returns no move when no immediate win exists.

## Results

Every seed gave the same accuracies:

| Diagonal fraction in training | Destination: diagonal win accuracy | Relative: diagonal win accuracy |
| --- | --- | --- |
| 0% | 100% | 0% |
| 5% | 100% | 100% |
| 33.3% | 100% | 100% |

Both encodings scored 100% on ordinary and straight-jump finishes in every
condition. The 8- and 32-simulation searches reproduced the table exactly on
their smaller diagonal subset. The explicit immediate-win guard succeeded on
all 300 test positions regardless of representation.

The destination head can reuse goal-square outputs already trained by normal
steps and straight jumps. In the relative head, the four diagonal outputs
receive no positive supervision in the 0% condition. The results are consistent
with this explanation; they do not isolate every internal learning mechanism.
They also demonstrate that limited PUCT search need not rescue an immediate
win whose prior is too small.

## Limits and recommendation

This is an easy, deliberately narrow tactical task: every position is one move
from winning, and the performance saturates. It does not measure overall game
strength, interior diagonal-jump planning, self-play sample efficiency, or the
actual frequency of missed diagonals in existing checkpoints. It does not
compare canonical player-oriented inputs or spatial policy heads. Three seeds
on one fixed test set are not three independent datasets.

Keep destination encoding for now. An exact immediate-win check is a more
direct safeguard against this specific failure than changing encoding. The
check and alternate head here are experiment-only; production search, training,
and checkpoint formats remain unchanged.

`results.json` contains per-run measurements; `positions.json` saves the full
training pools and test cases for inspection. Rerunning deterministically
regenerates the data and replaces the experiment's result files.
