# Gumbel AlphaZero vs PUCT, budget-matched on L4 (2026-10-30)

Question: does training with Gumbel search (`GUMBEL_IMPLEMENTATION_GUIDE.md`) produce a stronger model than
the current PUCT pipeline in the same L4 time? If so, it would be used for the final run.

**Answer: no.** All three arms end level head-to-head. Against negamax, the PUCT-trained model played with
PUCT search is the strongest combination. The final run keeps PUCT (`colab_az.sh` unchanged).

## Setup

Three arms, one Colab L4 each (`training/colab_gumbel.sh <arm> 75 ...`), same starting point and settings:
A100 pretrained 128×10 model (`experiments/training/az/colab/pre-model.pt`), the `colab_az.sh` RL settings (2,048 games
per iteration, half against negamax-2000, batch 1,024, lr 3e-4, replay 600k with a quarter seeded from 30k
regenerated negamax games, reuse 4). Each arm stopped starting iterations after **75 minutes of self-play
and training** (`--budget-minutes 75`, in-run evaluations excluded).

| Arm | Search during training | Iterations | Games | Positions | Optimizer steps |
|---|---|---:|---:|---:|---:|
| `puct200` | PUCT, 200 simulations, tree reuse, Dirichlet noise, sampled moves for 8 plies | 52 | 106,865 | 6.42M | 25,053 |
| `gumbel64` | Gumbel root (16 candidates), PUCT below, 64 simulations, fresh tree per move | 52 | 107,119 | 6.40M | 24,983 |
| `gumbel64full` | Gumbel root and completed-Q interior selector, 64 simulations, fresh tree | 50 | 103,334 | 6.17M | 24,061 |

The budget match came out nearly equal in games and training: PUCT's 200-simulation budget counts the
visits already in the retained subtree, so each move costs about as many new network evaluations as a
fresh 64-simulation Gumbel search.

## Results

Held-out openings (`--suites heldout --seed 2027`), 200 games per row, paired colors. Score =
(wins + ½ cutoffs) / games; one standard error is about 2–2.5 points here, and rows share openings.

Against fixed negamax:

| Model | Network only vs 2000 | PUCT 128 vs 2000 | PUCT 128 vs 32000 | Gumbel 128 vs 2000 | Gumbel 128 vs 32000 |
|---|---:|---:|---:|---:|---:|
| `puct200` | 0.860 | **0.928** | **0.893** | 0.908 | 0.878 |
| `gumbel64` | 0.875 | 0.883 | 0.840 | 0.900 | 0.873 |
| `gumbel64full` | **0.895** | 0.918 | 0.825 | 0.900 | 0.830 |

"Gumbel 128" is deterministic Gumbel search (no noise) with the arm's own interior selector; for
`puct200` it used the root-only variant.

Head-to-head, 128 simulations, 200 games, score from the first model's side:

| Pairing | PUCT search | Gumbel search |
|---|---:|---:|
| `puct200` vs `gumbel64` | 0.485 | 0.493 |
| `puct200` vs `gumbel64full` | 0.495 | 0.490 |
| `gumbel64` vs `gumbel64full` | 0.500 | 0.490 |

Gumbel-search head-to-heads use the interior selector of the Gumbel arm involved (root-only for the
`gumbel64` pairing, completed-Q for the other two).

## Observations

- No arm is stronger head-to-head: all six results lie within 1.5 points of 0.5.
- The Gumbel-trained raw networks score 1.5–3.5 points higher against negamax-2000, within about one
  standard error. With search, the PUCT-trained model is ahead of both Gumbel models in every column, by
  0.5–7 points, most clearly with PUCT search against negamax-32000 (5–7 points).
- Gumbel arms finished more games against negamax (63% of finished games vs 51%). With a fresh tree per
  move, a self-play game needs a full search on every ply and takes longer than a game against negamax,
  so negamax games finish more often within an iteration. The opponent ablation (`AZ_REPORT.md`) found
  self-play the better teacher, so this shift works against Gumbel here.
- Gumbel's improved policy targets start almost one-hot (mean entropy 0.11 nats) and widen as training goes
  on (about 0.72 per iteration at the end). The value loss stayed higher than PUCT's (0.31 vs 0.23 at the end), partly because
  the game mix differs.

## Caveats

- One seed per arm; three different VMs (the PUCT and root-only arms ran simultaneously, the full arm
  afterwards on the PUCT arm's VM).
- One Gumbel configuration: 64 simulations, 16 candidates, mctx Q-transform defaults with value rescaling.
  Lower simulation counts, subtree reuse, or a negamax share corrected for the slower self-play games were
  not tried.
- 75 minutes is shorter than a final run; differences that appear only later in training would not show.

`results.json` holds every evaluation file and each arm's config and per-iteration metrics. Models and
logs: `experiments/training/az/gumbel/` (not tracked).
