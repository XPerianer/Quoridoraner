# Beating the 2000-node negamax — investigation and fix (2026-10-28)

**Result:** the new pipeline (`training/az.py`) produces a model that beats the
fixed 2000-node negamax in **94–97% of games** on fresh held-out openings
(and 78% with no search at all). Every earlier model in this repo scored
0 wins in its 40-game held-out evaluation. Final model:
`training/runs/az/colab/rl-model.pt` (128 channels × 10 residual blocks).

All numbers below are paired-color matches against `negamax(nodes=2000)` (the
same C++ searcher and the same 100-ply limit the old trainer used). Openings come
from `evaluation_opening`, bit-identical to `curriculum.evaluation_opening`
(verified by `test_az.py`). "Held-out" = `--suites heldout --seed 2027`, which no
training or tuning step ever used. Score = (wins + ½ cutoffs) / games.

### Headline: final model (A100, bootstrap + 40 RL iterations), 200 held-out games each

| Opponent | MCTS simulations | Wins | Losses | Cutoffs | Score |
|---|---:|---:|---:|---:|---:|
| negamax 2000 | 1 (raw network) | 154 | 43 | 3 | **0.778** |
| negamax 2000 | 128 | 187 | 11 | 2 | **0.940** |
| negamax 2000 | 400 | 192 | 6 | 2 | **0.965** |
| negamax 8000 | 400 | 189 | 9 | 2 | 0.950 |
| negamax 32000 | 400 | 186 | 12 | 2 | 0.935 |
| negamax 2000, `monitor` suite (the old trainer's 40 games) | 200 | 40 | 0 | 0 | **1.000** |

Wins are near-equal by color (e.g. 96/96 at 400 simulations). Re-evaluated
locally on MPS after download: 93/100 at 128 simulations.

## Why the earlier models lost

1. **The negamax opponent's strength is its evaluation, not its depth.** Its
   leaf evaluation is the exact shortest-path difference (BFS around walls) plus
   wall difference. A deeper negamax with the same evaluation does *not*
   reliably beat it (challenger score over 40 paired games: 500 nodes 0.38,
   2000 0.50, 8000 0.60, 32000 0.35, 128000 0.51). A learner has to reach at
   least that level of positional understanding before search helps.
2. **The network had to rediscover BFS from raw planes.** Inputs were raw pawn
   and wall-anchor planes in absolute coordinates (player 1 seen upside down
   relative to player 0), and the default network was two 3×3 convolutions
   (receptive field 5×5 on a 9×9 board) followed by a fully connected layer.
   Shortest-path length around walls is a global, iterative computation that a
   5×5 receptive field cannot express. Residual variants were tried, but each
   only for 20 short iterations.
3. **Far too little, too sparse signal for tabula-rasa AlphaZero.** Colab runs
   used 20 iterations × 128 games per arm (≈2.5k games), 64 simulations over
   ~130 legal moves, and mostly curriculum race positions. Random networks
   produce random play, games end by cutoff or loss, and value targets
   collapse (DIAGNOSTICS.md: the negamax arm predicted −1.0 everywhere).
   Nothing in the loop injects knowledge of what good Quoridor looks like.
4. **Search details that hurt a weak network.** Unvisited children used
   Q = 0 (first-play urgency of 0). When the network thinks it is losing,
   every unexplored move looks better than every explored one, which spreads
   64 simulations thinly over ~130 moves.

## What changed (and why)

| Decision | Why |
|---|---|
| **Perspective-canonical encoding** (side to move always walks to row 0; board flipped for player 1) | Both colors share one pattern set; doubles effective data. Tested: flipped position ⇒ identical features. |
| **Path features** (19 planes): BFS distance-to-goal maps for both players, cells on a shortest path, four blocked-step planes, wall counts, clipped path difference, turn, repetition count, remaining plies | Gives the network what the negamax evaluation knows, so it can learn *corrections* to it (tempo, jumps, wall timing) instead of re-deriving BFS. |
| **Left-right mirror augmentation** at every training batch | Exact symmetry of Quoridor. |
| **ResNet with BatchNorm and global-pooling bias** (64×6 locally, 128×10 on A100) | Global pooling lets every square see far-away walls. |
| **Board-aligned policy head** (1×1 conv → pawn-square, h-wall-anchor and v-wall-anchor planes = 81+64+64 = 209 logits) | Moves are spatial; avoids learning 209 unrelated FC outputs. |
| **Supervised bootstrap from negamax games** (`az.py gen` + `pretrain`) | Cheapest way to a competent prior and a calibrated value net. Diversity: random openings (0–10 plies, 25% wall moves), random node budgets per side (500/1000/2000/4000), 5% random moves. Policy target = negamax's choice at that position; value target = game outcome. 4,000 games take 17 s on the M1. |
| **PUCT with FPU = parent Q − 0.25**, c_puct 1.5, tree reuse across moves | Standard Leela-style fix for item 4. |
| **RL against negamax and self-play** (`az.py rl`): 50% of games vs negamax (learner alternates colors), 50% self-play; Dirichlet noise, sampled moves for the first 8 plies, 200 simulations; a quarter of the replay seeded with supervised data | Self-play generalises; games vs the target opponent concentrate learning on what beats it. Negamax-turn positions get value targets only (no imitation in RL). |

## Experiments

### Local pilot (M1 Pro, MPS) — 4,000 negamax games, 64×6 net, 3 epochs, ~3 minutes training

Held-out, 100 games (50 paired openings), vs negamax-2000:

| Model | Simulations | Wins | Losses | Cutoffs | Score |
|---|---:|---:|---:|---:|---:|
| Pilot (all features) | 1 (raw policy) | 34 | 65 | 1 | 0.345 |
| Pilot (all features) | 128 | 74 | 25 | 1 | 0.745 |
| Pilot (all features) | 400 | 86 | 13 | 1 | 0.865 |
| Ablation: path planes zeroed (4–11, 14) | 128 | 54 | 45 | 1 | 0.545 |

The pilot also scored 30/40 on the 20-pair `monitor` suite at 128 simulations.
It was retrained once with the same seed (the first run's loss log was lost) and
again scored 74/100 at 128 simulations.

Training loss after 1,446 steps (batch 512): with path features policy 0.416 /
value 0.525; without, 0.506 / 0.505. Policy fitting is clearly better with the
features. The lower training value loss without features does not translate
into play, which is consistent with memorising outcomes rather than evaluating
positions.

Conclusions:
- The **bootstrap + new architecture/encoding** already gets past 50% (54%).
- **Path features** add +20 points at the same search budget.
- **Search** matters: the raw policy alone loses (34%), and MCTS turns the
  learned prior and value into a win rate of 75–87%.

### A100 pipeline (`training/colab_az.sh`, Colab A100-40GB, 12 vCPUs)

1. `gen`: 100,000 negamax games → 6.19M positions (16 min).
2. `pretrain`: 128 channels × 10 blocks, 4 epochs, batch 1024, OneCycle
   lr 2e-3, bf16 autocast (16 min). Final training loss policy 0.162 / value 0.512.
   Monitor suite at 200 simulations: 33/40.
3. `rl`: 40 iterations × 512 games (256 vs negamax-2000 with alternating colors,
   256 self-play), 200 simulations, replay 600k with 150k supervised rows
   mixed in initially, ~4 samples per new position, lr 3e-4 (85 min total,
   ~105 s per iteration, 20,480 games).

Monitor suite = 40 games, 200 simulations. Train wins are against negamax-2000
with Dirichlet noise and sampled opening moves:

| RL iteration | Policy loss | Value loss | Train wins vs negamax | Mean plies | Monitor wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.223 | 0.500 | 229/256 | 67 | |
| 5 | 0.313 | 0.442 | 238/256 | 67 | 39/40 |
| 10 | 0.380 | 0.406 | 238/256 | 64 | 38/40 |
| 15 | 0.453 | 0.378 | 245/256 | 64 | 35/40 |
| 20 | 0.557 | 0.342 | 249/256 | 63 | 40/40 |
| 25 | 0.608 | 0.321 | 252/256 | 61 | 40/40 |
| 30 | 0.653 | 0.311 | 247/256 | 61 | 39/40 |
| 35 | 0.703 | 0.295 | 252/256 | 59 | 39/40 |
| 40 | 0.748 | 0.281 | 248/256 | 59 | 40/40 |

Policy loss rises because targets change from one-hot negamax moves to
MCTS visit distributions, which have more entropy; it is not a regression signal.
Value loss falls steadily. Games get shorter, consistent with winning faster.

### Same 200 held-out games across stages (vs negamax 2000)

| Model | Raw network | 128 simulations |
|---|---:|---:|
| Local pilot (4k games, 64×6) | 0.390 | 0.748 |
| A100 pretrain (100k games, 128×10) | 0.450 | 0.868 |
| + 10 RL iterations | 0.535 | 0.903 |
| + 40 RL iterations (final) | **0.778** | **0.940** |

The RL-10 snapshot also generalised to other budgets (100 held-out games,
128 simulations): negamax 500 → 0.94, 8000 → 0.895, 32000 → 0.775.

All three stages contribute:
- **More data and a bigger net** add about 12 points at 128 simulations.
- **RL** adds about 7 more, and lifts the raw network from 0.45 to 0.78.
  RL is what makes the network itself stronger than negamax's evaluation,
  not just its MCTS.

### Opponent ablation: negamax games vs self-play (2026-10-29)

Question: does RL improve more from games against negamax or from self-play? The main run used a 50/50 mix,
so it could not say. Three arms, each starting from the A100 pretrained model with the `colab_az.sh` RL settings
(512 games, 200 simulations, lr 3e-4, a quarter of the replay seeded with supervised rows) for 19 iterations:

| Arm | Training games | `--negamax-fraction` |
|---|---|---:|
| negamax | all 512 vs negamax-2000 (learner alternates colors) | 1.0 |
| selfplay | all 512 self-play | 0.0 |
| mixed | 256 vs negamax + 256 self-play (as in the main run) | 0.5 |

Same number of games per arm, so the negamax arm gets about half as many positions with a policy target
(negamax's own moves only give value targets). All evaluation ran on a Colab L4 (`colab_opponents_eval.sh`):
held-out suite, seed 2027, 200 games per row, score = (wins + ½ cutoffs) / games.

Against fixed negamax:

| Model | Network only vs 2000 | 128 sims vs 2000 | 128 sims vs 32000 |
|---|---:|---:|---:|
| pretrained (start) | 0.450 | 0.868 | 0.760 |
| negamax | 0.570 | 0.927 | **0.882** |
| mixed | 0.625 | 0.938 | 0.868 |
| selfplay | **0.637** | **0.943** | 0.855 |

Round robin, 128 simulations, 200 games per pairing, paired colors (score from the first model's side):

| Pairing | W | L | Cutoffs | Score |
|---|---:|---:|---:|---:|
| negamax vs pretrained | 140 | 56 | 4 | 0.710 |
| selfplay vs pretrained | 157 | 38 | 5 | 0.798 |
| mixed vs pretrained | 160 | 35 | 5 | 0.813 |
| negamax vs selfplay | 78 | 114 | 8 | 0.410 |
| negamax vs mixed | 66 | 129 | 5 | 0.343 |
| mixed vs selfplay | 96 | 97 | 7 | 0.497 |

Findings (one standard error is about 3.5 points at 200 games):

- **Against negamax with search, the arms are tied.** They are within 1.6 points of each other vs negamax-2000
  and within 2.7 points vs negamax-32000.
- **Self-play builds the better network.** The raw network scores 0.63–0.64 with self-play vs 0.57 without it.
  The negamax-only arm loses head-to-head to both arms that include self-play: 0.41 vs selfplay (about 2.5 SE)
  and 0.34 vs mixed (about 4.5 SE). It also gains the least against the pretrained model.
- **Mixed and pure self-play are equal** (0.497 head-to-head).
- **Interpretation:** games against negamax teach how to beat negamax, but little beyond that. Self-play
  produces stronger general play, with no cost against negamax. Keep self-play in the loop; the negamax
  share can go to zero without losing strength against negamax.

Caveats:

- One seed per arm.
- 19 iterations rather than the main run's 40.
- The supervised mix data came from 30k regenerated games rather than 100k (same generator and seed on
  every VM, so identical across arms).
- Colab removed the selfplay VM after iteration 10. That arm continued from its iteration-10 model for
  9 more iterations with a fresh optimizer, a fresh replay buffer and seed 2. The other two arms ran
  uninterrupted.

Infrastructure notes:

- Colab removed four VMs during this experiment, 25–80 minutes after creation, with no visible cause.
- All training ran on L4s. The evaluation ran on one L4. Results were synced every 3–5 minutes.
- An L4 (1.54 units/h) was as fast as the A100 here, because MCTS is partly CPU-bound.
- Total cost: about 14 units, of which about 7 were lost with the first A100 VM.
- Artifacts are in `training/runs/az/opponents/`: `models/` and the per-arm logs/metrics.

## Reproduce

```sh
# Tests (opening equivalence with the old harness, perspective symmetry, canonical moves, search)
.venv/bin/python -m unittest discover -s training -p test_az.py

# Local pilot (~5 minutes on an M1 Pro)
.venv/bin/python training/az.py gen --out training/runs/az/gen-local.npz --games 4000 --parallel 500
.venv/bin/python training/az.py pretrain --data training/runs/az/gen-local.npz --out training/runs/az/pre-local \
    --epochs 3 --eval-pairs 0
.venv/bin/python training/az.py eval training/runs/az/pre-local/model.pt --pairs 50 --simulations 128 \
    --suites heldout --seed 2027

# Ablation without path planes
.venv/bin/python training/az.py pretrain ... --drop-planes 4 5 6 7 8 9 10 11 14
```

The A100 pipeline is `training/colab_az.sh`, driven from the `colab` CLI:

```sh
colab new -s qaz --gpu A100
zip az-src.zip training/az.py training/test_az.py training/native/az.cpp training/native/engine.hpp
colab upload -s qaz az-src.zip /content/az-src.zip
colab upload -s qaz training/colab_az.sh /content/pipeline.sh
# unzip to /content/q, then: nohup bash /content/pipeline.sh &   (logs in /content/runs)
# then training/colab_az_eval.sh for the final evaluation table
colab download -s qaz /content/runs/rl/model.pt training/runs/az/colab/rl-model.pt
colab stop -s qaz
```

The opponent ablation runs one arm per VM: `bash colab_opponents.sh <negamax|selfplay|mixed> <fraction>`.
It needs `/content/pre-model.pt`. Afterwards, `colab_opponents_eval.sh` evaluates
`/content/models/{pretrain,negamax,mixed,selfplay}.pt`. Head-to-head matches use `az.py h2h`.

## Limits and caveats

- Opponent = one fixed deterministic engine. A deterministic opponent can be
  exploited by a learner trained against it. Held-out openings reduce, but do not
  remove, that concern. The self-play half of RL and the supervised bootstrap
  keep the model general, but strength against other engines or humans has not
  been measured.
- Scores are on 100–200-game samples. A rough 95% interval is ±9 points near
  0.75 with 100 games, and ±3 points near 0.95 with 200 games.
- No tabula-rasa control was run with the new features and network (RL from
  random weights). So the bootstrap's share of the gain is shown by the
  pilot, not by a matched RL-only run.
- The MCTS player uses more wall-clock time per move than negamax-2000 (about
  0.3–1 s vs ~1 ms per move on the CPU).
- Old `train.py`, `core.py` and checkpoints are unchanged. `az.py` uses its own
  features and checkpoint format, so old checkpoints cannot be loaded by it.
