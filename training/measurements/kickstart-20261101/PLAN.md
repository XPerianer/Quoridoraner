# Kickstarted larger students: ResNet 256×20 vs transformer 256×4

Question: can a larger network, started from random weights and taught by the plateaued ResNet h013
early on, get past h013 within 8 L4-hours? And which architecture gets further in the same time?

## Arms (one L4 each, `run.sh <arm>`)

| Arm | Network | Parameters |
|---|---|---:|
| `resnet` | ResNet 256 channels × 20 blocks | about 24M |
| `transformer` | pre-LN transformer, width 512, 6 blocks, 8 heads, MLP ×2 | about 12.6M |

The transformer is the largest benchmarked size that is at least as fast as the ResNet on the L4
(`bench-transformers`): inference 15.9k vs 15.1k positions/s, training step with teacher 340 vs 360 ms. So the
arms match L4 hours and, roughly, games and updates per hour; they do not match parameters.

## Shared settings

- Random initial weights. No negamax pretraining and no supervised replay seed: the teacher is the only
  outside knowledge.
- Simplified 10-channel inputs (`--feature-set simplified`, see `AZ_FEATURES.md`): pawns, walls, both
  distance maps, both shortest-path masks, wall counts. The native encoder still builds 19 channels; the student
  selects its 10 inside the network, and the 19-channel teacher reads the same batch.
- Teacher: frozen ResNet h013 (`model-h013-it00451.pt`, sha256 `5815804d…`). Its raw policy (temperature 1)
  and value are extra targets on every training row: loss = RL loss + λ·(cross-entropy to the teacher
  policy + squared error to the teacher value). λ = 1 at the start and decays linearly to 0 at 4 training hours
  (`--teacher-weight 1 --teacher-hours 4`). The teacher term is computed live per batch, so nothing from the
  teacher is stored in the replay once λ = 0. The teacher puts 0.07% of its probability on illegal moves, so
  its policy is used unmasked.
- RL recipe from the final ResNet run: pure self-play, 2,048 games per iteration, randomized 40/400 playout
  caps, AdamW at constant 3e-4, batch 1,024, replay 600k, reuse 4, seed 1.
- Budget: 8 hours of self-play and training. Evaluations are excluded, as before.

## Measurements

- Hourly snapshots, each playing h013 (100 pairs, seed 731, 200 simulations). This is the live learning curve.
- After training: the 1, 2, 4 and 8 h snapshots against h013 with the transformer-night protocol
  (`evaluate_transformer.py`, 200 pairs, seed 731, 200 simulations). That makes the transformer arm directly
  comparable with the no-teacher transformer run (19.5 / 23.0 / 27.0 / 30.9 / 31.75% against h013 at 1/2/4/8/12 h).
- The final model of each arm also plays h013 at equal time per move (50 ms, `evaluate_timed.py`, 100 pairs).
- Metrics per iteration: `teacher_weight`, `teacher_kl` (KL from the teacher's policy) and `teacher_value`.

## What to read at 8 h

- Score against h013 above 50% (interval excluding 50%) at 8 h, still rising after λ reached 0: the larger student
  has passed its teacher.
- A score flat at about 50% after 4 h: the student is stuck at the teacher's level (see the risks in the
  discussion). Continuing from `state-latest.pt.gz` costs nothing extra to set up.
- Transformer arm against the no-teacher transformer at the same hours: teacher plus simplified inputs
  together, not separable here.
- There is no no-teacher 256×20 control, so a gain of the ResNet arm cannot be split into "capacity" and "teacher".

## Limits

One seed per arm. Two different VMs (cross-VM speed variation). 8 hours with a 4-hour decay leaves only 4 hours of
pure self-play. Equal simulations is not equal thinking time; the timed match covers that once.

## Budget

About 2 compute units per L4-hour. 8 h training plus 1–1.5 h of evaluation per arm, two arms: about 40 units.
