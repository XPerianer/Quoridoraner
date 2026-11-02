# One-hour transformer versus ResNet

**Recommendation: keep the 128×10 ResNet for the current L4 training pipeline.** The tested transformer scored 28.6% against the one-hour ResNet (400 games; paired 95% interval 24.8–32.5%), about −159 descriptive Elo. It also used 5.0% more RL time and has slower inference. This is evidence against this particular 6×192 transformer and training recipe, not every transformer design.

Separate NVIDIA L4 experiment; the existing `qfinal` training process was not modified.

## Training budgets

| Model | RL minutes | Games | Optimizer steps |
|---|---:|---:|---:|
| ResNet 128×10 | 61.22 | 69,750 | 17,528 |
| Transformer 6×192 | 64.30 | 43,085 | 10,942 |

RL budget includes self-play and optimizer updates; evaluation and compilation outside that accounting are separate. Both use pure self-play, randomized 40/400 playouts, AdamW 3e-4, batch 1,024, replay 600,000 and reuse 4.

The requested duration was matched to the recorded ResNet checkpoint, but the transformer finished its final iteration at 64.30 minutes versus 61.22 minutes for ResNet (+5.0%). This small mismatch favors the transformer. ResNet generated 1.62× as many games and performed 1.60× as many optimizer steps. An exact-duration rerun is unlikely to reverse this result; the measured score interval is well below 50%.

Transformer supervised training: 60.22 minutes, four epochs. The historical ResNet log records 15.84 minutes on different hardware. The original supervised dataset was unavailable; its 100,000-game generation recipe was recreated. Replay seed data was copied identically. **This comparison matches RL time, not total lifetime compute.**

## Playing strength

Scores are from the transformer’s perspective; cutoffs receive half a point. Intervals bootstrap paired openings, not individual games; constant observed pair scores use conservative bounds. Intervals do not include training-seed uncertainty.

| Opponent / search | Games | W / L / cutoff | Score | Paired 95% CI | Descriptive Elo |
|---|---:|---|---:|---|---:|
| hour000-s200 | 200 | 177 / 22 / 1 | 88.8% | 84.2%–92.8% | +359 |
| hour002-s200 | 200 | 50 / 150 / 0 | 25.0% | 20.0%–30.0% | -191 |
| hour003-s200 | 200 | 53 / 146 / 1 | 26.8% | 21.5%–31.8% | -175 |
| hour004-s200 | 200 | 46 / 154 / 0 | 23.0% | 18.0%–28.0% | -210 |
| hour1-s1 | 400 | 119 / 277 / 4 | 30.2% | 26.1%–34.2% | -145 |
| hour1-s200 | 400 | 114 / 285 / 1 | 28.6% | 24.8%–32.5% | -159 |

Only the h001 match is the primary matched-RL-budget comparison. Later ResNet checkpoints are context. One-simulation matches are a policy-dominated diagnostic. Equal search simulations do not mean equal thinking time.

## Inference cost

Synthetic steady-state production CUDA-graph timings; compilation excluded. These are network-only timings, not full search latency.

| Architecture | Parameters | Batch | Milliseconds | Positions/s |
|---|---:|---:|---:|---:|
| transformer | 1,850,696 | 64 | 0.709 | 90,279 |
| transformer | 1,850,696 | 256 | 3.888 | 65,842 |
| transformer | 1,850,696 | 1024 | 25.372 | 40,359 |
| transformer | 1,850,696 | 2048 | 54.440 | 37,619 |
| resnet | 3,188,012 | 64 | 0.617 | 103,699 |
| resnet | 3,188,012 | 256 | 1.898 | 134,908 |
| resnet | 3,188,012 | 1024 | 10.405 | 98,415 |
| resnet | 3,188,012 | 2048 | 25.527 | 80,230 |

## Limits

One seed and one compact transformer configuration cannot establish that an architecture family is universally better. Pretraining initialization differs, shared hyperparameters may favor one model, and cutoff adjudication can affect strength estimates. A deployment choice should also consider equal-time matches if inference costs differ materially.

Raw match JSON includes every game, paired uncertainty, checkpoint hashes, and settings. See PLAN.md, run.sh, source-sha256.txt, resnet-checkpoints.json and benchmark.json for provenance.
