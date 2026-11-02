# Transformer versus ResNet, 2026-10-31

Protected existing session: `qfinal`. Never upload code, modify files, interrupt processes, or run evaluation there. Read-only copies of completed checkpoints and immutable data are permitted. Experiment session: `qtransformer`, separate NVIDIA L4.

Transformer: 81 tokens, learned absolute square embeddings, 6 pre-LN blocks, width 192, 6 heads, 2x GELU MLP, final LN. 1,850,696 parameters. ResNet: 128 channels, 10 blocks, 3,188,012 parameters.

Primary comparison: one hour of self-play plus optimizer time, excluding evaluation, matching the existing run's budget definition. Report actual consumed seconds because iteration boundaries can overshoot. Both use 2,048 games/iteration, pure self-play, randomized 40/400 playouts, AdamW constant 3e-4, batch 1,024, replay 600,000, reuse 4, seed 1. Copy the existing 30k-game replay seed.

The ResNet h001 checkpoint records 3,673.474 seconds (61.225 minutes), so the transformer budget is set to that measured duration rather than nominal 60 minutes. Final actual duration must still be checked.

Initialization limitation: original ResNet supervised dataset is unavailable at checked locations. Recreate its documented 100k-game seed-1 generation recipe and train transformer for four epochs, batch 1,024, OneCycle max LR .002. This matches data recipe and epochs, not pretraining GPU time; original pretraining log reports 950.7 seconds (historical hardware differs). Report pretraining cost separately. Do not describe this as an equal total lifetime compute experiment or definitive architecture ranking.

Evaluation: primary 200 paired openings (400 games), heldout seed 731, 200 simulations per move, max 100 plies, same PUCT settings. Secondary policy-only 1-simulation match and longer-trained ResNet checkpoints are contextual. Score cutoffs as 0.5 and report their count. Compare inference throughput separately; equal simulations do not imply equal thinking time. Save raw results and confidence intervals, preferably bootstrap over paired openings. Single training seed limits generalization.

Local tests: 3 architecture and 7 existing tests passed. Remote CUDA benchmark must pass before training. Source and artifacts are under this directory; large artifacts should remain untracked.
