# Exploratory forecast, recorded before the 8h evaluation

Fit descriptive Elo linearly against log2(training hours), separately for each frozen opponent, using only1h,2h,4h matches. Assumes the same improvement per doubling of RL time continues. No ceiling parameter is identifiable from these three early checkpoints. Forecasts are descriptive and have no calibrated uncertainty interval; their model uncertainty dominates sampling uncertainty.

| Frozen opponent | 8h forecast score | 12h forecast score |
|---|---:|---:|
| ResNet1h |47.8%|52.8%|
| ResNet13h |31.4%|34.1%|

These are scores from the transformer perspective, not decisive-game win percentages. A saturating curve can fit the early points and plateau below50%; a later improvement could also violate this model. The8h checkpoint is an out-of-sample check, not an architecture-tuning trigger. Preserve the preplanned12h run and evaluate before drawing a recommendation.
