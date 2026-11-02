# Optimized transformer: overnight learning curve

**Matched-budget result: the ResNet remains stronger.** The 12-hour transformer scored 34.5% against the 12-hour ResNet (95% interval 31.0%–38.0%), approximately 111 Elo weaker under equal 200-simulation search. This establishes lower strength at this budget, not a lower asymptotic ceiling. The early gains flattened between eight and twelve hours. This run provides no evidence of a higher ceiling, and does not justify extending this exact configuration unchanged.

Architecture: pre-LN transformer width 256, four blocks, four heads, 2× MLP expansion, 2,185,864 parameters. Inference microbatch 64; training microbatch 128 with effective batch 1,024. Fresh four-epoch supervised pretraining, then up to 12 hours of self-play and optimization. See PLAN.md for the controlled settings and limits.

All scores below are from the transformer’s perspective. Matches use 400 games, 200 paired heldout openings, 200 simulations per move, seed 731. Cutoffs receive half a point. Confidence intervals resample opening pairs. Equal simulations is not equal thinking time.

| Transformer target hours | ResNet checkpoint | W–L–cutoffs | Score | 95% CI | Descriptive Elo |
|---:|---|---:|---:|---:|---:|
| 1 | model-h001-it00034.pt | 100–299–1 | 25.1% | 21.4%–29.1% | -190 |
| 1 | model-h013-it00451.pt | 78–322–0 | 19.5% | 16.2%–22.8% | -246 |
| 2 | model-h001-it00034.pt | 121–277–2 | 30.5% | 26.9%–34.0% | -143 |
| 2 | model-h002-it00068.pt | 108–292–0 | 27.0% | 23.5%–30.5% | -173 |
| 2 | model-h013-it00451.pt | 91–307–2 | 23.0% | 19.4%–26.6% | -210 |
| 4 | model-h001-it00034.pt | 160–240–0 | 40.0% | 36.2%–43.8% | -70 |
| 4 | model-h004-it00137.pt | 120–279–1 | 30.1% | 26.6%–33.5% | -146 |
| 4 | model-h013-it00451.pt | 108–292–0 | 27.0% | 23.2%–30.8% | -173 |
| 8 | model-h001-it00034.pt | 191–208–1 | 47.9% | 44.4%–51.5% | -15 |
| 8 | model-h008-it00279.pt | 132–267–1 | 33.1% | 29.6%–36.6% | -122 |
| 8 | model-h013-it00451.pt | 123–276–1 | 30.9% | 27.4%–34.2% | -140 |
| 12 | model-h001-it00034.pt | 183–215–2 | 46.0% | 42.5%–49.5% | -28 |
| 12 | model-h012-it00418.pt | 138–262–0 | 34.5% | 31.0%–38.0% | -111 |
| 12 | model-h013-it00451.pt | 126–272–2 | 31.8% | 28.1%–35.2% | -133 |

## Direct progress checks

| Match (score from first model) | W–L–cutoffs | Score | 95% CI |
|---|---:|---:|---:|
| resnet-h013-vs-h008 | 213–183–4 | 53.8% | 50.5%–57.0% |
| transformer-h012-vs-h008 | 194–204–2 | 48.8% | 45.1%–52.4% |

## Actual training budget

| Target hours | Actual RL hours | Iterations | Completed games | Optimizer updates |
|---:|---:|---:|---:|---:|
| 1 | 1.025 | 28 | 57,454 | 14,690 |
| 2 | 2.021 | 56 | 114,905 | 29,466 |
| 4 | 4.017 | 115 | 236,006 | 59,012 |
| 8 | 8.011 | 235 | 482,487 | 118,480 |
| 12 | 12.025 | 357 | 733,187 | 178,379 |

## Did more training still help?

Changes from 8 to 12 RL hours, using the same opening pairs against each frozen opponent. The intervals bootstrap the difference within opening-pair clusters; they describe opening-sample uncertainty, not variation between training seeds.

| Frozen opponent | Change in score | 95% interval for change |
|---|---:|---:|
| 1-hour ResNet | -1.88 percentage points | -6.63 to +3.00 |
| 13-hour ResNet | +0.88 percentage points | -3.38 to +5.12 |

Neither frozen-opponent comparison shows a clear gain over these final four hours. The direct 12h-versus-8h transformer match is an additional check. A plateau here could reflect architecture, optimization, data or search; it does not establish a universal transformer ceiling.

## Interpretation

Compare the same frozen 13-hour ResNet across stages to measure the transformer’s progress. Matching-hour opponents address early compute efficiency. A rising score against a frozen opponent is stronger evidence of continued learning than saturated negamax scores or adjacent-checkpoint matches. Twelve hours and one seed cannot determine the asymptotic ceiling.

The pipeline restarts from full optimizer/replay/RNG/tree state between evaluation stages. It retains a constant learning rate 3e-4 throughout; no result-dependent tuning is applied. Hourly snapshots record actual training seconds; stage budgets can overshoot at iteration boundaries. Pretraining, generation and evaluation consume additional hardware time and are excluded from the 12-hour RL budget.

Raw match records, model fingerprints and confidence-interval metadata are under artifacts/backup/eval. Full model checkpoints and resumable states are kept locally in ignored artifacts; source and launch recipe are tracked separately.
