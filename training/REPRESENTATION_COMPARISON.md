# Board representation pilot

Two seeds (1 and 2), 50 AdamW updates per model, identical replay minibatches,
game-pair validation split with exact validation states removed from training.
Source: both-values-full-curriculum/latest.pt. CPU; another trainer was active,
so timing results are not a controlled comparison. Lower errors are better.

| Targets | Architecture | Parameters | Validation policy KL | Validation value MSE |
| --- | --- | ---: | ---: | ---: |
| selfplay | small | 370,866 | 2.7038 | 0.3820 |
| selfplay | edges | 372,018 | 2.7388 | 0.4794 |
| selfplay | lattice | 371,154 | 2.7037 | 0.3835 |
| selfplay | mlp | 356,690 | 2.7328 | 0.3818 |
| negamax | small | 370,866 | 3.8237 | 0.8738 |
| negamax | edges | 372,018 | 3.8010 | 0.8307 |
| negamax | lattice | 371,154 | 3.7891 | 0.8883 |
| negamax | mlp | 356,690 | 3.8045 | 0.9139 |

No consistent winner in this short pilot. Keep small as the default.
Lattice approximately matches the self-play baseline and has lower mean negamax
policy error; edges improve mean negamax value error but worsen self-play errors.
These small-sample averages do not establish statistical significance or strength.

Next: 500-update frozen fits, then equal-time online training and paired matches
with the same architecture list and seeds, without a concurrent trainer.
See [commands and representations](README.md#board-representation-experiments).
Raw per-seed results and source fingerprint: [measurements](measurements/representation-pilot.json).
