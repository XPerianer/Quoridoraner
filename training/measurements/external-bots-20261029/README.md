# Public-bot matches — 2026-10-29

The final Colab model won both games against Gorisanson Strong and lost both
against SigmaQuoridor. These are four standard-start games with colors swapped,
not a statistical strength estimate.

| Opponent | Our side | Outcome | Plies | Our mean sec/move | Opponent mean sec/move |
|---|---|---|---:|---:|---:|
| Gorisanson Strong | First | Win | 81 | 0.65 | 6.26 |
| Gorisanson Strong | Second | Win | 68 | 0.53 | 3.11 |
| SigmaQuoridor | First | Loss | 38 | 0.82 | 1.58 |
| SigmaQuoridor | Second | Loss | 39 | 0.66 | 1.53 |

Every game ended by reaching the goal. No repetitions, ply cutoffs, illegal
moves, fallback moves, or forfeits occurred. Each saved game was independently
replayed through both our engine and the corresponding opponent engine:
legal sets, pawn locations, wall inventories, side to move, saved positions,
and terminal winners agreed throughout.

## Settings

- Standard 9×9 Quoridor, ten walls each; first player starts at the bottom.
- Our checkpoint: `experiments/training/az/colab/rl-model.pt`, 128 channels, ten residual
  blocks, 19 input planes; unchanged weights and native MCTS.
- Our budget: 400 visits including retained visits, at least one new simulation.
- [Gorisanson](https://github.com/gorisanson/quoridor-ai): original JavaScript
  heuristic MCTS, Strong preset, 60,000 rollouts and UCT 0.2.
- [SigmaQuoridor](https://github.com/bartolomeo3000/SigmaQuoridor): published
  `runs/models_9x9_scratch/best.pt`, original Python MCTS, 400 **new** simulations
  per move, no noise, temperature zero. This is its Python implementation,
  not the optional C++ tournament engine or browser implementation.
- Both neural networks ran locally on CPU with one PyTorch thread. Move times
  are observed elapsed times, not a controlled hardware-performance benchmark.
- Referee limits: threefold repetition or 200 plies, neither reached. Our
  training horizon was 100 plies; the harness uses 200 for these matches.
- Sigma's internal repetition-avoidance rule remains unchanged. See the
  [harness documentation](../../EXTERNAL_BOTS.md) for budget and rules details.

The 400 settings are **not equal-work budgets**: our search counts reused visits,
Sigma adds fresh simulations, and their evaluations have different costs. The
results establish that our checkpoint can beat an independent heuristic MCTS
bot in these two games, but did not beat this independently trained neural bot.
They do not establish Elo, a general win rate, or the cause of the Sigma losses.

## Artifacts and reproduction

- [Configuration, source commits and SHA-256 hashes](config.json)
- [Summary and timing measurements](summary.json)
- [Gorisanson, our first-player win](gorisanson-pair0-ours0.json)
- [Gorisanson, our second-player win](gorisanson-pair0-ours1.json)
- [Sigma, our first-player loss](sigma-pair0-ours0.json)
- [Sigma, our second-player loss](sigma-pair0-ours1.json)

```sh
.venv/bin/python training/fetch_external_bots.py
.venv/bin/python training/external_arena.py --validate
.venv/bin/python -m unittest discover -s training -p test_external_arena.py -v
.venv/bin/python -u training/external_arena.py --out experiments/training/external-matches-repeat
```

Pre-match random validation covered 280 positions per opponent. Scripted tests
also passed straight jumps, blocked-jump diagonals and a complete goal-reaching
sequence. Match records contain each move's ID, notation, time, resulting
packed position, and our search's root value when our model moved.
