# Independent opponent matches

[2026-10-29 results](measurements/external-bots-20261029/README.md): our Colab
checkpoint beat Gorisanson Strong 2–0 and lost to SigmaQuoridor 0–2, swapping
colors. All four games ended at a goal and passed replay verification.

The harness uses the final Colab checkpoint `runs/az/colab/rl-model.pt` against
unmodified third-party game/search implementations. It does not use our negamax
as a stand-in for another author's bot.

| Opponent | Source | Configuration |
|---|---|---|
| Gorisanson | [quoridor-ai](https://github.com/gorisanson/quoridor-ai), commit `e15d6e7825e6f7c2f3a41de0ecef1af7aa290f1e` | 60,000 rollouts, UCT 0.2: website's Strong preset |
| SigmaQuoridor | [SigmaQuoridor](https://github.com/bartolomeo3000/SigmaQuoridor), commit `751186344fc52ad0c29bc65922e62c6fa915f006` | `runs/models_9x9_scratch/best.pt`, original Python MCTS, 400 simulations, temperature 0, no training noise |

Our model uses its original native MCTS, a 400-visit simulation budget, CPU inference
with one PyTorch thread, tree reuse, and deterministic visit-count argmax. Retained
visits count toward our budget (at least one new simulation runs). Sigma retains
its original tree reuse and adds 400 new simulations per turn, plus an initial
root expansion when needed. These are deliberately the engines' existing budget
semantics, not an equal-work comparison. Gorisanson retains its opening heuristics,
wall pruning, rollouts, and post-search heuristics. Its RNG is seeded by the
adapter. Network weights are not retrained or adjusted for these opponents.

## Run

```sh
.venv/bin/python training/fetch_external_bots.py
.venv/bin/python training/external_arena.py --validate
.venv/bin/python -u training/external_arena.py --out training/runs/external-matches
```

Requires existing PyTorch/NumPy environment and Node. Downloaded sources,
weights and their original licenses stay in ignored `training/runs/external-bots`.
No build or third-party installation script is run. Sigma weights are loaded
with its restricted `weights_only=True` loader.

`--pairs 1` plays the standard starting position once per color against each
opponent. Further pairs use shared seeded four-ply random openings and swap
colors. `--opponents gorisanson` or `--opponents sigma` selects one opponent.
Search budgets are configurable with `--simulations`, `--goris-sims`, and
`--sigma-sims`.

## Validation and limits

- Standard 9×9 board, 10 walls per player. Pawn and wall coordinates are
  translated between engines; no opponent evaluation or move selection is replaced.
- Validation compares complete legal move sets over 280 random positions per
  opponent. Matches repeat the legal-set check before every move and reject an
  illegal selected move. There are no silent fallback moves or counted forfeits.
- Sigma uses bottom-origin coordinates and directional pawn actions; the bridge
  maps these to our destination-cell actions, including straight and diagonal jumps.
- The referee stops at a goal, a third occurrence of a position including side
  to move, or 200 plies. Repetition and ply cutoffs are reported separately from
  wins/losses. They are evaluation conventions, not universal Quoridor rules.
- Sigma internally excludes third-repetition pawn moves rather than assigning
  them a draw. Its search behavior is preserved. Geometry validation temporarily
  ignores that repetition filter; actual play retains history. A move creating
  a third occurrence ends the referee game before another opponent turn.
- Our model was trained with a 100-ply horizon; these matches use 200, matching
  Sigma's horizon. This also changes our network's remaining-plies feature.
- Equal simulation counts do not mean equal compute. Gorisanson uses rollouts;
  the neural bots use network evaluations. Every move records elapsed seconds.
- Two standard-start games are a smoke tournament, not a win-rate estimate,
  Elo rating, or evidence of general superiority. No human strength is inferred.

Each output directory contains `config.json` (settings, source commits and
our checkpoint hash), per-game JSON move records, and `summary.json`. The
JSON records include positions, move IDs, notation, search times and our root values.

## Borowski and compute scaling

The arena now also accepts `--opponents borowski`. [Results and replays](measurements/compute-and-borowski-20261029/README.md) include Borowski at 400 visits and Sigma at 1,600 and 6,400 visits. Borowski uses the original source from [the linked page](https://danielborowski.github.io/site/quoridor-ai/display.html); its adapter preserves the page candidate-list behavior, validates state translation, and rejects illegal selected moves. All 63 bot decisions in the two completed games matched a replay through the original HTML/jQuery DOM.
