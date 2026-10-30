# More compute against Sigma and Borowski matches

Standard 9×9, ten walls, unchanged final Colab weights. Two standard-start games per setting, colors swapped. Sigma remains fixed at 400 new simulations per move.

| Our visit budget | Multiplier | Wins–losses vs Sigma |
|---|---:|---|
| 400 | 1× | 0–2 |
| 1,600 | 4× | 0–2 |
| 6,400 | 16× | 0–2 |

Increasing search did not produce a win in these tested games. This does not establish that additional compute can never win; these are small deterministic standard-start samples. Our budget includes reused visits, while Sigma adds 400 fresh simulations each turn. Each move records elapsed time. The existing 200-ply horizon and repetition conventions remain in force.

Borowski's linked bot scored 1–1 against our model at 400 visits: our first-player win took 71 plies; our second-player loss took 55. All completed games ended at goals.

The Borowski source is pinned to `4723b0eae6e524c232bcfeb9fc49eb8742daafb6`; its five JavaScript files matched the linked website byte-for-byte. The adapter runs its original decision, A*, wall simulation and candidate-list updates. It reflects the board vertically to let the top-side AI play either color, captures UI move output, and seeds its RNG. Its original candidate pruning and list quirks are preserved. The referee checks every selected move against standard rules; no illegal move was replaced or counted as a loss. Its board-state translation passed 280 random positions. Unlike the other adapters, its candidate set is not asserted equal to the complete legal-move set.

Subdirectories contain configurations, hashes, summaries, and complete per-game move records. Reproduce with:

```sh
.venv/bin/python training/fetch_external_bots.py
.venv/bin/python training/external_arena.py --opponents borowski --out experiments/training/borowski-repeat
.venv/bin/python training/external_arena.py --opponents sigma --simulations 1600 --out experiments/training/sigma-1600-repeat
.venv/bin/python training/external_arena.py --opponents sigma --simulations 6400 --out experiments/training/sigma-6400-repeat
```

All 63 Borowski bot choices were also reproduced exactly using the original page HTML and jQuery 1.7.1 in jsdom. See `borowski-dom-verification.json` and `verify_borowski_dom.cjs`.
