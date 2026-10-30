# Self-play harness

Plays engine configurations against a baseline and measures the Elo difference.

```sh
npm run selfplay -- selfplay/experiments/wall-weight.json             # run an experiment
npm run selfplay -- selfplay/experiments/smoke.json --pairs 2          # override pair count
npm run selfplay:report -- experiments/selfplay/<run-dir>                  # rebuild report from saved games
```

## Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `<experiment.json>` | required | Experiment config (see below) |
| `--pairs N` | config value | Opening pairs per candidate (each pair = 2 games, colours swapped) |
| `--concurrency N` | CPU cores − 1 | Games played in parallel (worker threads) |
| `--out DIR` | `experiments/selfplay` | Parent folder for run directories |

## Experiment config

```jsonc
{
  "name": "wall-weight",            // required, used in the run directory name
  "pairs": 60,                      // openings per candidate; each played with both colours
  "openingPlies": 4,                // random plies from the start position
  "openingWallProb": 0.15,          // chance an opening ply is a random wall
  "seed": 1,                        // openings are reproducible from this
  "maxPlies": 200,                  // draw after this many plies (also draw on threefold repetition)
  "base": { "maxNodes": 50000 },    // engine overrides shared by baseline and candidates
  "baseline": { "name": "baseline", "params": {} },
  "candidates": [{ "name": "ww5", "params": { "wallWeight": 5 } }],
  "sweep": { "wallWeight": [0, 5, 10] }   // cartesian product → one candidate per combination
}
```

Every field except `name` has a default (see `config.ts`). Unknown parameter names are rejected. Swept candidates start from the baseline's `params`.

Engine parameters (`SearchOptions` in `src/engine/types.ts`, defaults in `DEFAULT_OPTIONS`):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `maxNodes` | harness: 50 000 | Node budget per move; deterministic, so runs are reproducible |
| `timeMs` | harness: 1e9 | Time budget per move; use instead of `maxNodes` to compare configs whose nodes cost differently (e.g. `moveGen`) |
| `maxDepth` | harness: 64 | Iterative deepening limit |
| `moveGen` | `all` | `all` legal walls, or `likely` (walls near pawns/walls) |
| `exactRoot` | `false` | Full-window search of every root move |
| `wallWeight` | `10` | Eval points per wall-in-hand difference (one path step = 10) |
| `tempo` | `0` | Eval bonus for the side to move |

To add a new hyperparameter: add it to `SearchOptions` and `DEFAULT_OPTIONS`, use it in `search.ts`; the harness picks it up automatically.

## Outputs

Each run writes `experiments/selfplay/<timestamp>-<name>/`:

- `config.json`: the config, every engine's fully resolved parameters, git commit and whether the tree was dirty
- `games.jsonl`: one line per finished game (players, opening, every move with depth / nodes / score / time, result, termination), appended as games finish, so an interrupted run keeps its games
- `summary.json`: per-candidate statistics
- `report.html`: self-contained report (results table, Elo with CI, search depth, sweep plot, all games)

## Reading the results

- Each opening is played twice with colours swapped; the **pair** is the unit of variance, which cancels most of the opening and first-move bias.
- **Elo [95% CI]** is the candidate's strength relative to the baseline. `∞` means the data cannot bound it yet (score interval reaches 0% or 100%): play more pairs.
- **LOS** is the probability that the candidate is genuinely stronger. Treat > 95% as a real improvement.
- Rough guide (pair score std ≈ 0.35): a 95% CI of ±50 Elo needs ~100 pairs, ±20 Elo ~600 pairs.
- With a node budget, compare search-speed changes (e.g. `moveGen`) with `timeMs` instead, since cheaper nodes are the point of such changes.
