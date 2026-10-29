# Engine performance and algorithm development

## Wall generation and search optimizations

The TypeScript and native training engines implement the techniques described in
[Solving Quoridor](https://grantslatton.com/solving-quoridor):

- The existing wall-contact shortcut skips reachability checks unless a wall
  touches obstacles at two or more points. Differential tests verify it against
  exhaustive path checks.
- Wall occupancy bitboards filter overlaps and crossings in bulk. TypeScript
  uses two 32-bit limbs per orientation; C++ uses a 64-bit integer.
- Interior alpha-beta nodes defer reachability checks until evaluation, terminal
  detection, or the first invalid child. Illegal branches never receive a score
  or enter the transposition table. Root moves and displayed replies remain
  strictly legal; MCTS always uses strict move generation.
- Transposition entries share horizontally reflected positions, with cached
  moves mapped back to the current orientation. Both hashes are verified.
  TypeScript invalidates cached scores when evaluation or move-generation
  options change.
- Move ordering combines cached moves, pawn goal distance, cutoff history, and
  walls cutting an opponent's shortest route (penalizing cuts to our own route).
  The extra wall analysis runs only above the last search layer to limit overhead.

A before/after check on corpus indices 0, 2, 5, 9 at depth 4, all moves, cold
tables, and unlimited practical budgets measured **98.12 ms → 71.95 ms** median
total time across five rounds after two warmups, alternating version order.
Search allocations were included. Total nodes fell **83,978 → 55,063**, with
identical scores `[0, 0, 30, 50]`. This is about 27% less elapsed time on this
small corpus, not a playing-strength measurement or a universal speedup.
Strict move generation alone can cost more due to the additional maintained
state; the benefit is measured over complete searches.

The default `Board.distance()` uses nine scalar row bitboards. Walls are loaded
once, and each synchronous wave expands all rows with integer operations. This
preserves exact shortest-path lengths, including unreachable goals. Both the web
worker and self-play harness use it automatically. No extra runtime dependency.

## Measurements

Apple M1 Pro, Node 26.3.1; medians of five measured rounds after two warmups,
rotating implementation order. `kernel-results.json` contains all samples.

| Implementation | 48,600 distances | 810 legal-move generations | Four depth-4 searches |
| --- | ---: | ---: | ---: |
| Original row arrays | 24.58 ms | 4.22 ms | 265.97 ms |
| Queue BFS | 55.36 ms | 7.82 ms | 390.00 ms |
| Fused array waves | 26.44 ms | 4.44 ms | 282.46 ms |
| Bounded array waves | 26.07 ms | 3.41 ms | 228.05 ms |
| Scalar bitboards (default) | **4.66 ms** | **1.19 ms** | **102.53 ms** |

The default delivers 5.28× pathfinding throughput, 3.55× move-generation
throughput, and 2.59× full-search throughput on this corpus. These are speed
measurements, not Elo improvements or a claim of the fastest possible engine.

The corpus contains 81 deterministic legal positions from eight random games,
including the initial position, wall-heavy positions, and positions with exhausted
wall supplies. Full searches use corpus indices 0, 2, 5, 9, all legal moves, cold
transposition tables, depth 4, and effectively unlimited time/node budgets.
Allocation of the searcher and its TT is included. Microbenchmarks reuse boards
and output buffers. Results can vary across machines, runtimes, and positions.

## MPS experiment

`mps.py` implements exact batched shortest paths using PyTorch boolean waves.
It checks CPU and MPS against the original TypeScript distances. It runs all 80
possible path-expansion steps to avoid per-wave host synchronization, with input
packing/transfers excluded and synchronized GPU timing. It is an experimental
pathfinding primitive, not a complete GPU game engine or GPU tree search.

| Positions per batch (two players each) | PyTorch CPU | PyTorch MPS |
| --- | ---: | ---: |
| 1 | 2.12 ms | 8.66 ms |
| 81 | 6.01 ms | 9.97 ms |
| 1,024 | 51.16 ms | 17.42 ms |

MPS beats this tensor CPU implementation at the largest tested batch, but its
small-batch latency is unsuitable for the current sequential alpha-beta search.
The tensor experiment does more wave work than the early-exit TypeScript kernels;
it does not establish a hardware-wide CPU/GPU limit. A custom Metal kernel,
neural evaluation, or a batched MCTS architecture would be separate experiments.
The selected default is the fastest complete-search implementation tested here.

MPS timing uses [PyTorch synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
Raw timings and the PyTorch version are in `mps-results.json`.

## Reproduce

```sh
npm test
npm run bench:kernels
# Optional, separate environment; not required by the game:
python3 -m venv .venv
.venv/bin/python -m pip install -r bench/requirements-mps.txt
.venv/bin/python bench/mps.py
```

The CPU benchmark regenerates `positions.json` for the MPS benchmark. Without
MPS, the Python benchmark records CPU results and `mpsAvailable: false`.

## Build an algorithm

```ts
import { Board, MAX_MOVES } from '../src/engine/board'

const board = Board.initial() // or Board.from(position)
const moves = new Int16Array(MAX_MOVES) // allocate one per search ply
const n = board.generate(moves, false) // false = every legal action
for (let i = 0; i < n; i++) {
  const move = moves[i]
  board.make(move)
  try {
    // Check board.winner() before expanding terminal positions.
    // Evaluate using board.distance(0/1), wallsLeft, turn, etc.
    // Recurse with a different move buffer for each ply.
  } finally {
    board.unmake(move)
  }
}
```

`make()` assumes a legal action; it does not validate arbitrary input. `generate()`
is for nonterminal positions. `distance()` ignores pawn occupancy, as required
for wall reachability; `pawnMoves()` implements jumps and diagonals. Keep a board
per search worker. Existing make/unmake history supports up to 1,024 outstanding
pawn moves. `hash1`/`hash2` provide incremental keys for a transposition table.

`TsSearcher` accepts an optional `(position: Position) => Board` factory for
experiments. The original kernel remains available as `distanceReference()`;
benchmark factories demonstrate how to substitute each candidate.

Use the existing self-play harness to measure playing strength. For speed changes,
compare equal `timeMs` budgets with a sufficiently large `maxNodes`, and paired
openings. More nodes per second alone does not establish stronger move selection.
