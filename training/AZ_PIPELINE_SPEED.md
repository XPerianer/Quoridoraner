# AlphaZero pipeline speed and VM recovery

These changes apply to `az.py` and `native/az.cpp`, the pipeline used by
`colab_az.sh`. The independent legacy `train.py` pipeline is unchanged.

The shared lookup has two levels:

- Native workers share up to 4,096 wall geometries' goal-distance maps. These
  depend on walls only; pawn-source distances are still calculated separately.
- All concurrent games share up to 1,048,576 neural evaluations in native code
  (about 1 GB when full). Keys are the board, repetition count and remaining
  plies, which fully determine the network input. Lookups happen during tree
  selection, so a hit needs no GPU round-trip; identical leaves in one wave are
  evaluated once. Eviction is first in, first out. The cache is cleared when the
  network or its parameter/buffer versions change. Trees, visits and search
  histories remain separate. `metrics.json` includes cumulative cache, search,
  and inference counters in `search_stats`.
- With one network, `play()` does not advance games in lockstep: a game plays
  its move as soon as its own search finishes, and starts the next search in the
  following wave. Each game samples noise and temperature from its own random
  stream, so this scheduling does not change any game. Checkpoints stop new
  searches, finish those in flight, and save at a move boundary. Matches between
  two networks still move all games together.
- Searches are split into two groups. While one group's batch runs on the GPU,
  the CPU backs up the other group's results and selects its next leaves.
  Inputs and outputs go through pinned host memory, so copies do not block.
- On CUDA, search uses a separate inference copy of the network: BatchNorm
  folded into the convolutions, bf16 channels-last weights, `torch.compile`
  with dynamic batch sizes, replayed as one CUDA graph per batch size. Batches
  are padded to a multiple of 64 rows. After training, new weights are loaded
  into the same copy, which needs no recompilation. Training uses
  channels-last layout on CUDA.
- An RL iteration ends once `--games` games have finished. A game that ends is
  replaced immediately, so `--games` games always run and batches stay full.
  Games still running at the boundary finish their current move and carry over
  (their trees are dropped). The end-of-iteration checkpoint includes them.
- Per-move bookkeeping is native: each tree keeps its game history in C++,
  moves and finished searches go through one call per loop pass, and root
  noise is drawn in C++ from a per-move seed.
- Negamax replies run on a separate native pool and thread. They are applied
  one loop pass later, so results do not depend on timing.

Pawn-only leaves use bounded exact proof search (four plies, at most 256 proof
nodes per attempted leaf). Goals, threefold repetition and the move limit are
checked before depth limits. Proven values bypass neural inference. Unknown
results fall back to the network. This is a limited endgame solver, not a
complete tablebase, and it can change policies and playing strength.

New RL runs default to 1,024 simultaneous games (previously 256). The Colab
launch script requests 2,048, which is about 6% faster per game on an L4 and
matches SigmaQuoridor's cycle size. It also sets `--checkpoint-seconds 600`:
iterations take about 95 s and every iteration end saves the carried games, so
in-iteration checkpoints would only add drains. Existing resumed runs retain
their saved game count.

## Recovery

Mount Drive before running the launch script. By default it writes to
`/content/drive/MyDrive/quoridor-az`; set `QUORIDOR_RUNS` for another persistent
volume. Local ephemeral VM storage cannot survive deletion. Logs append across
restarts. Re-run the same script after mounting the volume on a replacement VM.

- Generation saves replay, completed-game count, settings and RNG after each
  batch to `gen.npz.state.pt`. An interrupted batch is replayed. The final dataset
  is replaced atomically so an interrupted write cannot look like a completed
  stage.
- Pretraining saves model, optimizer, OneCycle scheduler, training cursor, logs,
  NumPy RNG and Torch/CUDA RNG after chunks of at most 200 optimizer steps, and
  before the first chunk. An interrupted chunk is replayed. Input data must also
  remain on persistent storage.
- RL saves one atomic `state.pt` containing model, optimizer, replay ring and
  cursor, settings, metrics and Python/NumPy/Torch/CUDA RNG. During rollouts it
  additionally saves every game's position, samples, repetition history, moves
  and full retained native search trees at completed move boundaries, every
  `--checkpoint-seconds` (default 60). It saves again after every iteration.
  Interrupted training/evaluation work since the last checkpoint is replayed.
  In-flight saves can exceed their cadence by one complete move batch and I/O.
- Files are flushed and fsynced before atomic replacement. An advisory run lock
  rejects a second writer. Back up `state.pt` rather than mixing separate replay
  and model generations. `model.pt` remains an inference export, not a resume
  checkpoint. Old RL checkpoints can migrate when their `replay.npz` is present;
  old files lack exact RNG history and cannot retroactively recover it.

Resume with the same `--out`. Saved learning settings are restored; device,
workers, checkpoint cadence, output location and an explicitly supplied total
iteration limit may change. Native tree bytes are for the same architecture and
engine format. CPU tests establish bitwise recovery, including a real SIGKILL
in a fresh process; different GPU kernels/batch shapes or devices may round
outputs differently. Only load checkpoints you trust (the existing Torch loader
uses Python object deserialization).

## Measured results

[Profile of the full `rl` loop on one L4](measurements/l4-pipeline-profile-20261029/README.md),
200 simulations, half the games against negamax: an iteration of 1,024 games
took 96 s before these pipeline changes; 2,048 games now take 84 s (about
2.3× the games per hour, including training).


[One L4, 128×10 network, 1,024 self-play games](measurements/l4-native-lookup-20261029/README.md):
51.4k games/h at 200 simulations and 13.8k at 800, against 27.6k and 7.3k for the
best configuration of the previous Python lookup with lockstep play (1.86× and
1.89×). SigmaQuoridor reports ~24k games/h at 800 simulations on an RTX 5070 Ti.

Earlier CPU measurement of the Python lookup alone:
`measurements/az-pipeline-speed-20261029.json`.

```sh
.venv/bin/python -m unittest discover -s training -p 'test_az*.py' -v
```
