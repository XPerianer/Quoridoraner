# AlphaZero pipeline speed and VM recovery

These changes apply to `az.py` and `native/az.cpp`, the pipeline used by
`colab_az.sh`. The independent legacy `train.py` pipeline is unchanged.

The shared lookup has two levels:

- Native workers share up to 4,096 wall geometries' goal-distance maps. These
  depend on walls only; pawn-source distances are still calculated separately.
- All concurrent games share up to 32,768 neural evaluations. Keys hash every
  input feature, including repetition count and remaining move budget. Duplicate
  leaves in the same wave are evaluated once. Entries are invalidated when the
  network or its parameter/buffer versions change. Trees, visits and search
  histories remain separate. `metrics.json` includes cumulative cache, search,
  and inference counters in `search_stats`.

Pawn-only leaves use bounded exact proof search (four plies, at most 256 proof
nodes per attempted leaf). Goals, threefold repetition and the move limit are
checked before depth limits. Proven values bypass neural inference. Unknown
results fall back to the network. This is a limited endgame solver, not a
complete tablebase, and it can change policies and playing strength.

New RL runs default to 1,024 simultaneous games (previously 256). The A100
launch script now requests 1,024 instead of 512. This also doubles games per
iteration in that script; compare games/second and strength per hour, rather
than treating iteration time as equal work. Existing resumed runs retain their
saved game count. Memory and GPU throughput must be checked on the actual VM.

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

## Measured results and remaining target

Raw local CPU measurements: `measurements/az-pipeline-speed-20261029.json`.
32-channel, two-block model, 64 simulations, two native workers, three repeats;
median cache-on versus cache-off search throughput:

| Concurrent games | Varied openings | Shared initial position |
| --- | ---: | ---: |
| 64 | 1.49× | 7.12× |
| 256 | 1.67× | 8.35× |

This isolates the neural lookup, with the same native optimizations in both
variants. It is **not** an end-to-end 3–4× pipeline result. It excludes replay
training, durable I/O and negamax turns. Increasing concurrent games did not
materially improve varied-opening CPU throughput here. GPU measurements of the
128×10 model at 512/1024 games and persistent-storage checkpoint overhead are
still needed to establish whether the requested 3–4× target is reached.

Reproduce and measure the deployment model:

```sh
.venv/bin/python training/benchmark_az_pipeline.py
python training/benchmark_az_pipeline.py --device cuda --channels 128 --blocks 10 \
  --batches 512 1024 --out /path/to/persistent/az-throughput.json
.venv/bin/python -m unittest discover -s training -p 'test_az*.py' -v
```

The benchmark alternates variant order, warms inference, and reports individual
measurements and searches/second. It compares identical jobs and budgets at each
batch size. Compare full-run `play_seconds`, `seconds`, game counts and durable
save costs separately. Retain the same seeds and evaluation suites for strength
checks because the endgame solver changes search behavior.
