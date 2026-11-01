# Model input encodings

New `az.py pretrain` and `az.py rl` models accept `--feature-set full` (default, 19 channels) or `--feature-set simplified` (10 channels). For example:

```sh
python training/az.py rl --out training/runs/simplified --feature-set simplified --channels 128 --blocks 10
```

The simplified order is:

| New channel | Original channel | Feature |
|---|---|---|
| 0–1 | 0–1 | Current player and opponent pawn positions |
| 2–3 | 2–3 | Horizontal and vertical walls |
| 4–5 | 8–9 | Both distance-to-goal maps, divided by 16 |
| 6–7 | 10–11 | Both binary shortest-path masks |
| 8–9 | 12–13 | Both remaining wall counts, divided by 10 |

Distances and masks use wall geometry, ignoring pawn blocking and jumps. Perspective canonicalization and training mirroring are unchanged.

Checkpoints store their input channel count. Loading, evaluating, initializing from, or resuming an existing checkpoint uses its saved encoding, including the 19-channel h013 model. `--feature-set` only controls newly initialized models; it does not convert h013 weights into a 10-channel network. Full and simplified models can play against each other and share replay data.

The native encoder still produces 19 channels. A 10-channel model selects original channels `0,1,2,3,8,9,10,11,12,13` before its stem. Its stem has 10 input channels and trains only on these features; native feature generation and host-to-device transfer costs remain unchanged. Direct model calls also accept an already compact 10-channel tensor in the order above.

`--drop-planes` uses indices within the selected encoding: shortest-distance and path-mask channels are 4–7 in simplified mode.
