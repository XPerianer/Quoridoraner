# Experiment outputs

Everything in this folder except this README is gitignored. Results worth
keeping in git are copied into `training/measurements/` with a README.

| Folder | Contents | Written by |
| --- | --- | --- |
| `training/` | Checkpoints, replay buffers, metrics, match logs, external bots | `training/*.py` (`--out` defaults), dashboard reads it |
| `selfplay/` | Engine self-play games and reports | `npm run selfplay` |
| `colab/` | Raw Google Drive / Colab downloads (zips, sweep JSON) | manual |
