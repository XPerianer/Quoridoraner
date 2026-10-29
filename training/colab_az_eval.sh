#!/bin/bash
# Final held-out evaluation on the same persistent volume as colab_az.sh.
set -euo pipefail
cd /content/q
R=${QUORIDOR_RUNS:-/content/drive/MyDrive/quoridor-az}
until [ -f "$R/DONE" ]; do sleep 30; done
mkdir -p "$R/final"
for s in 1 128 400; do
  python training/az.py eval "$R/rl/model.pt" --pairs 100 --simulations "$s" --suites heldout --seed 2027 --out "$R/final/rl-s$s-n2000.json"
done
for n in 8000 32000; do
  python training/az.py eval "$R/rl/model.pt" --pairs 100 --simulations 400 --nodes "$n" --suites heldout --seed 2027 --out "$R/final/rl-s400-n$n.json"
done
python training/az.py eval "$R/rl/model.pt" --pairs 20 --simulations 200 --suites monitor --out "$R/final/rl-monitor.json"
echo done > "$R/final/DONE"
