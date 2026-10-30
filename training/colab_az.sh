#!/bin/bash
# Mount Drive first, or set QUORIDOR_RUNS to another persistent volume.
set -euo pipefail
cd /content/q
R=${QUORIDOR_RUNS:-/content/drive/MyDrive/quoridor-az}
if [ -z "${QUORIDOR_RUNS:-}" ] && [ ! -d /content/drive/MyDrive ]; then
    echo 'Mount Google Drive before starting, or set QUORIDOR_RUNS to a persistent volume.' >&2
    exit 1
fi
mkdir -p "$R"
[ -f "$R/gen.npz" ] || python training/az.py gen --out "$R/gen.npz" --games 100000 --parallel 1000 >> "$R/gen.log" 2>&1
[ -f "$R/pre/model.pt" ] || python training/az.py pretrain --data "$R/gen.npz" --out "$R/pre" --channels 128 --blocks 10 \
    --epochs 4 --batch-size 1024 --lr 2e-3 --eval-pairs 20 --simulations 200 >> "$R/pre.log" 2>&1
python training/az.py rl --init "$R/pre/model.pt" --mix-data "$R/gen.npz" --out "$R/rl" --iterations 40 --games 2048 \
    --negamax-fraction 0.5 --simulations 200 --eval-every 5 --eval-pairs 20 --eval-simulations 200 \
    --batch-size 1024 --lr 3e-4 --replay 600000 --checkpoint-seconds 600 >> "$R/rl.log" 2>&1
echo done > "$R/DONE"
