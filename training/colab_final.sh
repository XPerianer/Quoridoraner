#!/bin/bash
# Final RL run: the rl-ablations-20261030 recommendation (AdamW, constant lr 3e-4, randomized 40/400
# playout caps) from the same pretrained model, on one L4. Usage: bash colab_final.sh [hours]
# Pure self-play by default: the opponent ablation (AZ_REPORT.md) found negamax games add nothing
# over self-play; NEGAMAX_FRACTION=0.5 restores the mix the RL ablations used.
#
# The live state.pt (~330 MB, rewritten every iteration) stays on the VM disk. Every hour of self-play
# and training, a model (~13 MB) and a gzipped full resumable state (~55 MB) go to $SNAP, which must be
# persistent; state-latest.pt.gz there is refreshed every 20 minutes. On a fresh VM, upload
# /content/pre-model.pt and re-run: it resumes from state-latest (losing at most 20 minutes).
# Set SNAPSHOT_STATE=0 to keep only the models.
# Each hourly model also plays 200 games against the previous one (the first against pre-model.pt,
# kept as model-h000-it00000.pt); results are `prev_eval` in metrics.json.
set -euo pipefail
cd /content/q
HOURS=${1:-60}
SNAP=${QUORIDOR_SNAPSHOTS:-/content/drive/MyDrive/quoridor-final}
R=/content/final
if [ -z "${QUORIDOR_SNAPSHOTS:-}" ] && [ ! -d /content/drive/MyDrive ]; then
    echo 'Mount Google Drive before starting, or set QUORIDOR_SNAPSHOTS to a persistent folder.' >&2
    exit 1
fi
mkdir -p "$SNAP" "$R"
START=()
if [ ! -f "$R/rl/state.pt" ]; then
    # state-latest (every 20 training minutes) is never older than the kept hourly states.
    LATEST=$(ls "$SNAP"/state-latest.pt.gz 2>/dev/null || ls "$SNAP"/state-h*.pt.gz 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "$LATEST" ]; then
        START=(--resume-from "$LATEST")
    else
        # Seeds a quarter of the replay buffer, as in the ablations; only needed for a fresh start.
        [ -f "$R/gen.npz" ] || python training/az.py gen --out "$R/gen.npz" --games 30000 --parallel 1000 >> "$SNAP/gen.log" 2>&1
        START=(--init /content/pre-model.pt --mix-data "$R/gen.npz")
    fi
fi
STATE=()
[ "${SNAPSHOT_STATE:-1}" = 1 ] && STATE=(--snapshot-state)
# Drive may upload a file only when it is closed, so log locally and copy the log every 5 minutes.
LOG=$R/rl.log
[ -f "$SNAP/rl.log" ] && [ ! -f "$LOG" ] && cp "$SNAP/rl.log" "$LOG"
(while sleep 300; do cp "$LOG" "$SNAP/rl.log.tmp" && mv "$SNAP/rl.log.tmp" "$SNAP/rl.log"; done) &
COPIER=$!
trap 'kill $COPIER 2>/dev/null; cp "$LOG" "$SNAP/rl.log"' EXIT
python training/az.py rl "${START[@]}" --out "$R/rl" --iterations 100000 --budget-minutes $((HOURS * 60)) \
    --games 2048 --negamax-fraction "${NEGAMAX_FRACTION:-0}" --simulations 200 --playout-cap randomized-40-400 \
    --optimizer adamw --schedule none --batch-size 1024 --lr 3e-4 --replay 600000 --checkpoint-seconds 600 \
    --eval-every 5 --eval-pairs 20 --eval-simulations 200 \
    --snapshot-dir "$SNAP" --snapshot-minutes 60 --latest-minutes 20 --prev-eval-pairs 100 "${STATE[@]}" >> "$LOG" 2>&1
cp "$R/rl/model.pt" "$R/rl/metrics.json" "$SNAP/"
echo done > "$SNAP/DONE"
