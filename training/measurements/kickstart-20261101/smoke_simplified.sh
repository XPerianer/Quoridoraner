#!/bin/bash
# CUDA smoke of both arms with the simplified inputs: two short iterations, teacher and anchor.
set -uo pipefail
cd /content/q
B=/content/kick-bench
export OMP_NUM_THREADS=4
for arm in resnet transformer; do
 if [ $arm = resnet ]; then NET=(--architecture resnet --channels 256 --blocks 20); else
  NET=(--architecture transformer --width 256 --blocks 4 --heads 4 --expansion 2); fi
 AZ_TRANSFORMER_INFERENCE_BATCH=64 AZ_TRANSFORMER_TRAIN_BATCH=128 python -u training/az.py rl "${NET[@]}" \
  --feature-set simplified --out "$B/smoke-simplified-$arm" --iterations 2 --games 128 --negamax-fraction 0 \
  --simulations 200 --playout-cap randomized-40-400 --batch-size 1024 --lr 3e-4 --replay 600000 --eval-every 0 \
  --teacher /content/teacher.pt --teacher-hours 0.001 --anchor /content/teacher.pt --anchor-pairs 4 \
  --snapshot-dir "$B/smoke-simplified-$arm/snapshots" --snapshot-minutes 0.001 \
  > "$B/smoke-simplified-$arm.log" 2>&1 || echo "smoke-simplified-$arm-failed" >> "$B/FAILURES"
done
echo SIMPLIFIED_DONE > "$B/SIMPLIFIED_STATUS"
