#!/bin/bash
# L4 throughput benchmark and CUDA smoke runs for the kickstart arms (before the 8-hour runs).
set -uo pipefail
cd /content/q
B=/content/kick-bench
mkdir -p "$B"
export OMP_NUM_THREADS=4
echo testing > "$B/STATUS"
python -m unittest discover -s training -p 'test_az*.py' > "$B/tests.log" 2>&1 || echo tests-failed >> "$B/STATUS"
for arm in resnet transformer; do
 echo "smoke_$arm" > "$B/STATUS"
 if [ $arm = resnet ]; then NET=(--architecture resnet --channels 256 --blocks 20); else
  NET=(--architecture transformer --width 256 --blocks 4 --heads 4 --expansion 2); fi
 AZ_TRANSFORMER_INFERENCE_BATCH=64 AZ_TRANSFORMER_TRAIN_BATCH=128 python -u training/az.py rl "${NET[@]}" \
  --out "$B/smoke-$arm" --iterations 2 --games 128 --negamax-fraction 0 --simulations 200 \
  --playout-cap randomized-40-400 --batch-size 1024 --lr 3e-4 --replay 600000 --eval-every 0 \
  --teacher /content/teacher.pt --teacher-hours 0.001 --anchor /content/teacher.pt --anchor-pairs 4 \
  --snapshot-dir "$B/smoke-$arm/snapshots" --snapshot-minutes 0.001 > "$B/smoke-$arm.log" 2>&1 || echo "smoke-$arm-failed" >> "$B/FAILURES"
done
echo benchmark > "$B/STATUS"
python -u training/benchmark_kickstart.py --teacher /content/teacher.pt --out "$B/bench.json" > "$B/bench.log" 2>&1 || echo bench-failed >> "$B/FAILURES"
echo benchmark_cudnn > "$B/STATUS"
python -u training/benchmark_kickstart.py --teacher /content/teacher.pt --cudnn-benchmark --out "$B/bench-cudnn.json" > "$B/bench-cudnn.log" 2>&1 || echo bench-cudnn-failed >> "$B/FAILURES"
echo DONE > "$B/STATUS"
