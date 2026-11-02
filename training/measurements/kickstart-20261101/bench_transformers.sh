#!/bin/bash
# After bench.sh: larger transformer candidates (speed-matching the ResNet 256x20 arm), then CUDA smoke
# runs of both arms with the simplified inputs.
set -uo pipefail
cd /content/q
B=/content/kick-bench
export OMP_NUM_THREADS=4
while [ "$(cat "$B/STATUS")" != DONE ]; do sleep 10; done
echo benchmark_transformers > "$B/STATUS2"
python -u training/benchmark_kickstart.py --teacher /content/teacher.pt --out "$B/bench-transformers.json" \
 --models transformer-384x8 transformer-512x6 transformer-512x8 transformer-512x12 \
 --chunks 0 64 128 256 --train-chunks 64 128 256 > "$B/bench-transformers.log" 2>&1 || echo bench-transformers-failed >> "$B/FAILURES"
echo smoke_simplified > "$B/STATUS2"
bash training/measurements/kickstart-20261101/smoke_simplified.sh
echo DONE > "$B/STATUS2"
