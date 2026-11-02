#!/bin/bash
set -euo pipefail
cd /content/q
R=/content/distillation-pilot
N=/content/transformer-night
TEACHER=$N/resnet/model-h013-it00451.pt
mkdir -p "$R/eval"
[ "$(cat "$N/STATUS")" = DONE ]
export AZ_TRANSFORMER_INFERENCE_BATCH=64
export OMP_NUM_THREADS=4
trap 'echo FAILED > /content/distillation-pilot/STATUS' ERR
sha256sum training/distill.py training/evaluate_timed.py training/evaluate_transformer.py training/native/az_timed.cpp > "$R/source-sha256.txt"
echo cuda_smoke > "$R/STATUS"
for architecture in transformer resnet; do
 python -u training/distill.py --teacher "$TEACHER" --replay "$N/rl/state.pt" \
  --out "$R/smoke-$architecture" --student "$architecture" --minutes .02 --batch 64 > "$R/smoke-$architecture.log" 2>&1
done
python -u training/evaluate_timed.py "$R/smoke-transformer/model.pt" "$TEACHER" \
 --pairs 1 --max-plies 6 --milliseconds 5 --out "$R/timed-smoke.json" > "$R/timed-smoke.log" 2>&1
echo benchmarking > "$R/STATUS"
python -u training/benchmark_transformer.py --models transformer-128x2 resnet-64x4 resnet \
 --out "$R/benchmark.json" --train-batch 512 --train-repeats 10 --repeats 50 > "$R/benchmark.log" 2>&1
for candidate in transformer resnet control; do
 echo "training_$candidate" > "$R/STATUS"
 architecture=$candidate
 EXTRA=(--hybrid-weight .1)
 if [ "$candidate" = control ]; then architecture=transformer; EXTRA=(--teacher-weight 0 --hybrid-weight 1); fi
 python -u training/distill.py --teacher "$TEACHER" --replay "$N/rl/state.pt" \
  --out "$R/$candidate" --student "$architecture" --minutes 20 --batch 512 "${EXTRA[@]}" > "$R/$candidate.log" 2>&1
 echo "evaluating_$candidate" > "$R/STATUS"
 python -u training/evaluate_transformer.py "$R/$candidate/model.pt" "$TEACHER" \
  --pairs 100 --simulations 200 --seed 731 --out "$R/eval/$candidate-s200.json" > "$R/eval/$candidate-s200.log" 2>&1
 python -u training/evaluate_timed.py "$R/$candidate/model.pt" "$TEACHER" \
  --pairs 100 --milliseconds 20 --seed 731 --out "$R/eval/$candidate-t20ms.json" > "$R/eval/$candidate-t20ms.log" 2>&1
done
echo DONE > "$R/STATUS"
date -u > "$R/DONE"
