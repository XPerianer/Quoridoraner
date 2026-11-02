#!/bin/bash
set -euo pipefail
cd /content/q
R=/content/transformer
trap 'echo FAILED > /content/transformer/STATUS' ERR
echo waiting_for_data > "$R/STATUS"
while [ ! -f "$R/gen.npz" ] || [ ! -f "$R/mix-data.npz" ] || [ ! -f "$R/benchmark.json" ]; do sleep 20; done
echo pretraining > "$R/STATUS"
python -u training/az.py pretrain --data "$R/gen.npz" --out "$R/pre" \
  --architecture transformer --width 192 --blocks 6 --heads 6 --expansion 2 \
  --epochs 4 --batch-size 1024 --lr .002 --eval-pairs 20 --simulations 200 > "$R/pre.log" 2>&1
echo reinforcement_learning > "$R/STATUS"
python -u training/az.py rl --init "$R/pre/model.pt" --mix-data "$R/mix-data.npz" \
  --out "$R/rl" --iterations 100000 --budget-minutes 61.2245742083 \
  --games 2048 --negamax-fraction 0 --simulations 200 --playout-cap randomized-40-400 \
  --optimizer adamw --schedule none --batch-size 1024 --lr 3e-4 --replay 600000 \
  --checkpoint-seconds 600 --eval-every 5 --eval-pairs 20 --eval-simulations 200 \
  --snapshot-dir "$R/snapshots" --snapshot-minutes 60 --prev-eval-pairs 0 > "$R/rl.log" 2>&1
echo evaluating > "$R/STATUS"
mkdir -p "$R/eval"
python -u training/evaluate_transformer.py "$R/rl/model.pt" "$R/resnet/model-h001-it00034.pt" \
  --pairs 200 --simulations 200 --seed 731 --out "$R/eval/hour1-s200.json" > "$R/eval/hour1-s200.log" 2>&1
python -u training/evaluate_transformer.py "$R/rl/model.pt" "$R/resnet/model-h001-it00034.pt" \
  --pairs 200 --simulations 1 --seed 731 --out "$R/eval/hour1-s1.json" > "$R/eval/hour1-s1.log" 2>&1
for hour in 000 002 003 004; do
  for model in "$R/resnet/model-h${hour}-"*.pt; do
    [ -f "$model" ] || continue
    python -u training/evaluate_transformer.py "$R/rl/model.pt" "$model" \
      --pairs 100 --simulations 200 --seed 731 --out "$R/eval/hour${hour}-s200.json" > "$R/eval/hour${hour}-s200.log" 2>&1
  done
done
echo DONE > "$R/STATUS"
