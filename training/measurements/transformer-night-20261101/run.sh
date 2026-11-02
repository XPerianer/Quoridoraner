#!/bin/bash
set -euo pipefail
cd /content/q
R=/content/transformer-night
S=$R/snapshots
mkdir -p "$S" "$R/eval" "$R/resnet"
export AZ_TRANSFORMER_INFERENCE_BATCH=64
export AZ_TRANSFORMER_TRAIN_BATCH=128
export OMP_NUM_THREADS=4
trap 'echo FAILED > /content/transformer-night/STATUS' ERR
{
  date -u
  python -c 'import torch;print(torch.__version__,torch.cuda.get_device_name())'
  env | sort | grep -E '^(AZ_TRANSFORMER_|OMP_NUM_THREADS=)'
  sha256sum training/az.py training/evaluate_transformer.py
} > "$R/environment.txt"
echo generating > "$R/STATUS"
if [ ! -f "$R/gen.npz" ]; then
python -u training/az.py gen --out "$R/gen.npz" --games 100000 --parallel 1000 --seed 1 > "$R/gen.log" 2>&1
fi
echo pretraining > "$R/STATUS"
if [ ! -f "$R/pre/model.pt" ]; then
python -u training/az.py pretrain --data "$R/gen.npz" --out "$R/pre" \
 --architecture transformer --width 256 --blocks 4 --heads 4 --expansion 2 \
 --epochs 4 --batch-size 1024 --lr .002 --seed 1 --eval-pairs 20 --simulations 200 > "$R/pretrain.log" 2>&1
fi
while [ ! -f "$R/INPUTS_READY" ]; do sleep 10; done
for hour in 1 2 4 8 12; do
 echo "training_to_${hour}h" > "$R/STATUS"
 python -u training/az.py rl --init "$R/pre/model.pt" --mix-data "$R/mix-data.npz" \
 --out "$R/rl" --iterations 100000 --budget-minutes "$((hour*60))" \
 --games 2048 --negamax-fraction 0 --simulations 200 --playout-cap randomized-40-400 \
 --optimizer adamw --schedule none --batch-size 1024 --lr 3e-4 --replay 600000 \
 --checkpoint-seconds 600 --eval-every 5 --eval-pairs 20 --eval-simulations 200 \
 --snapshot-dir "$S" --snapshot-minutes 60 --snapshot-state --latest-minutes 20 \
 --prev-eval-pairs 0 --seed 1 >> "$R/rl.log" 2>&1
 # Separate evaluation process: no overlap with training on the GPU.
 cp "$R/rl/model.pt" "$S/stage-h$(printf '%03d' "$hour").pt"
 cp "$R/rl/metrics.json" "$S/metrics.json"
 echo "evaluating_${hour}h" > "$R/STATUS"
 for opponent_hour in "$hour" 13; do
  for opponent in "$R/resnet/model-h$(printf '%03d' "$opponent_hour")-"*.pt; do
   [ -f "$opponent" ] || continue
   output="$R/eval/transformer-h$(printf '%03d' "$hour")-vs-$(basename "$opponent" .pt).json"
   [ -f "$output" ] && continue
   python -u training/evaluate_transformer.py "$S/stage-h$(printf '%03d' "$hour").pt" "$opponent" \
    --pairs 200 --simulations 200 --seed 731 --out "$output" > "${output%.json}.log" 2>&1
  done
 done
 # Each checkpoint also faces the fixed h001 anchor to reveal longitudinal gains.
 if [ "$hour" != 1 ]; then
  opponent=$(echo "$R"/resnet/model-h001-*.pt)
  output="$R/eval/transformer-h$(printf '%03d' "$hour")-vs-$(basename "$opponent" .pt).json"
  if [ ! -f "$output" ]; then
   python -u training/evaluate_transformer.py "$S/stage-h$(printf '%03d' "$hour").pt" "$opponent" \
    --pairs 200 --simulations 200 --seed 731 --out "$output" > "${output%.json}.log" 2>&1
  fi
 fi
done
echo DONE > "$R/STATUS"
date -u > "$R/DONE"
