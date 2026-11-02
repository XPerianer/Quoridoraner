#!/bin/bash
# Kickstarted student from random weights (no negamax pretraining or replay seed), simplified 10-channel
# inputs, one arm per L4: bash run.sh <resnet|transformer> [hours]
# Teacher and anchor: frozen ResNet h013 at /content/teacher.pt. The teacher weight decays linearly
# from 1 to 0 over the first 4 training hours; then pure self-play. Hourly snapshots play the anchor.
# Afterwards, the 1/2/4/8 h snapshots play h013 with the transformer-night protocol (200 pairs,
# seed 731, 200 simulations), and the last one also at equal time per move.
# Re-running resumes: from state.pt on the same VM, or from snapshots/state-latest.pt.gz on a new one.
set -euo pipefail
ARM=$1
HOURS=${2:-8}
cd /content/q
R=/content/kick-$ARM
S=$R/snapshots
mkdir -p "$S" "$R/eval"
export OMP_NUM_THREADS=4
export AZ_CUDNN_BENCHMARK=1
case $ARM in
 resnet)
  NET=(--architecture resnet --channels 256 --blocks 20 --feature-set simplified)
  # bench-20261101: chunks of 256 rows, 14.5k vs 12.0k positions/s at waves of 1,024.
  export AZ_RESNET_INFERENCE_BATCH=${AZ_RESNET_INFERENCE_BATCH:-256} ;;
 transformer)
  # Speed-matched to the ResNet arm (bench-transformers): 15.9k positions/s, 340 ms per training step.
  NET=(--architecture transformer --width 512 --blocks 6 --heads 8 --expansion 2 --feature-set simplified)
  export AZ_TRANSFORMER_INFERENCE_BATCH=${AZ_TRANSFORMER_INFERENCE_BATCH:-64}
  export AZ_TRANSFORMER_TRAIN_BATCH=${AZ_TRANSFORMER_TRAIN_BATCH:-64} ;;
 *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac
trap 'echo FAILED > "$R/STATUS"' ERR
{
 date -u
 python -c 'import torch;print(torch.__version__,torch.cuda.get_device_name())'
 nproc
 env | sort | grep -E '^(AZ_|OMP_NUM_THREADS=)' || true
 sha256sum training/az.py training/native/az.cpp training/evaluate_transformer.py training/evaluate_timed.py /content/teacher.pt
} > "$R/environment-$(date -u +%Y%m%dT%H%M%S).txt"
START=()
if [ ! -f "$R/rl/state.pt" ] && [ -f "$S/state-latest.pt.gz" ]; then
 START=(--resume-from "$S/state-latest.pt.gz")
fi
echo training > "$R/STATUS"
python -u training/az.py rl "${START[@]}" "${NET[@]}" --out "$R/rl" --iterations 100000 \
 --budget-minutes "$((HOURS*60))" --games 2048 --negamax-fraction 0 --simulations 200 \
 --playout-cap randomized-40-400 --optimizer adamw --schedule none --batch-size 1024 --lr 3e-4 \
 --replay 600000 --reuse 4 --seed 1 --checkpoint-seconds 600 --eval-every 5 --eval-pairs 20 \
 --eval-simulations 200 --teacher /content/teacher.pt --teacher-weight 1 --teacher-hours 4 \
 --anchor /content/teacher.pt --anchor-pairs 100 --anchor-seed 731 \
 --snapshot-dir "$S" --snapshot-minutes 60 --snapshot-state --latest-minutes 20 --prev-eval-pairs 0 \
 >> "$R/rl.log" 2>&1
cp "$R/rl/model.pt" "$R/rl/metrics.json" "$S/"
echo evaluating > "$R/STATUS"
for hour in 1 2 4 $HOURS; do
 for model in "$S"/model-h$(printf '%03d' "$hour")-*.pt; do
  [ -f "$model" ] || continue
  output="$R/eval/$ARM-h$(printf '%03d' "$hour")-vs-h013.json"
  [ -f "$output" ] || python -u training/evaluate_transformer.py "$model" /content/teacher.pt \
   --pairs 200 --simulations 200 --seed 731 --out "$output" > "${output%.json}.log" 2>&1
 done
done
# Equal thinking time: the final model, 50 ms per move for both sides.
output="$R/eval/$ARM-final-vs-h013-t50ms.json"
[ -f "$output" ] || python -u training/evaluate_timed.py "$R/rl/model.pt" /content/teacher.pt \
 --pairs 100 --milliseconds 50 --seed 731 --out "$output" > "${output%.json}.log" 2>&1
echo DONE > "$R/STATUS"
date -u > "$R/DONE"
