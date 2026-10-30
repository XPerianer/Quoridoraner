#!/bin/bash
# Gumbel vs PUCT, budget-matched: one arm per L4 VM. bash colab_gumbel.sh <arm> <minutes> [rl flags...]
# Every arm starts from /content/pre-model.pt with the colab_az.sh RL settings and stops starting
# iterations after <minutes> of self-play and training (in-run evaluations excluded), then plays
# the held-out evaluation with the same PUCT search for every arm. Results in /content/runs.
# Resumable: rerun after uploading a saved runs/<arm>/state.pt.
set -e
cd /content/q
R=/content/runs
ARM=$1
MINUTES=$2
shift 2
mkdir -p $R/evals
# Identical on every VM (fixed seed); only seeds a quarter of the replay buffer with supervised rows.
[ -f $R/gen.npz ] || python training/az.py gen --out $R/gen.npz --games 30000 --parallel 1000 > $R/gen.log 2>&1
[ -f $R/$ARM/DONE ] || {
  python training/az.py rl --init /content/pre-model.pt --mix-data $R/gen.npz --out $R/$ARM --iterations 1000 \
      --budget-minutes $MINUTES --games 2048 --negamax-fraction 0.5 --eval-every 5 --eval-pairs 20 \
      --eval-simulations 200 --batch-size 1024 --lr 3e-4 --replay 600000 --checkpoint-seconds 600 "$@" \
      >> $R/$ARM.log 2>&1
  touch $R/$ARM/DONE
}
for run in "1 2000" "128 2000" "128 32000"; do
  read s n <<< "$run"
  [ -f $R/evals/$ARM-s$s-n$n.json ] || python training/az.py eval $R/$ARM/model.pt --pairs 100 --simulations $s \
      --nodes $n --suites heldout --seed 2027 --out $R/evals/$ARM-s$s-n$n.json >> $R/evals.log 2>&1
done
echo done > $R/DONE
