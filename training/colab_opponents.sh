#!/bin/bash
# Opponent ablation, one arm per VM: bash colab_opponents.sh <negamax|selfplay|mixed> <negamax fraction> [pretrain]
# RL from the same pretrained model (/content/pre-model.pt) for 20 iterations with the colab_az.sh settings, then
# held-out evaluation of that arm (and of the pretrained model when "pretrain" is given). Results in /content/runs.
# Resumable: rerun on a fresh VM after uploading a saved runs/<arm>/ directory, or continue from a saved model
# with INIT=<model.pt> ITERATIONS=<remaining> SEED=<new seed>.
set -e
cd /content/q
R=/content/runs
ARM=$1
mkdir -p $R/evals
# Identical on every VM (fixed seed); only seeds a quarter of the replay buffer with supervised rows.
[ -f $R/gen.npz ] || python training/az.py gen --out $R/gen.npz --games 30000 --parallel 1000 > $R/gen.log 2>&1
[ -f $R/$ARM/DONE ] || {
  python training/az.py rl --init ${INIT:-/content/pre-model.pt} --seed ${SEED:-1} --mix-data $R/gen.npz --out $R/$ARM --iterations ${ITERATIONS:-20} \
      --games 512 --negamax-fraction $2 --simulations 200 --eval-every 5 --eval-pairs 20 --eval-simulations 200 \
      --batch-size 1024 --lr 3e-4 --replay 600000 >> $R/$ARM.log 2>&1
  touch $R/$ARM/DONE
}
evaluate() {
  for run in "1 2000" "128 2000" "128 32000"; do
    read s n <<< "$run"
    [ -f $R/evals/$2-s$s-n$n.json ] || python training/az.py eval $1 --pairs 100 --simulations $s --nodes $n \
        --suites heldout --seed 2027 --out $R/evals/$2-s$s-n$n.json >> $R/evals.log 2>&1
  done
}
evaluate $R/$ARM/model.pt $ARM
[ "$3" = pretrain ] && evaluate /content/pre-model.pt pretrain
echo done > $R/DONE
