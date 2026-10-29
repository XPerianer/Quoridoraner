#!/bin/bash
# Held-out evaluation for the opponent ablation (seed 2027): each model vs negamax, then a round robin.
# Expects /content/models/{pretrain,negamax,mixed,selfplay}.pt. One result file per job, skipped when present.
cd /content/q
M=/content/models
E=/content/runs/evals
mkdir -p $E
names="pretrain negamax mixed selfplay"
for name in $names; do
  for run in "1 2000" "128 2000" "128 32000"; do
    read s n <<< "$run"
    [ -f $E/$name-s$s-n$n.json ] || python training/az.py eval $M/$name.pt --pairs 100 --simulations $s --nodes $n \
        --suites heldout --seed 2027 --out $E/$name-s$s-n$n.json >> $E/evals.log 2>&1
  done
done
set -- $names
for a in "$@"; do
  shift
  for b in "$@"; do
    [ -f $E/h2h-$a-$b.json ] || python training/az.py h2h $M/$a.pt $M/$b.pt --names $a $b --pairs 100 \
        --simulations 128 --suite heldout --seed 2027 --out $E/h2h-$a-$b.json >> $E/evals.log 2>&1
  done
done
echo done > /content/runs/DONE
