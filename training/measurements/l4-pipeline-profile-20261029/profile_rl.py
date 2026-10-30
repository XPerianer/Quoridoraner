"""Profile the real `az.py rl` loop on one GPU: where does wall time go?"""
import argparse, json, os, random, subprocess, sys, threading, time
from collections import defaultdict
sys.path.insert(0, '/content/q/training')
import numpy as np
import psutil
import torch
import az

GAMES, SIMS, ITERS, EVAL, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
OUT = f'/content/profile-{TAG}.json'
T, N = defaultdict(float), defaultdict(int)
phase = ['setup']
batches = []
gpu_busy = [0.0]
samples = []


def timed(name, fn, is_phase=False):
    def wrapper(*a, **k):
        if is_phase:
            phase.append(name)
        key = '/'.join(phase[1:] + ([] if is_phase else [name]))
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[key] += time.perf_counter() - t
            N[key] += 1
            if is_phase:
                phase.pop()
    return wrapper


for name in ('negamax', 'begin', 'take', 'finished', 'advance', 'advance_many', 'history', 'legal', 'features', 'snapshot', 'drop', 'launch', 'collect'):
    setattr(az.Engine, name, timed(name, getattr(az.Engine, name)))
az.random_opening = timed('random_opening', az.random_opening)
az.save_rl_state = timed('save_rl_state', az.save_rl_state)
az.save_model = timed('save_model', az.save_model)
for name in ('train_steps', 'match', 'play'):
    setattr(az, name, timed(name, getattr(az, name), is_phase=True))


def monitor():
    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-lms', '500'],
                           stdout=subprocess.PIPE, text=True)
    psutil.cpu_percent(percpu=True)
    for line in smi.stdout:
        samples.append((time.time(), '/'.join(phase[1:]), int(line.strip()), sum(psutil.cpu_percent(percpu=True)) / 100))


threading.Thread(target=monitor, daemon=True).start()


def dump(extra=None):
    util = defaultdict(list)
    for _, p, g, c in samples:
        util[p].append((g, c))
    json.dump(dict(times=dict(T), counts=dict(N),
                   util={p: dict(gpu=float(np.mean([g for g, _ in v])), cpus=float(np.mean([c for _, c in v])), n=len(v))
                         for p, v in util.items()},
                   samples=samples, extra=extra), open(OUT, 'w'), indent=1)


t = time.time()
sys.argv = ['az.py', 'rl', *sys.argv[6:], '--init', '/content/rl-model.pt', '--out', f'/content/prof-{TAG}', '--iterations', ITERS,
            '--games', str(GAMES), '--negamax-fraction', '0.5', '--simulations', str(SIMS), '--eval-every', EVAL,
            '--eval-pairs', '20', '--eval-simulations', '200', '--batch-size', '1024', '--lr', '3e-4',
            '--replay', '600000', '--checkpoint-seconds', '600']
az.main()
wall = time.time() - t
dump(dict(wall=wall))
print('rl wall', wall, flush=True)

dump(dict(wall=wall, metrics=json.load(open(f'/content/prof-{TAG}/metrics.json'))))
print('DONE', flush=True)
