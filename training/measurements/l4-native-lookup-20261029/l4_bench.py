"""L4 self-play throughput: shared-lookup off/on vs SigmaQuoridor's reported cycle numbers."""
import json, os, platform, random, sys, time
sys.path.insert(0, '/content/q/training')
import numpy as np
import torch
from az import Engine, Game, Replay, load_model, new_optimizer, play, random_opening, samples, train_steps, infer

OUT = '/content/l4-bench-native.json'
MODEL = '/content/rl-model.pt'
device = 'cuda'
model, _ = load_model(MODEL, device)
model.eval()
info = dict(gpu=torch.cuda.get_device_name(0), cpus=os.cpu_count(), torch=torch.__version__,
            machine=platform.platform(), params=sum(p.numel() for p in model.parameters()))
results = []


def dump():
    with open(OUT, 'w') as f:
        json.dump(dict(info=info, results=results), f, indent=1)


# 1. Inference ceiling with synthetic full batches (Sigma: ~90k evals/s on a 5070 Ti).
e = Engine()
for bs in (1024,):
    x = np.random.rand(bs, e.planes, 9, 9).astype(np.float32)
    for _ in range(5):
        infer(model, x, device)
    torch.cuda.synchronize()
    t = time.perf_counter(); n = 0
    while time.perf_counter() - t < 10:
        infer(model, x, device); n += bs
    rate = n / (time.perf_counter() - t)
    results.append(dict(kind='inference', batch=bs, evals_per_second=rate))
    print('infer', bs, round(rate), flush=True); dump()
e.close()


def selfplay(games, sims, cache_limit, max_plies=100, seed=7):
    e = Engine()
    e.cache_limit = cache_limit
    rng = random.Random(seed)
    gs = []
    for i in range(games):
        pos, hist = random_opening(e, rng, rng.randint(0, 8))
        g = Game(i, pos, hist, ['mcts', 'mcts']); g.learner = -1
        gs.append(g)
    t0 = time.time()
    play(e, model, gs, device, sims, max_plies, noise=True, temperature_plies=8, rng=rng)
    wall = time.time() - t0
    stats = dict(e.stats)
    plies = sum(g.ply for g in gs)
    row = dict(kind='selfplay', games=games, simulations=sims, cache_limit=cache_limit, max_plies=max_plies,
               seconds=wall, games_per_hour=games / wall * 3600, plies=plies, moves_per_second=plies / wall,
               mean_plies=plies / games, cutoffs=sum(g.result == -1 for g in gs), stats=stats)
    e.close()
    return row, gs


configs = [(1024, 200, 0), (1024, 200, 1 << 20), (1024, 800, 0), (1024, 800, 1 << 20), (2048, 800, 1 << 20)]
last_games = None
for games, sims, limit in configs:
    row, gs = selfplay(games, sims, limit)
    results.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != 'stats'}), json.dumps(row['stats']), flush=True)
    dump()
    last_games = gs

# 2. Training throughput: Sigma does 1000 steps at batch 1024 in ~105 s.
e = Engine()
replay = Replay(400000)
for g in last_games:
    for r in samples(g):
        replay.add(*r)
opt = new_optimizer(model, 3e-4)
train_steps(model, opt, replay, e, 20, 1024, device, np.random.default_rng(0))
torch.cuda.synchronize(); t = time.perf_counter()
train_steps(model, opt, replay, e, 300, 1024, device, np.random.default_rng(1))
torch.cuda.synchronize()
dt = time.perf_counter() - t
results.append(dict(kind='train', steps=300, batch=1024, seconds=dt, seconds_per_1000_steps=dt / 300 * 1000))
print('train s/1000 steps', dt / 300 * 1000, flush=True)
dump()
print('DONE', flush=True)
