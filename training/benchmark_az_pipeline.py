"""Measure shared lookup gains and batching on this machine (no strength claim)."""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import torch
from az import Engine, Net, START, random_opening


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='experiments/training/az-pipeline-benchmark.json')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batches', nargs='+', type=int, default=[64, 256])
    p.add_argument('--simulations', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--blocks', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = Net(19, args.channels, args.blocks).to(args.device).eval()
    results = []
    for n in args.batches:
        for workload in ('shared', 'varied'):
            times = {False: [], True: []}
            for repeat in range(args.repeats):
                # Alternate order to reduce warmup/temperature bias.
                for cached in ([False, True] if repeat % 2 == 0 else [True, False]):
                    e = Engine(2)
                    if not cached:
                        e.cache_limit = 0
                    rng = random.Random(4)
                    positions = [START] * n if workload == 'shared' else [
                        random_opening(e, rng, rng.randrange(9))[0] for _ in range(n)]
                    jobs = [dict(tree=i, position=pos, history={pos: 1}, ply=0, max_plies=100)
                            for i, pos in enumerate(positions)]
                    # Warm up model, then drop all retained roots and cache entries.
                    e.search(model, jobs[:1], 2, args.device)
                    e.clear()
                    e.clear_cache()
                    start = time.perf_counter()
                    e.search(model, jobs, args.simulations, args.device)
                    elapsed = time.perf_counter() - start
                    times[cached].append(elapsed)
                    results.append(dict(batch=n, workload=workload, cached=cached, repeat=repeat,
                                        seconds=elapsed, searches_per_second=n / elapsed, stats=e.stats))
                    e.close()
            print(json.dumps(dict(batch=n, workload=workload,
                                  speedup=statistics.median(times[False]) / statistics.median(times[True]))), flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(args=vars(args), machine=platform.platform(), torch=torch.__version__,
                                   results=results), indent=2))


if __name__ == '__main__':
    main()
