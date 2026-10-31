"""Equal-simulation inference comparison on zero-wall positions from generated games.

Replay stores only the current position repetition count, so these queries use that
count and an otherwise fresh history. The full RL arm retains complete histories.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from az import Engine, Replay, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--positions', type=int, default=128)
    p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--modes', nargs='+', default=['legacy', 'exact'])
    args = p.parse_args()
    torch.set_num_threads(1)
    replay = Replay.load(args.data)
    eligible = np.flatnonzero(((replay.position[:replay.size, 2] >> 14) & 255) == 0)
    chosen = np.random.default_rng(19).permutation(eligible)[:args.positions]
    jobs = [dict(tree=j, position=tuple(replay.position[i]),
                 history={tuple(replay.position[i]): int(replay.repetitions[i])},
                 ply=int(round(100 * (1 - replay.remaining[i]))), max_plies=100) for j, i in enumerate(chosen)]
    model, _ = load_model(args.init, 'cuda')
    results = dict(eligible_rows=len(eligible), positions=len(jobs), simulations=args.simulations,
                   history='current repetition count only; other positions unvisited', rows=[])
    if jobs:
        engine = Engine()
        engine.search(model, jobs[:8], 8, 'cuda')  # compile and capture before timing
        for repeat in range(3):
            for mode in (args.modes if repeat % 2 == 0 else list(reversed(args.modes))):
                engine.clear()
                engine.clear_cache()
                engine.configure(race_solver=mode)
                before = dict(engine.stats)
                torch.cuda.synchronize()
                start = time.perf_counter()
                policies, values = engine.search(model, jobs, args.simulations, 'cuda')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                after = dict(engine.stats)
                row = dict(mode=mode, repeat=repeat, seconds=elapsed,
                           stats={k: after[k] - before[k] for k in after},
                           mean_value=float(np.mean(values)), actions=policies.argmax(1).tolist())
                results['rows'].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'actions'}), flush=True)
                args.out.write_text(json.dumps(results, indent=2))
        engine.close()
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
