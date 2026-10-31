"""L4 optimizer overhead on identical replay batches, with fresh models per repeat."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from az import Engine, Replay, load_model, new_rl_optimizer, train_steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(1)
    engine = Engine()
    replay = Replay.load(args.data)
    rows = []
    for repeat in range(args.repeats):
        for name in (['adamw', 'muon'] if repeat % 2 == 0 else ['muon', 'adamw']):
            torch.manual_seed(17)
            model, _ = load_model(args.init, 'cuda')
            model.to(memory_format=torch.channels_last)
            optimizer = new_rl_optimizer(model, 3e-4, name)
            rng = np.random.default_rng(17)
            train_steps(model, optimizer, replay, engine, 20, 1024, 'cuda', rng)
            torch.cuda.synchronize()
            start = time.perf_counter()
            losses = train_steps(model, optimizer, replay, engine, args.steps, 1024, 'cuda', rng)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            rows.append(dict(optimizer=name, repeat=repeat, steps=args.steps, seconds=elapsed,
                             milliseconds_per_step=elapsed * 1000 / args.steps, losses=losses))
            args.out.write_text(json.dumps(rows, indent=2))
            print(json.dumps(rows[-1]), flush=True)
            del optimizer, model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
