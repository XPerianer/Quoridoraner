"""Held-out paired-color evaluation with per-game records and pair-bootstrap intervals."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from az import Engine, Game, evaluation_opening, load_model, play


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--opponent')
    p.add_argument('--pairs', type=int, default=100)
    p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--nodes', type=int, default=32000)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--device', default='cuda')
    p.add_argument('--race-solver', choices=('legacy', 'exact', 'tablebase'), default='legacy')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    engine = Engine(race_solver=args.race_solver)
    model, _ = load_model(args.model, args.device)
    opponent = load_model(args.opponent, args.device)[0] if args.opponent else ('negamax', args.nodes)
    games = []
    for i in range(2 * args.pairs):
        pos, history = evaluation_opening(engine, i, 'heldout', args.seed)
        players = [model, opponent] if i % 2 == 0 else [opponent, model]
        g = Game(100000 + i, pos, history, players, record=False)
        g.learner = i % 2
        games.append(g)
    start = time.monotonic()
    play(engine, model, games, args.device, args.simulations, 100)
    records = [dict(pair=i // 2, color=g.learner, score=0.5 if g.result == -1 else float(g.result == g.learner),
                    plies=g.ply, reason=g.reason) for i, g in enumerate(games)]
    scores = np.array([g['score'] for g in records]).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(42)
    bootstrap = scores[rng.integers(0, len(scores), (10000, len(scores)))].mean(1)
    result = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  score=float(scores.mean()), pair_bootstrap_95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                  seconds=time.monotonic() - start, search_stats=engine.stats, games=records)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'games'}), flush=True)


if __name__ == '__main__':
    main()
