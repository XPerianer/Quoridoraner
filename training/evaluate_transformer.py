"""Paired-opening head-to-head with game records and pair-cluster bootstrap uncertainty.

Scores and descriptive Elo differences are from model A's perspective. Cutoffs
(repetitions or ply limit) receive half a point; they are not proven draws.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from az import Engine, Game, evaluation_opening, load_model, pick_device, play


class SearchBudgetEngine(Engine):
    """Measure each side's actual search time while allowing different visit budgets."""
    def search(self, model, jobs, simulations, device, moves=False):
        side, budget = self.model_budgets[id(model)]
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = super().search(model, jobs, budget, device, moves=moves)
        if device == 'cuda':
            torch.cuda.synchronize()
        timing = self.side_timings.setdefault(side, dict(seconds=0.0, moves=0, calls=0))
        timing['seconds'] += time.perf_counter() - start
        timing['moves'] += len(jobs)
        timing['calls'] += 1
        return result


def elo(score):
    # Strings preserve infinite endpoints in standards-compliant JSON.
    if score <= 0:
        return '-Infinity'
    if score >= 1:
        return 'Infinity'
    return 400 * math.log10(score / (1 - score))


def paired_summary(scores, seed=731, draws=10000):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or len(scores) % 2:
        raise ValueError('Scores must contain complete consecutive color pairs')
    if not np.isin(scores, [0, 0.5, 1]).all():
        raise ValueError('Game scores must be 0, 0.5 or 1')
    pairs = scores.reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    # Bounded memory even when evaluating many thousands of pairs.
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        means[start:start + count] = pairs[rng.integers(len(pairs), size=(count, len(pairs)))].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975]).tolist()
    score = float(scores.mean())
    method = 'percentile bootstrap, opening-pair clusters'
    # Resampling an all-win/all-loss sample gives a falsely exact [1,1]/[0,0].
    # Bound the probability of an unseen non-extreme pair instead; its score
    # contribution is at most one, so this is conservative for the mean score.
    if np.all(pairs == 0):
        interval = [0.0, 1 - 0.025 ** (1 / len(pairs))]
        method = 'conservative exact extreme-pair bound (all losses)'
    elif np.all(pairs == 1):
        interval = [0.025 ** (1 / len(pairs)), 1.0]
        method = 'conservative exact extreme-pair bound (all wins)'
    elif np.all(pairs == pairs[0]):
        radius = math.sqrt(math.log(40) / (2 * len(pairs)))
        interval = [max(0.0, score - radius), min(1.0, score + radius)]
        method = 'Hoeffding bound for constant observed pair scores'
    return dict(score=score, score_ci95=interval, descriptive_elo=elo(score),
                descriptive_elo_ci95=[elo(s) for s in interval],
                uncertainty=dict(method=method, draws=draws,
                                 seed=seed, clusters=len(pairs),
                                 caveat='Opening-sample uncertainty only; excludes training-seed uncertainty. '
                                        'Degenerate samples can produce overconfident intervals.'))


def artifact(path, saved):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), config=saved['config'],
                iteration=saved.get('iteration'), spent_seconds=saved.get('spent_seconds'),
                totals=saved.get('totals'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs=2, metavar=('A', 'B'))
    parser.add_argument('--pairs', type=int, default=200)
    parser.add_argument('--simulations', type=int, default=200)
    parser.add_argument('--simulations-a', type=int,
                        help='Override model A search visits; enables per-side timing diagnostics')
    parser.add_argument('--simulations-b', type=int,
                        help='Override model B search visits; enables per-side timing diagnostics')
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-plies', type=int, default=100)
    parser.add_argument('--cpuct', type=float, default=1.5)
    parser.add_argument('--fpu', type=float, default=0.25)
    parser.add_argument('--race-solver', choices=('legacy', 'exact', 'tablebase'), default='legacy')
    args = parser.parse_args()
    budget_a = args.simulations if args.simulations_a is None else args.simulations_a
    budget_b = args.simulations if args.simulations_b is None else args.simulations_b
    if args.pairs < 1 or min(args.simulations, budget_a, budget_b) < 1:
        parser.error('pairs and simulations must be positive')
    device = pick_device(args.device)
    torch.set_num_threads(args.workers)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    a, saved_a = load_model(args.models[0], device)
    b, saved_b = load_model(args.models[1], device)
    metadata = [artifact(path, saved) for path, saved in zip(args.models, (saved_a, saved_b))]
    del saved_a, saved_b
    asymmetric = args.simulations_a is not None or args.simulations_b is not None
    engine_type = SearchBudgetEngine if asymmetric else Engine
    engine = engine_type(args.workers, args.cpuct, args.fpu, race_solver=args.race_solver)
    if asymmetric:
        engine.model_budgets = {id(a): ('a', budget_a), id(b): ('b', budget_b)}
        engine.side_timings = {}
    try:
        games = []
        for i in range(2 * args.pairs):
            position, history = evaluation_opening(engine, i, 'heldout', args.seed)
            game = Game(100000 + i, position, history, [a, b] if i % 2 == 0 else [b, a], record=False)
            game.learner = i % 2
            games.append(game)
        start = time.perf_counter()
        last_report = [start]
        def progress(done, total):
            now = time.perf_counter()
            if now - last_report[0] >= 30 or done == total:
                print(json.dumps(dict(completed=done, total=total, elapsed_seconds=now - start)), flush=True)
                last_report[0] = now
        play(engine, None, games, device, args.simulations, args.max_plies, progress=progress)
        elapsed = time.perf_counter() - start
        records = []
        for i, game in enumerate(games):
            score = 0.5 if game.result == -1 else float(game.result == game.learner)
            records.append(dict(index=i, pair=i // 2, a_color=game.learner, score=score,
                                winner_color=game.result, reason=game.reason, plies=game.ply,
                                opening=list(game.opening), moves=game.moves))
        summary = paired_summary([r['score'] for r in records], args.seed)
        summary.update(games=len(games), pairs=args.pairs, wins=sum(r['score'] == 1 for r in records),
                       losses=sum(r['score'] == 0 for r in records), cutoffs=sum(r['score'] == 0.5 for r in records),
                       cutoff_reasons=dict(Counter(r['reason'] for r in records if r['score'] == 0.5)),
                       wins_by_color=[sum(r['score'] == 1 and r['a_color'] == c for r in records) for c in (0, 1)],
                       mean_plies=float(np.mean([r['plies'] for r in records])), elapsed_seconds=elapsed)
        result = dict(settings=vars(args), models=metadata, suite='heldout', search=engine.search_settings,
                      comparison='equal simulations; elapsed includes compilation/capture, not a speed comparison',
                      summary=summary, games=records)
        if asymmetric:
            result['comparison'] = 'different visit budgets; measured batched search time, not hard equal-time limits'
            result['search_timings'] = engine.side_timings
            result['timing_caveat'] = ('Includes network preparation, compilation and graph capture inside search; '
                                      'excludes game advancement outside search. Batching and CPU search affect throughput. '
                                      'Do not infer equal thinking time from a visit ratio or network-only benchmark.')
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(out)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
