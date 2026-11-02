"""Sequential paired games with equal target PUCT search time per move.

Two resident engines retain independent trees/caches. Network setup is warmed and
excluded; feature generation, tree work, inference and backup are inside the clock.
Stop after a completed wave, so an inference/native wave can overrun the target.
Actual time and evaluation counts are recorded. No batched-games speed assumption.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import az
from az import Engine, START, evaluation_opening, load_model, turn, winner
from evaluate_transformer import artifact, paired_summary


class TimedEngine(Engine):
    def __init__(self, workers=1):
        library = az.NATIVE / 'build' / ('az-timed.dylib' if sys.platform == 'darwin' else 'az-timed.so')
        sources = [az.NATIVE / name for name in ('az_timed.cpp', 'az.cpp', 'engine.hpp', 'race_tablebase.hpp')]
        if not library.exists() or any(p.stat().st_mtime > library.stat().st_mtime for p in sources):
            library.parent.mkdir(exist_ok=True)
            temporary = library.with_name(library.name + f'.{os.getpid()}.tmp')
            compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
            subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', '-pthread', '-O3', '-DNDEBUG',
                            str(sources[0]), '-o', str(temporary)], check=True)
            os.replace(temporary, library)
        original = az.LIBRARY
        try:
            az.LIBRARY = library
            super().__init__(workers, race_solver='legacy')
        finally:
            az.LIBRARY = original
        self.lib.az_stop_puct.restype = C.c_int
        self.lib.az_stop_puct.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int32)]

    def timed_search(self, model, job, milliseconds, device):
        if milliseconds <= 0:
            raise ValueError('Search time must be positive')
        self.prepare(model, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        before = self.stats['unique_evaluations']
        start = time.perf_counter()
        self.history(job['tree'], job['history'])
        slot = self.begin(job, 20000)
        completed = C.c_int32()
        while True:
            pending = self.step(device)
            if time.perf_counter()-start >= milliseconds/1000 or not pending:
                if self.check(self.lib.az_stop_puct(self.pool, slot, C.byref(completed))):
                    break
        policies, _, _ = self.take([slot])
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = 1000*(time.perf_counter()-start)
        if not np.isfinite(policies).all() or not np.isclose(policies.sum(), 1, atol=1e-5):
            raise RuntimeError('Timed search did not produce a normalized policy')
        return int(policies[0].argmax()), dict(milliseconds=elapsed, root_completed=completed.value,
                                             new_evaluations=self.stats['unique_evaluations']-before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs=2)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--milliseconds', type=float, default=20)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--max-plies', type=int, default=100)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if min(args.pairs, args.milliseconds, args.max_plies) <= 0:
        parser.error('Budgets must be positive')
    torch.set_num_threads(1)
    models, metadata = [], []
    for path in args.models:
        model, saved = load_model(path, args.device)
        models.append(model)
        metadata.append(artifact(path, saved))
        del saved
    engines = [TimedEngine(), TimedEngine()]
    games, timings = [], [[], []]
    start = time.perf_counter()
    try:
        for engine, model in zip(engines, models):
            # All actual searches use batch one, so this warms their CUDA graphs.
            engine.search(model, [dict(tree=0, position=START, history={START: 1}, ply=0,
                                      max_plies=args.max_plies)], 8, args.device)
            engine.drop(0)
        warmup = time.perf_counter()-start
        for i in range(2*args.pairs):
            position, history = evaluation_opening(engines[0], i, 'heldout', args.seed)
            opening, moves, record_times = position, [], []
            a_color = i % 2
            for engine in engines:
                engine.drop(0)
                engine.clear_cache()
                engine.history(0, history)
            result, reason = -1, 'ply_limit'
            for ply in range(args.max_plies):
                side = 0 if turn(position) == a_color else 1
                move, timing = engines[side].timed_search(models[side],
                    dict(tree=0, position=position, history=history, ply=ply, max_plies=args.max_plies),
                    args.milliseconds, args.device)
                timings[side].append(timing)
                record_times.append(dict(side=side, **timing))
                next_positions = [engine.advance(position, move, 0) for engine in engines]
                assert next_positions[0] == next_positions[1]
                position = next_positions[0]
                history[position] = history.get(position, 0)+1
                moves.append(move)
                result = winner(position)
                if result != -1:
                    reason = 'goal'
                    break
                if history[position] >= 3:
                    reason = 'repetition'
                    break
            score = .5 if result == -1 else float(result == a_color)
            games.append(dict(index=i, pair=i//2, a_color=a_color, score=score, winner_color=result,
                              reason=reason, opening=list(opening), moves=moves, timings=record_times))
            if (i+1) % 10 == 0:
                print(json.dumps(dict(completed=i+1, total=2*args.pairs, seconds=time.perf_counter()-start)), flush=True)
        summary = paired_summary([g['score'] for g in games], args.seed)
        summary.update(games=len(games), wins=sum(g['score']==1 for g in games),
                       losses=sum(g['score']==0 for g in games), cutoffs=sum(g['score']==.5 for g in games))
        timing_summary = []
        for rows in timings:
            ms = np.array([r['milliseconds'] for r in rows])
            timing_summary.append(dict(moves=len(rows), mean_ms=float(ms.mean()), median_ms=float(np.median(ms)),
                p95_ms=float(np.quantile(ms,.95)), max_ms=float(ms.max()),
                mean_new_evaluations=float(np.mean([r['new_evaluations'] for r in rows])),
                cap='20000 retained-plus-new root simulations, in addition to the clock target'))
        result = dict(settings=vars(args), models=metadata, summary=summary, timing=timing_summary,
                      warmup_seconds=warmup, total_seconds=time.perf_counter()-start, games=games,
                      comparison='equal target per-move wall-clock search, batch one; measured wave-boundary overruns',
                      excludes='model loading, warmed network preparation, game advancement; includes history/begin/search/take')
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(out)
        print(json.dumps(dict(summary=summary, timing=timing_summary)), flush=True)
    finally:
        for engine in engines:
            engine.close()


if __name__ == '__main__':
    main()
