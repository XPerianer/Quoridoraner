"""Audit saved runs without loading or modifying their checkpoints."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def report(directory):
    config = json.loads((directory / 'config.json').read_text())
    records = json.loads((directory / 'metrics.json').read_text())
    print(f'\n{directory.name}: walls={config.get("walls", 10)}, '
          f'simulations={config["simulations"]}, updates={config["updates"]}')
    print('W/C/L = goal wins / artificial cutoffs / goal losses; selfplay wins are not opponent strength.')
    progress_path = directory / 'curriculum.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text()) or {}
        for mode, arm in progress.items():
            print(f"  {mode}: curriculum stage {arm['stage']}, passing streak {arm['streak']}")
    for mode in ('selfplay', 'negamax'):
        training = [r for r in records if r['mode'] == mode and r['phase'] == 'train']
        latest = max((r['iteration'] for r in training), default=0)
        groups = defaultdict(list)
        for row in training:
            if row['iteration'] > latest - 10:
                groups[row.get('task', 'standard')].append(row)
        print(f'  {mode}: training through iteration {latest}; last 10 iterations by task')
        for task, rows in sorted(groups.items()):
            # Older negamax runs store learner turns only.
            samples = sum(sum((r['opening']['turn'] + i) % 2 == r['learner']
                              for i in range(r['plies'])) if mode == 'negamax'
                          and config.get('negamax_value_turns', 'learner') == 'learner'
                          else r['plies'] for r in rows)
            wins = sum(r['score'] == 1 for r in rows)
            cutoffs = sum(r['termination'] != 'goal' for r in rows)
            losses = len(rows) - wins - cutoffs
            print(f'    {task:10s} {len(rows):5d} games, {wins}/{cutoffs}/{losses}, '
                  f'{samples:6d} generated samples')
        evaluations = defaultdict(list)
        for row in records:
            if row['mode'] == mode and row['phase'] == 'eval':
                evaluations[row['iteration'], row.get('suite', 'monitor')].append(row)
        iterations = sorted({i for i, _ in evaluations})
        shown = set(iterations[:1] + iterations[-3:])
        for (iteration, suite), rows in sorted(evaluations.items()):
            if iteration not in shown:
                continue
            wins = sum(r['score'] == 1 for r in rows)
            cutoffs = sum(r['termination'] != 'goal' for r in rows)
            print(f'    eval {iteration:4d} {suite:9s}: '
                  f'{wins}/{cutoffs}/{len(rows)-wins-cutoffs} ({len(rows)} games)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    for run in parser.parse_args().runs:
        report(run)
