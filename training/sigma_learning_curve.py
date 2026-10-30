"""Local, resumable 100-game/checkpoint evaluation against frozen Sigma."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'experiments/training'
OUT = RUNS / 'sigma-learning-curve'
MODELS = {'pretrained': 'pre-model.pt', 'rl10': 'rl-model-it10.pt', 'rl40': 'rl-model.pt'}


def progress():
    result = {}
    for name in MODELS:
        games = []
        for file in (OUT / name).glob('sigma-pair*.json'):
            try:
                game = json.loads(file.read_text())
            except json.JSONDecodeError:
                continue
            if 'outcome' in game:
                games.append(game)
        counts = {o: sum(g['outcome'] == o for g in games) for o in ['win', 'loss', 'cutoff']}
        counts['completed'] = len(games)
        counts['score'] = (counts['win'] + 0.5 * counts['cutoff']) / len(games) if games else None
        result[name] = counts
    (OUT / 'progress.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, model in MODELS.items():
        log = (OUT / f'{name}.log').open('a')
        command = [sys.executable, '-u', str(HERE / 'external_arena.py'), '--opponents', 'sigma',
                   '--model', str(RUNS / 'az/colab' / model), '--pairs', '50',
                   '--seed', '20261030', '--simulations', '400', '--sigma-sims', '400',
                   '--resume', '--out', str(OUT / name)]
        jobs.append((name, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    while any(proc.poll() is None for _, proc, _ in jobs):
        progress()
        time.sleep(30)
    results = progress()
    codes = {name: proc.returncode for name, proc, _ in jobs}
    (OUT / 'completion.json').write_text(json.dumps({'exit_codes': codes, 'results': results}, indent=2))
    for _, _, log in jobs:
        log.close()
    if any(codes.values()):
        raise SystemExit(f'One or more evaluations failed: {codes}')
