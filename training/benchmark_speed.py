"""Sequential fixed-cohort, rolling, and randomized-budget comparisons."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--games', type=int, default=256)
    parser.add_argument('--parallel-games', type=int, default=128)
    parser.add_argument('--walls', type=int, default=10)
    parser.add_argument('--curriculum', choices=['none', 'races', 'gated'], default='races')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--variants', nargs='+', choices=['cohort', 'rolling', 'randomized'],
                        default=['cohort', 'rolling', 'randomized'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for index, seed in enumerate(args.seeds):
        variants = list(args.variants)
        if index % 2:
            variants.reverse()
        for variant in variants:
            directory = args.out / f'{variant}-{seed}'
            command = [sys.executable, str(ROOT / 'training/train.py'), '--out', str(directory),
                       '--device', args.device, '--plain', '--engine', 'direct', '--workers', '4',
                       '--mps-inference', 'hybrid', '--iterations', str(args.iterations),
                       '--games', str(args.games), '--parallel-games', str(args.parallel_games),
                       '--seed', str(seed), '--simulations', '64', '--updates', '32', '--batch-size', str(args.batch_size),
                       '--eval-games', '20', '--eval-every', '5', '--heldout-games', '20',
                       '--curriculum', args.curriculum, '--walls', str(args.walls), '--checkpoint-seconds', '60',
                       '--game-scheduler', 'cohort' if variant == 'cohort' else 'rolling',
                       '--search-budget', 'randomized' if variant == 'randomized' else 'fixed',
                       '--fast-simulations', '16', '--full-search-probability', '.25']
            print(f'Running {variant}, seed {seed}', flush=True)
            start = time.perf_counter()
            with (args.out / f'{variant}-{seed}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True)
            wall = time.perf_counter() - start
            perf = json.loads((directory / 'performance.json').read_text())
            records = json.loads((directory / 'metrics.json').read_text())
            train = [r for r in records if r['phase'] == 'train']
            row = dict(variant=variant, seed=seed, command=command, wall_seconds=wall,
                       performance=perf, mean_inference_batch=perf['inference_positions']/perf['inference_batches'],
                       mean_training_budget=perf['training_requested_simulations']/perf['training_search_moves'],
                       training_games=len(train), training_plies=sum(r['plies'] for r in train),
                       training_goals=sum(r['termination'] == 'goal' for r in train),
                       evaluation=[r for r in json.loads((directory / 'summary.json').read_text())
                                   if r['phase'] == 'eval'])
            results.append(row)
            (args.out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f"  {wall:.2f}s wall; batch {row['mean_inference_batch']:.1f}; budget {row['mean_training_budget']:.1f}", flush=True)


if __name__ == '__main__':
    main()
