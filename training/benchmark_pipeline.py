"""Sequential, reproducible end-to-end comparison of bridge/direct MPS pipelines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New benchmark directory')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--variants', nargs='+', choices=['bridge', 'eager', 'bucketed', 'hybrid'],
                        default=['bridge', 'eager', 'bucketed', 'hybrid'])
    args = parser.parse_args()
    if min(args.repeats, args.workers, args.games, args.iterations) < 1 or args.games % 2:
        parser.error('Positive budgets and an even game count are required')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    reference = None
    for repeat in range(args.repeats):
        for variant in args.variants:
            directory = args.out / f'{variant}-{repeat+1}'
            command = [sys.executable, str(ROOT / 'training/train.py'), '--out', str(directory),
                       '--device', args.device, '--plain', '--iterations', str(args.iterations),
                       '--games', str(args.games), '--parallel-games', str(args.games), '--batch-size', '256',
                       '--simulations', '16', '--updates', '8', '--eval-games', '2', '--max-plies', '100',
                       '--negamax-nodes', '2000', '--workers', str(args.workers), '--checkpoint-seconds', '15',
                       '--engine', 'typescript' if variant == 'bridge' else 'direct']
            if variant != 'bridge':
                command += ['--mps-inference', variant]
            log_path = args.out / f'{variant}-{repeat+1}.log'
            print(f'Benchmarking {variant}, repeat {repeat+1} ...', flush=True)
            start = time.perf_counter()
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log)
            wall_seconds = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(f'Benchmark failed; see {log_path}')
            records = json.loads((directory / 'metrics.json').read_text())
            losses = json.loads((directory / 'losses.json').read_text())
            weights = {mode: torch.load(directory / f'{mode}.pt', weights_only=True, map_location='cpu')['model']
                       for mode in ('selfplay', 'negamax')}
            if reference is None:
                reference = (records, losses, weights)
            row = {'variant': variant, 'repeat': repeat+1, 'wall_seconds': wall_seconds,
                   'games': len(records), 'plies': sum(r['plies'] for r in records),
                   'records_equal': records == reference[0], 'losses_equal': losses == reference[1],
                   'weights_equal': all(torch.equal(weights[mode][name], reference[2][mode][name])
                                        for mode in weights for name in weights[mode]),
                   'performance': json.loads((directory / 'performance.json').read_text()), 'command': command}
            rows.append(row)
            (args.out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(f"  loop {row['performance']['training_seconds']:.3f}s, wall {wall_seconds:.3f}s, "
                  f"games/losses/weights equal: {row['records_equal']}/{row['losses_equal']}/{row['weights_equal']}", flush=True)
    print(f'Results: {args.out / "summary.json"}')


if __name__ == '__main__':
    main()
