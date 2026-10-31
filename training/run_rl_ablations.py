"""Sequential, resumable L4 ablation queue; each arm gets the same training-time budget.

Run from the source checkout. Never shares replay or optimizer state between arms.
Existing arm checkpoints resume through az.py; completed evaluations are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent


def arms(schedule_steps):
    schedule = ['--schedule-steps', str(schedule_steps)]
    return {
        'baseline': [],
        'adamw_cosine': ['--schedule', 'cosine', *schedule],
        'adamw_restarts': ['--schedule', 'cosine-restarts', *schedule],
        'muon_flat': ['--optimizer', 'muon'],
        'muon_cosine': ['--optimizer', 'muon', '--schedule', 'cosine', *schedule],
        'muon_restarts': ['--optimizer', 'muon', '--schedule', 'cosine-restarts', *schedule],
        'playout_40_400': ['--playout-cap', 'randomized-40-400'],
        'race_solver': ['--race-solver', 'exact'],
        'tablebase': ['--race-solver', 'tablebase'],
    }


def run(command, log):
    print(json.dumps({'started': time.time(), 'command': list(map(str, command))}), flush=True)
    with log.open('a') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def archive_results(out):
    """Small atomic downloads survive a VM release; full resumable states stay separate."""
    from summarize_rl_ablations import summarize
    summarize(out)
    for label, paths in [('results', sorted(out.rglob('*.json')) + sorted(out.glob('*.log')) + sorted(out.glob('*.md'))),
                         ('models', sorted(out.glob('*/model.pt')))]:
        target = out.parent / f'{out.name}-{label}.zip'
        temporary = target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(out))
        temporary.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--minutes', type=float, default=45)
    p.add_argument('--schedule-steps', type=int, default=15000)
    p.add_argument('--pairs', type=int, default=100)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--arms', nargs='+')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    chosen = arms(args.schedule_steps)
    if args.arms:
        chosen = {name: chosen[name] for name in args.arms}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(minutes=args.minutes, schedule_steps=args.schedule_steps, seed=args.seed,
                pairs=args.pairs, arms=chosen, init=str(args.init.resolve()),
                init_sha256=hashlib.sha256(args.init.read_bytes()).hexdigest(),
                source_sha256={str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in [HERE / 'az.py', HERE / 'native/az.cpp', HERE / 'native/engine.hpp',
                                         HERE / 'native/race_tablebase.hpp']})
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Existing queue plan differs: use a new output directory')
    plan_path.write_text(json.dumps(plan, indent=2))
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
        raise RuntimeError('This benchmark requires an NVIDIA L4')
    (out / 'environment.json').write_text(json.dumps(dict(
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        cpus=os.cpu_count(), python=sys.version), indent=2))
    data = out / 'gen.npz'
    if not data.exists():
        run([sys.executable, HERE / 'az.py', 'gen', '--out', data, '--games', '30000',
             '--parallel', '1000', '--seed', '1'], out / 'gen.log')
    if not (out / 'race-inference.json').exists():
        run([sys.executable, HERE / 'benchmark_race_solver.py', '--init', args.init.resolve(),
             '--data', data, '--out', out / 'race-inference.json'], out / 'race-inference.log')
    for name, flags in chosen.items():
        arm = out / name
        arm.mkdir(exist_ok=True)
        if not (arm / 'DONE').exists():
            run([sys.executable, HERE / 'az.py', 'rl', '--device', 'cuda', '--init', args.init.resolve(),
                 '--mix-data', data, '--out', arm, '--iterations', '1000',
                 '--budget-minutes', args.minutes, '--games', '2048', '--negamax-fraction', '0.5',
                 '--simulations', '200', '--eval-every', '0', '--batch-size', '1024', '--lr', '3e-4',
                 '--replay', '600000', '--reuse', '4', '--checkpoint-seconds', '600',
                 '--seed', args.seed, *flags], out / f'{name}.log')
            (arm / 'DONE').write_text(str(time.time()))
            archive_results(out)
        for sims, nodes in [(1, 2000), (128, 2000), (128, 32000)]:
            result = out / f'{name}-s{sims}-n{nodes}.json'
            if not result.exists():
                run([sys.executable, HERE / 'ablation_eval.py', '--model', arm / 'model.pt',
                     '--pairs', args.pairs, '--simulations', sims, '--nodes', nodes,
                     '--seed', '2027', '--out', result], out / 'eval.log')
                archive_results(out)
        baseline = out / 'baseline/model.pt'
        result = out / f'{name}-vs-baseline.json'
        if name != 'baseline' and baseline.exists() and not result.exists():
            run([sys.executable, HERE / 'ablation_eval.py', '--model', arm / 'model.pt',
                 '--opponent', baseline, '--pairs', args.pairs, '--simulations', '128',
                 '--seed', '2027', '--out', result], out / 'eval.log')
            archive_results(out)
    (out / 'DONE').write_text(str(time.time()))
    archive_results(out)


if __name__ == '__main__':
    main()
