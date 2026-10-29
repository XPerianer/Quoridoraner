"""Isolated model/batch sweep: real update path and inference host round trips.

Synthetic targets measure execution, not learning or playing strength.
"""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch

from core import ARCHITECTURES, PolicyValue, update


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def measure(args):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime in Colab first.')
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            return dict(status='unsupported', reason='GPU lacks BF16 support')
    model = PolicyValue(args.architecture, args.precision).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    # Same seeded input/targets for each architecture. Include value-only rows.
    generator = torch.Generator().manual_seed(42)
    x = torch.rand(args.batch, 9, 9, 9, generator=generator)
    policy = torch.rand(args.batch, 209, generator=generator).softmax(1)
    replay = [(x[i], policy[i] if i % 2 else None, float(i % 3 - 1)) for i in range(args.batch)]
    rng = random.Random(1)
    def inference():
        with torch.inference_mode():
            logits, values = model(x.to(args.device))
            return torch.cat((logits, values[:, None]), 1).cpu()
    def train():
        return update(model, optimizer, replay, args.batch, rng, args.device)
    result = dict(status='ok', parameters=sum(p.numel() for p in model.parameters()))
    for label, operation in [('inference', inference), ('update', train)]:
        model.train(label == 'update')
        for _ in range(args.warmup):
            operation()
        synchronize(args.device)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.repeats):
            synchronize(args.device)
            start = time.perf_counter()
            operation()
            synchronize(args.device)
            durations.append(time.perf_counter() - start)
        median = statistics.median(durations)
        result[label] = dict(median_ms=median * 1000, positions_per_second=args.batch / median,
                             samples_seconds=durations)
        if args.device == 'cuda':
            result[label].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('training/runs/cuda-sweep.json'))
    parser.add_argument('--architectures', nargs='+', choices=ARCHITECTURES,
                        default=['small', 'residual64', 'residual128', 'residual256', 'residual512'])
    parser.add_argument('--batches', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--precisions', nargs='+', choices=['fp32', 'bf16'], default=['fp32', 'bf16'])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--architecture', choices=ARCHITECTURES)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--precision', choices=['fp32', 'bf16'])
    args = parser.parse_args()
    if min(args.batches) < 1 or args.repeats < 1 or args.warmup < 1 or (args.worker and (not args.batch or args.batch < 1)):
        parser.error('Batches, repeats and warmup must be positive')
    if args.worker:
        try:
            result = measure(args)
        except torch.cuda.OutOfMemoryError:
            result = dict(status='oom')
        print(json.dumps(result))
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; select a GPU runtime, or use --device cpu for a smoke check')
    hardware = dict(torch=torch.__version__, cuda=torch.version.cuda, cpu_count=os.cpu_count(), device=args.device)
    if args.device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        hardware.update(gpu=props.name, memory_bytes=props.total_memory,
                        capability=torch.cuda.get_device_capability(), bf16=torch.cuda.is_bf16_supported())
    report = dict(hardware=hardware, settings=vars(args).copy(), rows=[])
    report['settings']['out'] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for architecture in args.architectures:
        for precision in args.precisions:
            if args.device != 'cuda' and precision != 'fp32':
                continue
            for batch in args.batches:
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--architecture', architecture,
                           '--precision', precision, '--batch', str(batch), '--device', args.device,
                           '--warmup', str(args.warmup), '--repeats', str(args.repeats)]
                completed = subprocess.run(command, text=True, capture_output=True)
                if completed.returncode:
                    result = dict(status='error', returncode=completed.returncode, stderr=completed.stderr[-4000:])
                else:
                    result = json.loads(completed.stdout)
                row = dict(architecture=architecture, precision=precision, batch=batch, **result)
                report['rows'].append(row)
                args.out.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)
    if any(row['status'] == 'error' for row in report['rows']):
        raise SystemExit('Some cases failed; inspect the report.')


if __name__ == '__main__':
    main()
