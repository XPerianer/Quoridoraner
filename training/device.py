"""Measure actual small-network host/device round trips before selecting a device."""
import statistics
import time

import torch

from core import model_from_config


@torch.inference_mode()
def benchmark_devices(batch_sizes=(1, 2, 8, 32), repeats=12, model_config=None):
    # Device selection must not perturb training initialization/random streams.
    cpu_rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rows = []
    try:
        torch.manual_seed(0)
        initial = model_from_config(model_config).state_dict()
        for device in available_devices():
            model = model_from_config(model_config).to(device).eval()
            model.load_state_dict(initial)
            for batch in sorted(set(batch_sizes)):
                features = torch.zeros(batch, 9, 9, 9)
                samples = []
                for repeat in range(4 + repeats):
                    start = time.perf_counter()
                    logits, values = model(features.to(device))
                    torch.cat((logits, values.unsqueeze(1)), dim=1).cpu()
                    if device == 'mps':
                        torch.mps.synchronize()
                    elif device == 'cuda':
                        torch.cuda.synchronize()
                    if repeat >= 4:
                        samples.append((time.perf_counter() - start) * 1000)
                median = statistics.median(samples)
                rows.append({'device': device, 'batch': batch, 'median_ms': median,
                             'positions_per_second': batch * 1000 / median})
    finally:
        torch.set_rng_state(cpu_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return rows


def available_devices():
    return ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) + (['cuda'] if torch.cuda.is_available() else [])


def select_device(requested, batch_size, saved=None, model_config=None):
    if requested != 'auto':
        if requested not in available_devices():
            raise ValueError(f'{requested.upper()} is unavailable; use cpu or auto')
        return requested, []
    if saved in available_devices():
        return saved, []
    rows = benchmark_devices((1, max(1, batch_size)), model_config=model_config)
    measured = [r for r in rows if r['batch'] == batch_size]
    selected = min(measured, key=lambda r: r['median_ms'])['device']
    print('Inference benchmark (includes host/device transfers):', flush=True)
    for row in rows:
        print(f"  {row['device']:3s} batch {row['batch']:2d}: {row['median_ms']:.3f} ms", flush=True)
    print(f'Auto-selected {selected}; override with --device cpu, --device mps, or --device cuda.', flush=True)
    return selected, rows
