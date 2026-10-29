"""Compare MPS inference layouts/precision, including CPU/GPU transfers."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from core import PolicyValue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit('MPS is unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(1)
    results = []
    with torch.inference_mode():
        model = PolicyValue().to('mps').eval()
        for kind in ('eager', 'channels_last', 'traced', 'fp16', 'compiled', 'nonblocking', 'reused'):
            candidate = copy.deepcopy(model)
            if kind == 'channels_last':
                candidate = candidate.to(memory_format=torch.channels_last)
            if kind == 'fp16':
                candidate = candidate.half()
            if kind == 'traced':
                # A measured alternative, not the production inference path.
                candidate = torch.jit.freeze(torch.jit.trace(candidate, torch.zeros(128, 9, 9, 9, device='mps')))
            if kind == 'compiled':
                candidate = torch.compile(candidate, fullgraph=True, dynamic=False)
            for batch in (1, 8, 32, 64, 128, 256, 512):
                features = torch.rand(batch, 9, 9, 9)
                if kind == 'channels_last':
                    features = features.contiguous(memory_format=torch.channels_last)
                if kind == 'fp16':
                    features = features.half()
                device_buffer = torch.empty_like(features, device='mps')
                samples = []
                for repeat in range(38):
                    start = time.perf_counter()
                    if kind == 'reused':
                        device_buffer.copy_(features, non_blocking=True)
                        transferred = device_buffer
                    else:
                        transferred = features.to('mps', non_blocking=kind == 'nonblocking')
                    logits, values = candidate(transferred)
                    torch.cat((logits, values[:, None]), 1).cpu()
                    if repeat >= 8:
                        samples.append((time.perf_counter() - start) * 1000)
                row = {'kind': kind, 'batch': batch, 'median_ms': statistics.median(samples)}
                results.append(row)
                print(row, flush=True)
        for batch in (1, 32, 128):
            features = torch.rand(batch, 9, 9, 9)
            samples = []
            for _ in range(30):
                start = time.perf_counter()
                transferred = features.to('mps')
                torch.mps.synchronize()
                copied = time.perf_counter()
                logits, values = model(transferred)
                torch.mps.synchronize()
                forwarded = time.perf_counter()
                torch.cat((logits, values[:, None]), 1).cpu()
                finished = time.perf_counter()
                samples.append([copied-start, forwarded-copied, finished-forwarded])
            row = {'kind': 'synchronized_components', 'batch': batch}
            for index, name in enumerate(('copy_ms', 'model_ms', 'read_ms')):
                row[name] = statistics.median(sample[index] for sample in samples) * 1000
            results.append(row)
            print(row, flush=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
