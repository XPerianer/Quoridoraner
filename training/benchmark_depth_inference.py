"""Measure CPU versus bucketed MPS latency for each architecture; JSON on stdout."""
import json
import statistics
import time

import torch
from core import ARCHITECTURES, PolicyValue
from inference import MpsInference


def main():
    torch.set_num_threads(1)
    rows = []
    for architecture in ARCHITECTURES:
        torch.manual_seed(1)
        gpu = PolicyValue(architecture).to('mps').eval()
        cpu = PolicyValue(architecture).eval()
        cpu.load_state_dict({k: v.cpu() for k, v in gpu.state_dict().items()})
        runner = MpsInference(gpu, 'bucketed')
        with torch.inference_mode():
            for batch in (1, 2, 4, 8, 16, 32, 64):
                features = torch.zeros(batch, 9, 9, 9)
                row = dict(architecture=architecture, batch=batch)
                for device in ('cpu', 'mps'):
                    samples = []
                    for repeat in range(25):
                        start = time.perf_counter()
                        if device == 'cpu':
                            logits, values = cpu(features)
                            torch.cat((logits, values[:, None]), 1)
                        else:
                            runner(features)
                        if repeat >= 5:
                            samples.append((time.perf_counter() - start) * 1000)
                    row[device + '_ms'] = statistics.median(samples)
                rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
