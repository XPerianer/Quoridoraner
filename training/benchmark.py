"""Compare CPU/MPS/CUDA neural inference at the batching sizes used by the trainer."""
import argparse
import json
from pathlib import Path

import torch

from device import benchmark_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2, 8, 32, 64])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if min(args.batches) < 1 or args.repeats < 1:
        parser.error('Batch sizes and repeats must be positive')
    torch.set_num_threads(1)
    rows = benchmark_devices(args.batches, args.repeats)
    for row in rows:
        print(f"{row['device']:3s}  batch {row['batch']:3d}  {row['median_ms']:8.3f} ms  "
              f"{row['positions_per_second']:10.0f} positions/s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
