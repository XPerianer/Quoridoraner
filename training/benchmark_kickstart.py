"""L4 settings for kickstarted students: production inference and training-step timings per architecture.

python training/benchmark_kickstart.py --teacher model-h013.pt --out kickstart-bench.json [--cudnn-benchmark]
Inference goes through Engine.inference_net (folded bf16 CUDA graphs) at search-wave batch sizes, with
optional output microbatching. Training goes through train_steps on real features with the compiled
teacher, as in `az.py rl --teacher`. Network and training throughput only, not self-play games per hour.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from az import Engine, Net, Replay, TransformerNet, load_model, random_opening, train_steps

CANDIDATES = {
    'resnet-128x10': lambda planes: Net(planes, 128, 10),
    'resnet-192x15': lambda planes: Net(planes, 192, 15),
    'resnet-256x20': lambda planes: Net(planes, 256, 20),
    'transformer-256x4': lambda planes: TransformerNet(planes, width=256, blocks=4, heads=4, expansion=2),
    'transformer-384x8': lambda planes: TransformerNet(planes, width=384, blocks=8, heads=6, expansion=2),
    'transformer-512x6': lambda planes: TransformerNet(planes, width=512, blocks=6, heads=8, expansion=2),
    'transformer-512x8': lambda planes: TransformerNet(planes, width=512, blocks=8, heads=8, expansion=2),
    'transformer-512x12': lambda planes: TransformerNet(planes, width=512, blocks=12, heads=8, expansion=2),
}


def timed(call, repeats):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def inference(engine, name, model, batches, chunks, repeats):
    rows = []
    variable = 'AZ_TRANSFORMER_INFERENCE_BATCH' if isinstance(model, TransformerNet) else 'AZ_RESNET_INFERENCE_BATCH'
    with torch.inference_mode():
        for chunk in chunks:
            os.environ[variable] = str(chunk)
            net = engine.inference_net(model, key=f'{name}-{chunk}')
            for batch in batches:
                x = torch.randn(batch, model.config['planes'], 9, 9, device='cuda').to(
                    torch.bfloat16, memory_format=torch.channels_last)
                ms = timed(lambda: net(x), repeats)
                rows.append(dict(batch=batch, chunk=chunk, ms=ms, positions_per_second=1000 * batch / ms))
                print(json.dumps(dict(model=name, inference=rows[-1])), flush=True)
            del engine.inference[f'{name}-{chunk}']
    os.environ.pop(variable, None)
    return rows


def training(engine, name, model, replay, teacher, chunks, steps, batch):
    rows = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for chunk in chunks:
        for weight in (1.0, 0.0):
            if isinstance(model, TransformerNet):
                os.environ['AZ_TRANSFORMER_TRAIN_BATCH'] = str(chunk)
            rng = np.random.default_rng(0)
            train_steps(model, optimizer, replay, engine, 3, batch, 'cuda', rng, teacher=teacher, teacher_weight=weight)
            torch.cuda.synchronize()
            start = time.perf_counter()
            losses = train_steps(model, optimizer, replay, engine, steps, batch, 'cuda', rng, teacher=teacher,
                                 teacher_weight=weight)
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - start) / steps
            assert all(np.isfinite(v) for v in losses.values()), losses
            rows.append(dict(batch=batch, chunk=chunk, teacher_weight=weight, ms_per_step=ms, losses=losses))
            print(json.dumps(dict(model=name, training=rows[-1])), flush=True)
        if not isinstance(model, TransformerNet):
            break  # BatchNorm statistics depend on the batch: no training microbatches.
    os.environ.pop('AZ_TRANSFORMER_TRAIN_BATCH', None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', nargs='+', choices=list(CANDIDATES), default=list(CANDIDATES)[:4])
    parser.add_argument('--batches', type=int, nargs='+', default=[256, 512, 768, 1024])
    parser.add_argument('--chunks', type=int, nargs='+', default=[0, 64, 128, 256, 512])
    parser.add_argument('--train-chunks', type=int, nargs='+', default=[0, 128, 256])
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--cudnn-benchmark', action='store_true')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(1)
    engine = Engine()
    rng = random.Random(1)
    replay = Replay(4096)
    for _ in range(replay.capacity):
        position, history = random_opening(engine, rng, rng.randint(0, 40))
        replay.add(position, history[position], rng.random(), None, rng.choice((-1.0, 1.0)))
    frozen, _ = load_model(args.teacher, 'cuda')
    frozen.requires_grad_(False)
    compiled = engine.inference_net(frozen, key='teacher')
    teacher = lambda x: compiled(x.to(torch.bfloat16, memory_format=torch.channels_last))
    results = dict(torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                   cudnn_benchmark=args.cudnn_benchmark, teacher=args.teacher, models=[])
    try:
        for name in args.models:
            model = CANDIDATES[name](engine.planes).cuda().eval().to(memory_format=torch.channels_last)
            row = dict(model=name, config=model.config, parameters=sum(p.numel() for p in model.parameters()))
            row['inference'] = inference(engine, name, model, args.batches, args.chunks, args.repeats)
            row['training'] = training(engine, name, model, replay, teacher, args.train_chunks, args.steps,
                                       args.batch_size)
            results['models'].append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        engine.close()


if __name__ == '__main__':
    main()
