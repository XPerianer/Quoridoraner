"""Short CUDA smoke/timing through the production compiled CUDA-graph inference path.

python training/benchmark_transformer.py --out benchmark.json
Compilation/capture are reported separately from steady-state latency. Synthetic inputs
measure network compute only, not MCTS strength or end-to-end self-play throughput.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import torch

from az import Engine, Net, TransformerNet


def benchmark(name, constructor, batches, repeats, train_batch, train_repeats):
    model = constructor().cuda().eval().to(memory_format=torch.channels_last)
    engine = Engine(1)
    result = dict(architecture=name, config=model.config,
                  parameters=sum(p.numel() for p in model.parameters()), inference=[])
    try:
        with torch.inference_mode():
            net = engine.inference_net(model)
            for batch in batches:
                x = torch.randn(batch, model.config['planes'], 9, 9, device='cuda').to(
                    dtype=torch.bfloat16, memory_format=torch.channels_last)
                torch.cuda.synchronize()
                start = time.perf_counter()
                actual = tuple(t.clone() for t in net(x))
                torch.cuda.synchronize()
                setup = time.perf_counter() - start
                # Same folded bf16 copy, bypassing compile/graph capture.
                eager = next(iter(engine.inference.values()))[0]
                expected = eager(x)
                errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)]
                for a, b in zip(actual, expected):
                    assert torch.isfinite(a).all()
                    torch.testing.assert_close(a, b, atol=0.025, rtol=0.025)
                # Revisit captured sizes later to exercise the shared graph pool.
                for _ in range(3):
                    net(x)
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                begin.record()
                for _ in range(repeats):
                    net(x)
                end.record()
                end.synchronize()
                wall_ms = 1000 * (time.perf_counter() - start) / repeats
                gpu_ms = begin.elapsed_time(end) / repeats
                result['inference'].append(dict(batch=batch, setup_seconds=setup, gpu_ms=gpu_ms,
                                                wall_ms=wall_ms, positions_per_second=1000 * batch / wall_ms,
                                                max_abs_error=errors))
                print(json.dumps(dict(architecture=name, **result['inference'][-1])), flush=True)
                del actual, expected, x
            for batch in reversed(batches):
                x = torch.randn(batch, model.config['planes'], 9, 9, device='cuda').to(
                    dtype=torch.bfloat16, memory_format=torch.channels_last)
                actual = tuple(t.clone() for t in net(x))
                expected = eager(x)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0.025, rtol=0.025)
            # Existing captured graphs must see in-place checkpoint weight refreshes.
            model.policy_head[-1].bias.add_(0.25)
            refreshed = engine.inference_net(model)
            assert refreshed is net
            actual = tuple(t.clone() for t in refreshed(x))
            expected = eager(x)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0.025, rtol=0.025)
            result['graph_replay_and_weight_refresh'] = 'passed'
            # CPU dispatch profiling avoids a CUPTI dependency. This identifies the
            # eager SDPA backend; compiled kernels may choose a different implementation.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                eager(x)
            result['eager_inference_attention_ops'] = [event.key for event in profile.key_averages()
                if any(word in event.key.lower() for word in ('attention', 'scaled_dot', 'flash', 'fmha'))]
            del x, actual, expected, net, refreshed, eager
        result['peak_inference_allocated_mb'] = torch.cuda.max_memory_allocated() / 2**20
        engine.inference.clear()
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randn(train_batch, model.config['planes'], 9, 9, device='cuda')
        target_policy = torch.randint(209, (train_batch,), device='cuda')
        target_value = torch.rand(train_batch, device='cuda') * 2 - 1

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, value = model(x)
            loss = torch.nn.functional.cross_entropy(logits.float(), target_policy) + (
                value.float() - target_value).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            return loss, norm

        # Warm allocator, optimizer moments, and training kernel selection.
        for _ in range(3):
            loss, norm = train_step()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        for _ in range(train_repeats):
            loss, norm = train_step()
        end.record()
        end.synchronize()
        wall_ms = 1000 * (time.perf_counter() - start) / train_repeats
        assert torch.isfinite(loss) and torch.isfinite(norm)
        for parameter in model.parameters():
            assert torch.isfinite(parameter).all() and torch.isfinite(parameter.grad).all()
        result['training_smoke'] = dict(batch=train_batch, loss=float(loss), gradient_norm=float(norm), status='passed')
        result['training'] = dict(batch=train_batch, warmup_steps=3, repeats=train_repeats,
                                  wall_ms=wall_ms, gpu_ms=begin.elapsed_time(end) / train_repeats,
                                  positions_per_second=1000 * train_batch / wall_ms,
                                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                                  includes='forward, loss, backward, gradient clipping, AdamW; excludes data transfer')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            train_step()
        result['training_attention_ops'] = [event.key for event in profile.key_averages()
            if any(word in event.key.lower() for word in ('attention', 'scaled_dot', 'flash', 'fmha'))]
        torch.cuda.synchronize()
        return result
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[64, 256, 1024, 2048])
    parser.add_argument('--repeats', type=int, default=20)
    candidates = {
        'transformer': TransformerNet,
        'transformer-h3': lambda: TransformerNet(heads=3),
        'transformer-256x4': lambda: TransformerNet(width=256, blocks=4, heads=4),
        'transformer-128x6': lambda: TransformerNet(width=128, blocks=6, heads=2),
        'resnet': lambda: Net(channels=128, blocks=10),
        'transformer-128x2': lambda: TransformerNet(width=128, blocks=2, heads=2),
        'resnet-64x4': lambda: Net(channels=64, blocks=4),
    }
    parser.add_argument('--models', choices=list(candidates), nargs='+', default=list(candidates)[:5])
    parser.add_argument('--train-batch', type=int, default=1024)
    parser.add_argument('--train-repeats', type=int, default=10)
    parser.add_argument('--sdpa-backend', choices=['auto', 'flash', 'cudnn', 'efficient', 'math'], default='auto',
                        help='Force an SDPA backend for a diagnostic; unsupported kernels fail explicitly')
    args = parser.parse_args()
    if min(args.repeats, args.train_repeats, args.train_batch, *args.batches) < 1:
        parser.error('Batch sizes and repeat counts must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')
    torch.manual_seed(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    results = dict(torch_version=torch.__version__, device=torch.cuda.get_device_name(), sdpa_backend=args.sdpa_backend, models=[])
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = dict(flash=SDPBackend.FLASH_ATTENTION, cudnn=SDPBackend.CUDNN_ATTENTION,
                    efficient=SDPBackend.EFFICIENT_ATTENTION, math=SDPBackend.MATH)
    for name in args.models:
        torch.manual_seed(1)
        torch.cuda.reset_peak_memory_stats()
        context = nullcontext() if args.sdpa_backend == 'auto' else sdpa_kernel(backends[args.sdpa_backend])
        with context:
            results['models'].append(benchmark(name, candidates[name], args.batches, args.repeats,
                                                args.train_batch, args.train_repeats))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
