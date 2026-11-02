"""Benchmark graph-inference microbatching without altering model weights or architecture.

Includes chunk output clones and concatenation because Graphs reuses output buffers.
Optionally load the trained transformer with --model; otherwise use seeded default weights.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from az import Engine, TransformerNet, load_model


def chunked(net, x, chunk):
    outputs = []
    for start in range(0, len(x), chunk):
        # Clone BEFORE the next replay can overwrite this chunk's output buffers.
        outputs.append(tuple(t.clone() for t in net(x[start:start + chunk])))
    return tuple(torch.cat([output[i] for output in outputs]) for i in range(2))


def accumulated_backward(model, x, policy, value, chunk, amp=True):
    """A full-batch mean gradient, accumulating each sub-batch's weighted mean."""
    chunk = chunk or len(x)
    model.zero_grad(set_to_none=True)
    total = x.new_zeros((), dtype=torch.float32)
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            logits, prediction = model(x[start:stop])
        loss = (torch.nn.functional.cross_entropy(logits.float(), policy[start:stop]) +
                (prediction.float() - value[start:stop]).square().mean()) * ((stop - start) / len(x))
        loss.backward()
        total = total + loss.detach()
    return total


def benchmark_training(model, batch, chunks, repeats):
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError('Training microbatch equivalence requires no batch-dependent normalization')
    model.train().to(memory_format=torch.channels_last)
    x = torch.randn(batch, model.config['planes'], 9, 9, device='cuda')
    policy = torch.randint(209, (batch,), device='cuda')
    value = torch.rand(batch, device='cuda') * 2 - 1
    original = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    full_loss = float(accumulated_backward(model, x, policy, value, 0))
    reference = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
    reference_norm = sum(float(g.double().square().sum()) for g in reference.values()) ** 0.5
    rows = []
    for chunk in chunks:
        model.load_state_dict(original)
        loss = accumulated_backward(model, x, policy, value, chunk)
        errors, largest = 0.0, 0.0
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            difference = parameter.grad.detach().cpu().double() - reference[name].double()
            errors += float(difference.square().sum())
            largest = max(largest, float(difference.abs().max()))
        relative_error = errors ** 0.5 / max(reference_norm, 1e-12)
        # BF16 kernel reductions vary with batch shape; compare whole-gradient L2,
        # avoiding unstable per-element relative errors at gradients near zero.
        assert relative_error < 0.05, f'Gradient relative L2 error {relative_error:.4g} exceeds 5%'
        assert abs(float(loss) - full_loss) < 0.025
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        def step():
            loss = accumulated_backward(model, x, policy, value, chunk)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            return loss, norm

        for _ in range(3):
            loss, norm = step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        for _ in range(repeats):
            loss, norm = step()
        end.record()
        end.synchronize()
        wall_ms = 1000 * (time.perf_counter() - start) / repeats
        assert torch.isfinite(loss) and torch.isfinite(norm)
        row = dict(batch=batch, chunk=chunk, repeats=repeats, warmup_steps=3, wall_ms=wall_ms,
                   gpu_ms=begin.elapsed_time(end) / repeats, positions_per_second=1000 * batch / wall_ms,
                   gradient_relative_l2_error=relative_error, gradient_max_abs_error=largest,
                   gradient_tolerance=0.05, full_batch_loss=full_loss,
                   peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                   description='Eager bf16; weighted gradient accumulation; clipping and AdamW once per full batch')
        rows.append(row)
        print(json.dumps(dict(training=row)), flush=True)
        del optimizer
    model.load_state_dict(original)
    model.eval()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model')
    parser.add_argument('--architecture', choices=['default', 'h3', '256x4', '128x6'], default='default')
    parser.add_argument('--train-chunks', nargs='+', type=int,
                        help='Enable training timing, e.g. 0 128 256; 0 means full batch')
    parser.add_argument('--train-batch', type=int, default=1024)
    parser.add_argument('--train-repeats', type=int, default=10)
    parser.add_argument('--batches', type=int, nargs='+', default=[1024, 2048])
    parser.add_argument('--chunks', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if min(args.repeats, *args.batches, *args.chunks) <= 0:
        parser.error('Batch/chunk sizes and repeats must be positive')
    if args.train_batch < 1 or args.train_repeats < 1 or any(c < 0 for c in (args.train_chunks or [])):
        parser.error('Training batches/repeats must be positive and chunks nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(1)
    torch.set_num_threads(4)
    configurations = {'default': {}, 'h3': dict(heads=3), '256x4': dict(width=256, blocks=4, heads=4),
                      '128x6': dict(width=128, blocks=6, heads=2)}
    model = load_model(args.model, 'cuda')[0] if args.model else TransformerNet(
        **configurations[args.architecture]).cuda().eval()
    engine = Engine(1)
    results = dict(torch_version=torch.__version__, device=torch.cuda.get_device_name(), config=model.config,
                   model=args.model, repeats=args.repeats,
                   description='Production CUDA graphs; chunked timings include output clones and concatenation; '
                               'setup and host/device input transfer excluded.', measurements=[])
    try:
        with torch.inference_mode():
            net = engine.inference_net(model)
            for batch in args.batches:
                x = torch.randn(batch, model.config['planes'], 9, 9, device='cuda').to(
                    dtype=torch.bfloat16, memory_format=torch.channels_last)
                reference = tuple(t.clone() for t in net(x))
                for chunk in [None] + args.chunks:
                    call = (lambda: net(x)) if chunk is None else (lambda: chunked(net, x, chunk))
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = call()
                    torch.cuda.synchronize()
                    setup = time.perf_counter() - start
                    errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(actual, reference)]
                    for a, b in zip(actual, reference):
                        torch.testing.assert_close(a, b, atol=0.025, rtol=0.025)
                        assert torch.isfinite(a).all()
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter()
                    begin.record()
                    for _ in range(args.repeats):
                        actual = call()
                    end.record()
                    end.synchronize()
                    wall_ms = 1000 * (time.perf_counter() - start) / args.repeats
                    row = dict(batch=batch, chunk=chunk, setup_seconds=setup, wall_ms=wall_ms,
                               gpu_ms=begin.elapsed_time(end) / args.repeats,
                               positions_per_second=1000 * batch / wall_ms, max_abs_error=errors)
                    results['measurements'].append(row)
                    print(json.dumps(row), flush=True)
                    out = Path(args.out)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(results, indent=2) + '\n')
        if args.train_chunks:
            del actual, reference, net, x
            engine.inference.clear()
            gc.collect()
            torch.cuda.empty_cache()
            results['training'] = benchmark_training(model, args.train_batch, args.train_chunks, args.train_repeats)
            Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
    finally:
        engine.close()


if __name__ == '__main__':
    main()
