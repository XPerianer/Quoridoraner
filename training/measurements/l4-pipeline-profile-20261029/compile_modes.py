"""torch.compile modes for the fused bf16 inference net, with varying batch sizes; plus compiled training."""
import copy, json, sys, time
sys.path.insert(0, '/content/q/training')
import numpy as np
import torch
import torch.nn as nn
import az

OUT = '/content/compile-modes.json'
device = 'cuda'
base, _ = az.load_model('/content/rl-model.pt', device)
e = az.Engine()
results = []


def fuse_bn(model):
    model = copy.deepcopy(model).eval()
    for parent in model.modules():
        names = list(parent._modules)
        for a, b in zip(names, names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                parent._modules[a] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                parent._modules[b] = nn.Identity()
    return model


sizes = list(range(64, 577, 64))
xs = {bs: torch.rand(bs, e.planes, 9, 9, device=device).to(torch.bfloat16).contiguous(memory_format=torch.channels_last) for bs in sizes}


@torch.inference_mode()
def measure(name, fn):
    t = time.perf_counter()
    for bs in sizes:
        fn(xs[bs]); fn(xs[bs])
    torch.cuda.synchronize()
    warm = time.perf_counter() - t
    rates = {}
    for bs in (64, 256, 512):
        torch.cuda.synchronize(); t = time.perf_counter(); n = 0
        while time.perf_counter() - t < 4:
            fn(xs[bs]); n += bs
        torch.cuda.synchronize()
        rates[bs] = n / (time.perf_counter() - t)
    # Mixed sizes, as in self-play.
    order = np.random.default_rng(0).choice(sizes, 400)
    torch.cuda.synchronize(); t = time.perf_counter()
    for bs in order:
        fn(xs[bs])
    torch.cuda.synchronize()
    rates['mixed'] = int(order.sum()) / (time.perf_counter() - t)
    row = dict(kind='forward', variant=name, warmup_seconds=warm, **{str(k): v for k, v in rates.items()})
    results.append(row); json.dump(results, open(OUT, 'w'), indent=1)
    print(json.dumps(row), flush=True)


fused = fuse_bn(base).to(torch.bfloat16).to(memory_format=torch.channels_last)
measure('eager channels_last', fused)
measure('compile default dynamic', torch.compile(copy.deepcopy(fused), dynamic=True))
measure('compile max-autotune-no-cudagraphs dynamic', torch.compile(copy.deepcopy(fused), mode='max-autotune-no-cudagraphs', dynamic=True))
measure('compile default static', torch.compile(copy.deepcopy(fused), dynamic=False))

# In-place weight refresh must not recompile.
m = copy.deepcopy(fused)
c = torch.compile(m, dynamic=True)
with torch.inference_mode():
    c(xs[128])
    for p in m.parameters():
        p.add_(0.001)
    t = time.perf_counter(); c(xs[128]); torch.cuda.synchronize()
    print('refresh call seconds', time.perf_counter() - t, flush=True)

replay = az.Replay(200000)
replay.size = 200000
replay.position[:] = np.asarray(az.START, np.uint64)
replay.policy[:, 0] = 1
replay.has_policy[:] = True
rng = np.random.default_rng(0)
for name, compiled in (('channels_last', False), ('channels_last + compile', True)):
    m = copy.deepcopy(base).to(memory_format=torch.channels_last)
    opt = az.new_optimizer(m, 3e-4)
    run = torch.compile(m) if compiled else m
    t = time.perf_counter()
    az.train_steps(run, opt, replay, e, 10, 1024, device, rng)
    warm = time.perf_counter() - t
    torch.cuda.synchronize(); t = time.perf_counter()
    az.train_steps(run, opt, replay, e, 200, 1024, device, rng)
    torch.cuda.synchronize()
    row = dict(kind='train', variant=name, warmup_seconds=warm, seconds_per_1000=(time.perf_counter() - t) / 200 * 1000)
    results.append(row); json.dump(results, open(OUT, 'w'), indent=1)
    print(json.dumps(row), flush=True)
print('DONE', flush=True)
