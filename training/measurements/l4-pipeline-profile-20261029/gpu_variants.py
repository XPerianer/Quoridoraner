"""Forward and training-step variants of the 128x10 net on one GPU."""
import copy, json, sys, time
sys.path.insert(0, '/content/q/training')
import numpy as np
import torch
import torch.nn as nn
import az

OUT = '/content/gpu-variants.json'
device = 'cuda'
base, _ = az.load_model('/content/rl-model.pt', device)
e = az.Engine()
results = []


def dump():
    json.dump(results, open(OUT, 'w'), indent=1)


def rate(fn, rows, seconds=6):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t, n = time.perf_counter(), 0
    while time.perf_counter() - t < seconds:
        fn(); n += rows
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t)


def fuse_bn(model):
    """Fold eval-mode BatchNorm into the preceding conv."""
    model = copy.deepcopy(model).eval()
    for parent in model.modules():
        names = list(parent._modules)
        for a, b in zip(names, names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                parent._modules[a] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                parent._modules[b] = nn.Identity()
    return model


x_np = {bs: np.random.rand(bs, e.planes, 9, 9).astype(np.float32) for bs in (64, 256, 512, 1024, 2048)}
legal = {bs: np.tile(np.arange(136, dtype=np.int32), (bs, 1)) for bs in x_np}
counts = {bs: np.full(bs, 100, np.int32) for bs in x_np}


def variant(name, model, autocast=True, channels_last=False, dtype=torch.float32, pinned=False, sizes=(64, 256, 512, 1024, 2048)):
    for bs in sizes:
        host = torch.from_numpy(x_np[bs])
        if pinned:
            host = host.pin_memory()

        @torch.inference_mode()
        def model_only(xd=host.to(device, dtype)):
            if channels_last:
                xd = xd.contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                return model(xd)

        @torch.inference_mode()
        def full():
            xd = host.to(device, non_blocking=pinned).to(dtype)
            if channels_last:
                xd = xd.contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                logits, values = model(xd)
            logits = logits.float().gather(1, torch.from_numpy(legal[bs]).to(device).long())
            mask = torch.arange(136, device=device)[None, :] >= torch.from_numpy(counts[bs]).to(device)[:, None]
            p = logits.masked_fill(mask, -torch.inf).softmax(1)
            return p.cpu().numpy(), values.float().cpu().numpy()

        row = dict(kind='forward', variant=name, batch=bs, model_only=rate(model_only, bs), full=rate(full, bs))
        results.append(row); dump()
        print(json.dumps(row), flush=True)


variant('baseline bf16 autocast', base)
variant('baseline pinned input', base, pinned=True)
variant('channels_last', copy.deepcopy(base).to(memory_format=torch.channels_last), channels_last=True)
fused = fuse_bn(base)
variant('fused BN', fused)
variant('fused BN, bf16 weights', copy.deepcopy(fused).to(torch.bfloat16), autocast=False, dtype=torch.bfloat16)
variant('fused BN, fp16 weights', copy.deepcopy(fused).half(), autocast=False, dtype=torch.float16)
variant('fused BN, bf16 weights, channels_last', copy.deepcopy(fused).to(torch.bfloat16).to(memory_format=torch.channels_last),
        autocast=False, dtype=torch.bfloat16, channels_last=True)
compiled = torch.compile(copy.deepcopy(fused).to(torch.bfloat16), mode='max-autotune-no-cudagraphs', dynamic=False)
variant('fused BN, bf16 weights, compile', compiled, autocast=False, dtype=torch.bfloat16)
graphs = torch.compile(copy.deepcopy(fused).to(torch.bfloat16), mode='max-autotune', dynamic=False)
variant('fused BN, bf16 weights, compile + CUDA graphs', graphs, autocast=False, dtype=torch.bfloat16)


# Training step at batch 1024: where does the time go, and what helps?
replay = az.Replay(200000)
rng = np.random.default_rng(0)
replay.size = 200000
replay.position[:] = np.asarray(az.START, np.uint64)
replay.policy[:, 0] = 1
replay.has_policy[:] = True


def train_rate(name, model, optimizer, steps=200):
    az.train_steps(model, optimizer, replay, e, 10, 1024, device, rng)
    torch.cuda.synchronize(); t = time.perf_counter()
    az.train_steps(model, optimizer, replay, e, steps, 1024, device, rng)
    torch.cuda.synchronize()
    row = dict(kind='train', variant=name, seconds_per_1000=(time.perf_counter() - t) / steps * 1000)
    results.append(row); dump()
    print(json.dumps(row), flush=True)


t = time.perf_counter()
for _ in range(200):
    idx = rng.integers(0, replay.size, 1024)
    mirror = rng.integers(0, 2, 1024).astype(np.uint8)
    e.features(replay.position[idx], replay.repetitions[idx], replay.remaining[idx], mirror)
row = dict(kind='train', variant='features only (CPU)', seconds_per_1000=(time.perf_counter() - t) / 200 * 1000)
results.append(row); dump(); print(json.dumps(row), flush=True)

m = copy.deepcopy(base)
train_rate('baseline AdamW', m, az.new_optimizer(m, 3e-4))
m = copy.deepcopy(base)
train_rate('fused AdamW', m, torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4, fused=True))
m = copy.deepcopy(base).to(memory_format=torch.channels_last)
train_rate('channels_last + fused AdamW', m, torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4, fused=True))
print('DONE', flush=True)
