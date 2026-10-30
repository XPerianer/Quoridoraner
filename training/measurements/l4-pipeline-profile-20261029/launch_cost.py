"""CPU cost per wave on the L4: native wave vs queueing the network (eager, compiled, CUDA graphs)."""
import json, random, sys, time
sys.path.insert(0, '/content/q/training')
import numpy as np
import torch
import az

device = 'cuda'
model, _ = az.load_model('/content/rl-model.pt', device)
e = az.Engine()
e.prepare(model, device)
compiled = e.net
eager = next(iter(e.inference.values()))[0]
rng = random.Random(1)
jobs = {}
for i in range(1024):
    pos, hist = az.random_opening(e, rng, rng.randint(0, 8))
    job = dict(tree=i, position=pos, history=hist, ply=0, max_plies=100)
    jobs[e.begin(job, 200, i % 2)] = (job, i % 2)
results = []


features, legal, counts = e.buffers[0]
k = e.check(e.lib.az_wave(e.pool, 0, az.ptr(features), az.ptr(legal), az.ptr(counts)))
batch = (features[:k].copy(), legal[:k].copy(), counts[:k].copy())
pinned = [torch.from_numpy(a).pin_memory().numpy() for a in batch]


def timed(net, label, calls=300):
    """CPU time to queue one batch, and sustained rows/s when two batches are kept in flight."""
    for _ in range(10):
        az.priors(net, *pinned, device)[2].synchronize()
    queue, previous = 0.0, None
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(calls):
        t = time.perf_counter()
        done = az.priors(net, *pinned, device)[2]
        queue += time.perf_counter() - t
        if previous is not None:
            previous.synchronize()
        previous = done
    previous.synchronize()
    wall = time.perf_counter() - start
    row = dict(variant=label, rows=k, enqueue_ms=queue / calls * 1e3, wall_ms_per_batch=wall / calls * 1e3,
               evals_per_second=k * calls / wall)
    results.append(row)
    print(json.dumps(row), flush=True)


class Graphs:
    """One CUDA graph per padded batch size (multiples of 64), sharing a memory pool."""

    def __init__(self, net, planes, sizes):
        self.graphs, self.pool = {}, torch.cuda.graph_pool_handle()
        for n in sizes:
            x = torch.zeros(n, planes, 9, 9, device=device, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    net(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool), torch.inference_mode():
                out = net(x)
            self.graphs[n] = (graph, x, out)

    def __call__(self, x):
        graph, static, out = self.graphs[len(x)]
        static.copy_(x)
        graph.replay()
        return out


timed(compiled, 'compile dynamic (current)')
timed(eager, 'eager fused bf16 channels_last')
t = time.perf_counter()
graphs = Graphs(eager, e.planes, range(64, 1025, 64))
print('eager graphs capture seconds', time.perf_counter() - t, flush=True)
timed(graphs, 'CUDA graphs of eager')
t = time.perf_counter()
cgraphs = Graphs(compiled, e.planes, range(64, 1025, 64))
print('compiled graphs capture seconds', time.perf_counter() - t, flush=True)
timed(cgraphs, 'CUDA graphs of compiled')

# Throughput of graph replays alone at 512 rows.
for label, net in (('compiled', compiled), ('graphs of compiled', cgraphs), ('graphs of eager', graphs)):
    x = torch.zeros(512, e.planes, 9, 9, device=device, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        for _ in range(5):
            net(x)
        torch.cuda.synchronize(); t = time.perf_counter(); n = 0
        while time.perf_counter() - t < 4:
            net(x); n += 512
        torch.cuda.synchronize()
    row = dict(variant=label, evals_per_second_512=n / (time.perf_counter() - t))
    results.append(row); print(json.dumps(row), flush=True)
json.dump(results, open('/content/launch-cost.json', 'w'), indent=1)
print('DONE', flush=True)
