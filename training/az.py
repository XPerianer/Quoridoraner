"""AlphaZero-style Quoridor with perspective-canonical path features.

Subcommands (see AZ_REPORT.md for the experiments and decisions):
  gen       negamax-vs-negamax games -> supervised dataset (.npz)
  pretrain  fit the policy/value network to a dataset
  rl        MCTS reinforcement learning against negamax and self-play
  eval      paired-opening match of a checkpoint against fixed negamax
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes as C
import json
import fcntl
import gzip
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NATIVE = HERE / 'native'
LIBRARY = NATIVE / 'build' / ('az.dylib' if sys.platform == 'darwin' else 'az.so')
START = (0, 0, 76 | (4 << 7) | (10 << 14) | (10 << 18))
MOVES = 209


def playout_budget(rng, mode, fallback):
    """Game-stream budget draw; fixed mode consumes no randomness for exact legacy parity."""
    if mode == 'fixed':
        return fallback
    if mode == 'randomized-40-400':
        return 400 if rng.random() < 0.25 else 40
    raise ValueError(f'Unknown playout cap {mode!r}')


# ---------------------------------------------------------------- engine

def build():
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    LIBRARY.parent.mkdir(exist_ok=True)
    temporary = LIBRARY.with_name(LIBRARY.name + f'.{os.getpid()}.tmp')
    subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', '-pthread', '-O3', '-DNDEBUG',
                    str(NATIVE / 'az.cpp'), '-o', str(temporary)], check=True)
    os.replace(temporary, LIBRARY)


def ptr(array):
    """Pointer to a contiguous array; it keeps the array alive, so temporaries are safe."""
    assert array.flags.c_contiguous
    return array.ctypes.data_as(C.c_void_p)


def winner(position):
    p = position[2]
    return 0 if (p & 127) < 9 else 1 if ((p >> 7) & 127) >= 72 else -1


def turn(position):
    return (position[2] >> 22) & 1


def walls_left(position):
    return (position[2] >> 14) & 15, (position[2] >> 18) & 15


def with_walls(position, walls):
    p = position[2] & ~((15 << 14) | (15 << 18))
    return (position[0], position[1], p | (walls << 14) | (walls << 18))


# Search settings saved with runs; checkpoints without them resolve to PUCT.
SEARCH_DEFAULTS = dict(search='puct', gumbel_interior='puct', gumbel_candidates=16, gumbel_scale=1.0,
                       gumbel_value_scale=0.1, gumbel_maxvisit_init=50.0, gumbel_rescale_values=True,
                       race_solver='legacy', race_depth=100, race_nodes=2048)


class Engine:
    def __init__(self, workers=None, cpuct=1.5, fpu=0.25, **search):
        sources = [NATIVE / 'az.cpp', NATIVE / 'engine.hpp', NATIVE / 'race_tablebase.hpp']
        if not LIBRARY.exists() or any(s.stat().st_mtime > LIBRARY.stat().st_mtime for s in sources):
            build()
        lib = self.lib = C.CDLL(str(LIBRARY))
        P, I = C.c_void_p, C.c_int
        for name, (result, args) in {
            'az_snapshot': (I, [P, P, I]), 'az_restore': (I, [P, P, I]),
            'az_create': (P, [I]), 'az_destroy': (None, [P]), 'az_error': (C.c_char_p, []), 'az_planes': (I, []),
            'az_settings': (I, [P, C.c_double, C.c_double, I, I, I, C.c_double, C.c_double, C.c_double,
                                I, I, I, I]),
            'az_clear': (I, [P]), 'az_drop': (I, [P, I]),
            'az_legal': (I, [P, P]), 'az_distances': (I, [P, P]), 'az_advance': (I, [P, I, P, P, P, P]),
            'az_history': (I, [P, I, P, P, I]),
            'az_features': (I, [P, P, P, P, P, I, P]), 'az_canonical_moves': (I, [I, I, P]),
            'az_start': (I, [P]), 'az_begin': (I, [P, I, P, I, I, I, C.c_uint64, I, I]),
            'az_wave': (I, [P, I, P, P, P]), 'az_evaluate': (I, [P, I, P, P, I]),
            'az_finished': (I, [P, P]), 'az_take': (I, [P, P, I, P, P, P]), 'az_cache': (I, [P, C.c_int64]),
            'az_cache_clear': (I, [P]), 'az_stats': (I, [P, P]), 'az_negamax': (I, [P, P, I, P, P]),
            'az_considered_visits': (I, [I, I, P]), 'az_root_visits': (I, [P, I, P]),
            'az_race_solve': (I, [P, I, I, I, I, P, P, I, P]),
        }.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = result, args
        self.workers = workers or min(256, os.cpu_count() or 1)
        self.pool = lib.az_create(self.workers)
        # Negamax opponents get their own pool and thread, so their replies overlap the searches.
        self.opponents = lib.az_create(self.workers)
        if not self.pool or not self.opponents:
            raise RuntimeError(lib.az_error().decode())
        self.background = ThreadPoolExecutor(1)
        # Native evaluation cache shared by all games; 0 disables lookups and in-wave deduplication.
        self.cache_limit = 1 << 20
        self.cache_version = None
        self.planes = lib.az_planes()
        self.configure(cpuct, fpu, **search)
        # canonical[turn][mirror][real move] -> canonical move; each map is an involution.
        self.canonical = np.zeros((2, 2, MOVES), np.int64)
        for t in range(2):
            for m in range(2):
                row = np.empty(MOVES, np.int32)
                self.check(lib.az_canonical_moves(t, m, ptr(row)))
                self.canonical[t, m] = row
        self.pinned = False
        self.allocate(0)
        self.inflight = {}  # group -> (rows, queued evaluation)
        self.net, self.inference = None, {}
        self.stats = dict(cache_hits=0, unique_evaluations=0, waves=0, leaves=0, forward_seconds=0.0, search_seconds=0.0)

    def configure(self, cpuct=1.5, fpu=0.25, **search):
        """PUCT (c_puct, FPU) or Gumbel search settings; keys and defaults as in SEARCH_DEFAULTS."""
        unknown = set(search) - set(SEARCH_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown search settings {sorted(unknown)}')
        s = self.search_settings = SEARCH_DEFAULTS | search
        assert s['search'] in ('puct', 'gumbel') and s['gumbel_interior'] in ('puct', 'completed-q')
        self.check(self.lib.az_settings(self.pool, cpuct, fpu, s['search'] == 'gumbel',
                                        s['gumbel_interior'] == 'completed-q', s['gumbel_candidates'],
                                        s['gumbel_scale'], s['gumbel_value_scale'], s['gumbel_maxvisit_init'],
                                        bool(s['gumbel_rescale_values']), {'legacy': 0, 'exact': 1, 'tablebase': 2}[s['race_solver']],
                                        s['race_depth'], s['race_nodes']))
        self.cpuct, self.fpu = cpuct, fpu

    def close(self):
        if getattr(self, 'background', None):
            self.background.shutdown()
            self.background = None
        for name in ('pool', 'opponents'):
            if getattr(self, name, None):
                self.lib.az_destroy(getattr(self, name))
                setattr(self, name, None)

    def __del__(self):
        self.close()

    def check(self, result):
        if result == -1:
            raise RuntimeError(self.lib.az_error().decode())
        return result

    def snapshot(self):
        n = self.check(self.lib.az_snapshot(self.pool, None, 0))
        data = np.empty(n, np.uint8)
        self.check(self.lib.az_snapshot(self.pool, ptr(data), n))
        return data.tobytes()

    def restore(self, data):
        packed = np.frombuffer(data, np.uint8)
        self.check(self.lib.az_restore(self.pool, ptr(packed), len(packed)))

    def legal(self, position):
        out = np.empty(136, np.int32)
        n = self.check(self.lib.az_legal(ptr(np.asarray(position, np.uint64)), ptr(out)))
        return out[:n].tolist()

    def distances(self, position):
        out = np.empty(2, np.int32)
        self.check(self.lib.az_distances(ptr(np.asarray(position, np.uint64)), ptr(out)))
        return int(out[0]), int(out[1])

    def solve_race(self, position, history, ply, max_plies, depth=100, node_cap=2048):
        """Bounded proof result: 1 win, 0 draw, -1 loss, 2 unknown; respects history and cutoff."""
        keys = np.asarray(list(history), np.uint64).reshape(-1, 3)
        counts = np.asarray(list(history.values()), np.int32)
        out = np.empty(3, np.int64)
        self.check(self.lib.az_race_solve(ptr(np.asarray(position, np.uint64)), ply, max_plies, depth, node_cap,
                                          ptr(keys), ptr(counts), len(counts), ptr(out)))
        return tuple(map(int, out))

    def advance(self, position, move, tree=-1):
        return tuple(self.advance_many([tree], [position], [move])[0])

    def advance_many(self, trees, positions, moves):
        """Play one move per position (trees >= 0 also move their root and history); returns new positions."""
        out = np.empty((len(moves), 3), np.uint64)
        self.check(self.lib.az_advance(self.pool, len(moves), ptr(np.asarray(trees, np.int32)),
                                       ptr(np.asarray(positions, np.uint64)), ptr(np.asarray(moves, np.int32)), ptr(out)))
        return out.tolist()

    def history(self, tree, history):
        """Register a game's {position: occurrences} for tree `tree`; `advance` keeps it current."""
        keys = np.asarray(list(history), np.uint64).reshape(-1, 3)
        counts = np.asarray(list(history.values()), np.int32)
        self.check(self.lib.az_history(self.pool, tree, ptr(keys), ptr(counts), len(counts)))

    def drop(self, tree):
        self.check(self.lib.az_drop(self.pool, tree))

    def clear(self):
        self.check(self.lib.az_clear(self.pool))

    def negamax(self, positions, nodes=2000):
        """Negamax moves; `nodes` is one budget or one per position."""
        if not positions:
            return []
        packed = np.asarray(positions, np.uint64).reshape(-1, 3)
        budgets = np.broadcast_to(np.asarray(nodes, np.int32), len(positions)).copy()
        out = np.empty((len(positions), 3), np.int64)
        self.check(self.lib.az_negamax(self.opponents, ptr(packed), len(positions), ptr(budgets), ptr(out)))
        return out[:, 0].tolist()

    def negamax_async(self, positions, nodes):
        """`negamax` on a background thread; returns a future of the moves."""
        return self.background.submit(self.negamax, positions, nodes)

    def features(self, positions, repetitions, remaining, mirror):
        n = len(positions)
        out = np.empty((n, self.planes, 9, 9), np.float32)
        self.check(self.lib.az_features(
            self.pool, ptr(np.ascontiguousarray(positions, np.uint64).reshape(-1, 3)),
            ptr(np.ascontiguousarray(repetitions, np.int32)), ptr(np.ascontiguousarray(remaining, np.float32)),
            ptr(np.ascontiguousarray(mirror, np.uint8)), n, ptr(out)))
        return out

    def clear_cache(self):
        self.check(self.lib.az_cache_clear(self.pool))

    def prepare(self, model, device):
        """Start a new set of searches; reset the evaluation cache if the network changed."""
        # BatchNorm buffers also change during training. Never share values between
        # networks or optimizer generations.
        version = (model, tuple(t._version for t in model.parameters()),
                   tuple(t._version for t in model.buffers()))
        if version != self.cache_version:
            self.clear_cache()
            self.cache_version = version
            self.net = self.inference_net(model) if device == 'cuda' else model
        if self.pinned != (device == 'cuda'):
            self.pinned = device == 'cuda'
            self.allocate(0)
        self.check(self.lib.az_cache(self.pool, self.cache_limit))
        self.check(self.lib.az_start(self.pool))

    def inference_net(self, model):
        """Compiled search copy of `model`: BatchNorm folded into the convolutions, bf16 channels-last weights,
        replayed as CUDA graphs. One copy per architecture; later networks are loaded into it in place."""
        net = copy.deepcopy(model).eval()
        for parent in net.modules():
            names = list(parent._modules)
            for a, b in zip(names, names[1:]):
                if isinstance(parent._modules[a], nn.Conv2d) and isinstance(parent._modules[b], nn.BatchNorm2d):
                    parent._modules[a] = torch.nn.utils.fusion.fuse_conv_bn_eval(parent._modules[a], parent._modules[b])
                    parent._modules[b] = nn.Identity()
        net = net.to(torch.bfloat16, memory_format=torch.channels_last)
        key = json.dumps(model.config, sort_keys=True)
        if key not in self.inference:
            self.inference[key] = (net, Graphs(torch.compile(net, dynamic=True)))
        else:
            with torch.no_grad():
                self.inference[key][0].load_state_dict(net.state_dict())
        return self.inference[key][1]

    def begin(self, job, simulations, group=0):
        """Queue one search in a wave group; returns its slot. The job's tree needs a registered `history`."""
        seed = job.get('seed')
        slot = self.check(self.lib.az_begin(self.pool, job['tree'], ptr(np.asarray(job['position'], np.uint64)),
                                            job['ply'], job['max_plies'], simulations, seed or 0, seed is not None, group))
        if slot >= self.capacity:
            self.allocate(2 * slot + 2)
        return slot

    def allocate(self, capacity):
        """Per search group: feature, legal-move and count rows for the next wave (pinned on CUDA).
        A batch already queued on the device keeps its old buffers alive until collected."""
        self.capacity = capacity
        shapes = (((capacity, self.planes, 9, 9), torch.float32), ((capacity, 136), torch.int32), ((capacity,), torch.int32))
        self.buffers = [tuple(torch.empty(shape, dtype=dtype, pin_memory=self.pinned).numpy() for shape, dtype in shapes)
                        for _ in range(2)]

    def launch(self, group, device):
        """Advance every search of `group` to its next uncached leaf and queue those leaves on the
        device without waiting. Returns the number of rows (0 once the group has nothing to evaluate)."""
        assert group not in self.inflight
        features, legal, counts = self.buffers[group]
        start = time.perf_counter()
        k = self.check(self.lib.az_wave(self.pool, group, ptr(features), ptr(legal), ptr(counts)))
        if k:
            self.inflight[group] = (k, priors(self.net, features[:k], legal[:k], counts[:k], device))
        self.stats['search_seconds'] += time.perf_counter() - start
        return k

    def collect(self, group):
        """Wait for `group`'s queued evaluations and back them up."""
        k, (probabilities, values, done) = self.inflight.pop(group)
        start = time.perf_counter()
        if done is not None:
            done.synchronize()
        self.stats['forward_seconds'] += time.perf_counter() - start
        start = time.perf_counter()
        self.check(self.lib.az_evaluate(self.pool, group, ptr(probabilities), ptr(values), k))
        self.stats['search_seconds'] += time.perf_counter() - start
        self.sync_stats()

    def sync_stats(self):
        out = np.empty(9, np.int64)
        self.check(self.lib.az_stats(self.pool, ptr(out)))
        hits, duplicates, unique, waves, race_queries, race_solved, race_unknown, race_nodes, race_ns = out.tolist()
        self.stats.update(cache_hits=hits + duplicates, unique_evaluations=unique, waves=waves,
                          leaves=hits + duplicates + unique, race_queries=race_queries, race_solved=race_solved,
                          race_unknown=race_unknown, race_nodes=race_nodes, race_seconds=race_ns / 1e9)

    def step(self, device):
        """One synchronous wave of group 0; returns its row count (0 once every search is finished)."""
        k = self.launch(0, device)
        if k:
            self.collect(0)
        return k

    def finished(self):
        out = np.empty(self.capacity, np.int32)
        return out[:self.check(self.lib.az_finished(self.pool, ptr(out)))].tolist()

    def take(self, slots):
        """Policy targets [n, 209], root values and moves of finished searches; frees the slots.

        PUCT: root visit distributions, moves -1 (the caller picks). Gumbel: improved policies over
        all legal moves and the moves chosen by sequential halving."""
        n = len(slots)
        policies, roots, moves = np.empty((n, MOVES), np.float32), np.empty(n, np.float32), np.empty(n, np.int32)
        self.check(self.lib.az_take(self.pool, ptr(np.asarray(slots, np.int32)), n, ptr(policies), ptr(roots),
                                    ptr(moves)))
        self.sync_stats()
        if n and self.search_settings['search'] == 'gumbel':
            # Diagnostics: entropy of the improved policy targets.
            p = policies.astype(np.float64)
            self.stats['target_entropy'] = self.stats.get('target_entropy', 0.0) - float(
                np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0).sum())
            self.stats['targets'] = self.stats.get('targets', 0) + n
        return policies, roots, moves

    def search(self, model, jobs, simulations, device, moves=False):
        """jobs: dicts with tree, position, history {position: count}, ply, max_plies, seed (optional:
        Dirichlet root noise, or Gumbel noise). Returns real-move policy targets [n, 209] and root values
        (side to move); with `moves`, also the chosen moves (-1 under PUCT: take the target's argmax)."""
        self.prepare(model, device)
        for job in jobs:
            self.history(job['tree'], job['history'])
        slots = [self.begin(job, simulations) for job in jobs]
        while self.step(device):
            pass
        out = self.take(slots)
        return out if moves else out[:2]


class Graphs:
    """CUDA graph replay of a network, one graph per batch size. Queueing a batch then costs
    ~0.5 ms of CPU instead of ~3 ms of kernel launches (L4, 128x10). Graphs replay in stream
    order and share one memory pool; each output is overwritten by the next replay of its size."""

    def __init__(self, net):
        self.net, self.graphs, self.pool = net, {}, torch.cuda.graph_pool_handle()

    def __call__(self, x):
        if len(x) not in self.graphs:
            static = torch.zeros_like(x)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.net(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self.net(static)
            self.graphs[len(x)] = graph, static, out
        graph, static, out = self.graphs[len(x)]
        static.copy_(x)
        graph.replay()
        return out


# ---------------------------------------------------------------- network

class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Global pooling bias (KataGo-style): walls far away matter for path lengths.
        self.pool = nn.Linear(channels, channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = y + self.pool(y.mean((2, 3)))[:, :, None, None]
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class Net(nn.Module):
    """Residual tower; policy logits are laid out on the board (pawn squares, h/v wall anchors)."""

    def __init__(self, planes=19, channels=64, blocks=6, drop_planes=()):
        super().__init__()
        self.config = dict(planes=planes, channels=channels, blocks=blocks, drop_planes=list(drop_planes))
        # Ablation: zero selected input planes (e.g. the engineered path features).
        mask = torch.ones(planes)
        mask[list(drop_planes)] = 0
        self.register_buffer('mask', mask[None, :, None, None], persistent=False)
        self.stem = nn.Sequential(nn.Conv2d(planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*(Block(channels) for _ in range(blocks)))
        self.policy_head = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.Conv2d(32, 3, 1))
        self.value_head = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
                                        nn.Flatten(), nn.Linear(4 * 81, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.tower(self.stem(x * self.mask))
        p = self.policy_head(h)
        logits = torch.cat((p[:, 0].flatten(1), p[:, 1, :8, :8].flatten(1), p[:, 2, :8, :8].flatten(1)), 1)
        return logits, self.value_head(h).squeeze(1).tanh()


@torch.inference_mode()
def priors(net, features, legal, counts, device):
    """Log-softmax over each row's legal canonical moves (in native edge order), plus values.

    On CUDA `net` is an `Engine.inference_net` copy and the work is only queued: the
    returned pinned arrays are filled once the returned event completes (else None)."""
    n = len(features)
    cuda = device == 'cuda'
    x = torch.from_numpy(features).to(device, non_blocking=True)
    if cuda:
        if n % 64:
            # Every new batch shape costs cuDNN a fresh kernel plan (~3x a cached one on an
            # L4); padding keeps the shapes to a few dozen. Rows are independent in eval mode.
            x = torch.cat((x, x.new_zeros((64 - n % 64, *x.shape[1:]))))
        x = x.to(torch.bfloat16, memory_format=torch.channels_last)
    logits, values = net(x)
    logits = logits[:n].float().gather(1, torch.from_numpy(legal).to(device, non_blocking=True).long())
    mask = torch.arange(136, device=device)[None, :] >= torch.from_numpy(counts).to(device, non_blocking=True)[:, None]
    probabilities = logits.masked_fill(mask, -torch.inf).log_softmax(1)
    values = values[:n].float()
    if not cuda:
        return np.ascontiguousarray(probabilities.cpu().numpy()), np.ascontiguousarray(values.cpu().numpy()), None
    host = [torch.empty(t.shape, pin_memory=True).copy_(t, non_blocking=True) for t in (probabilities, values)]
    done = torch.cuda.Event()
    done.record()
    return host[0].numpy(), host[1].numpy(), done


@torch.inference_mode()
def infer(model, features, device):
    x = torch.from_numpy(features).to(device, non_blocking=True)
    with torch.autocast('cuda', torch.bfloat16, enabled=device == 'cuda'):
        logits, values = model(x)
    return logits.float().cpu(), values.float().cpu()


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        if path.suffix == '.gz':
            # Archived snapshots only: replay policy rows are mostly zeros and compress well.
            with gzip.GzipFile(fileobj=stream, mode='wb', compresslevel=1, mtime=0) as packed:
                torch.save(state, packed)
        else:
            torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_model(model, path, **extra):
    atomic_save(dict(config=model.config, state=model.state_dict(), **extra), path)


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = Net(**saved['config'])
    model.load_state_dict(saved['state'])
    return model.to(device).eval(), saved


def pick_device(name):
    if name != 'auto':
        return name
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ---------------------------------------------------------------- replay and training

class Replay:
    """Ring buffer of packed positions; features are rebuilt (with random mirroring) per batch."""

    def __init__(self, capacity):
        self.capacity, self.size, self.next = capacity, 0, 0
        self.position = np.zeros((capacity, 3), np.uint64)
        self.repetitions = np.ones(capacity, np.int32)
        self.remaining = np.ones(capacity, np.float32)
        self.policy = np.zeros((capacity, MOVES), np.float16)  # canonical move IDs, unmirrored
        self.has_policy = np.zeros(capacity, bool)
        self.value = np.zeros(capacity, np.float32)

    def add(self, position, repetitions, remaining, policy, value):
        i = self.next
        self.position[i], self.repetitions[i], self.remaining[i], self.value[i] = position, repetitions, remaining, value
        self.has_policy[i] = policy is not None
        self.policy[i] = 0 if policy is None else policy
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, rows):
        for row in rows:
            self.add(*row)

    def snapshot(self):
        return dict(capacity=self.capacity, size=self.size, next=self.next,
                    data={name: getattr(self, name)[:self.size].copy() for name in
                          ('position', 'repetitions', 'remaining', 'policy', 'has_policy', 'value')})

    @classmethod
    def restore(cls, state):
        replay = cls(state['capacity'])
        replay.size, replay.next = state['size'], state['next']
        for name, values in state['data'].items():
            getattr(replay, name)[:replay.size] = values
        return replay

    def save(self, path):
        n = self.size
        path = Path(path)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, position=self.position[:n], repetitions=self.repetitions[:n], remaining=self.remaining[:n],
                     policy=self.policy[:n], has_policy=self.has_policy[:n], value=self.value[:n], next=self.next)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, capacity=None):
        data = np.load(path)
        n = len(data['value'])
        replay = cls(max(capacity or n, n))
        for name in ('position', 'repetitions', 'remaining', 'policy', 'has_policy', 'value'):
            getattr(replay, name)[:n] = data[name]
        replay.size, replay.next = n, int(data['next']) % replay.capacity if 'next' in data else n % replay.capacity
        return replay


def train_steps(model, optimizer, replay, engine, steps, batch_size, device, rng, value_weight=1.0, scheduler=None):
    model.train()
    totals = dict(policy=0.0, value=0.0)
    for _ in range(steps):
        idx = rng.integers(0, replay.size, batch_size)
        mirror = rng.integers(0, 2, batch_size).astype(np.uint8)
        x = engine.features(replay.position[idx], replay.repetitions[idx], replay.remaining[idx], mirror)
        policy = replay.policy[idx].astype(np.float32)
        flipped = mirror.astype(bool)
        # Mirroring is applied after canonicalization (the two axes commute).
        policy[flipped] = policy[flipped][:, engine.canonical[0, 1]]
        x = torch.from_numpy(x).to(device)
        target = torch.from_numpy(policy).to(device)
        mask = torch.from_numpy(replay.has_policy[idx]).to(device)
        value = torch.from_numpy(replay.value[idx]).to(device)
        with torch.autocast('cuda', torch.bfloat16, enabled=device == 'cuda'):
            logits, prediction = model(x)
        logits, prediction = logits.float(), prediction.float()
        rows = -(target * logits.log_softmax(1)).sum(1)
        policy_loss = (rows * mask).sum() / mask.sum().clamp(min=1)
        value_loss = (prediction - value).square().mean()
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        totals['policy'] += policy_loss.item()
        totals['value'] += value_loss.item()
    model.eval()
    return {k: v / max(steps, 1) for k, v in totals.items()}


# ---------------------------------------------------------------- games

def random_opening(engine, rng, lengths, wall_probability=0.25):
    """Same procedure as curriculum.evaluation_opening (mostly pawn moves, some walls)."""
    position, history = START, {START: 1}
    for _ in range(lengths):
        legal = engine.legal(position)
        candidates = list(legal if rng.random() < wall_probability else [m for m in legal if m < 81])
        rng.shuffle(candidates)
        for move in candidates:
            child = engine.advance(position, move)
            if winner(child) == -1 and history.get(child, 0) < 2:
                position = child
                history[child] = history.get(child, 0) + 1
                break
    return position, history


def evaluation_opening(engine, index, suite='monitor', seed=1):
    """Bit-identical to curriculum.evaluation_opening(varied_eval=True) for the same suite/seed."""
    rng = random.Random(f"eval:{suite}:{seed}:{index // 2}")
    lengths = (10, 12, 14, 16, 18) if suite == 'heldout' else (0, 2, 4, 6, 8)
    return random_opening(engine, rng, lengths[index // 2 % 5])


class Game:
    def __init__(self, ident, position, history, players, record=True, sample_plies=0, opening=None):
        self.id, self.position, self.history = ident, position, dict(history)
        self.players = players  # per color: 'mcts' (the play() model), a Net, or ('negamax', nodes)
        # Two different networks keep separate search trees (one per color).
        self.split = all(isinstance(p, nn.Module) for p in players) and players[0] is not players[1]
        self.ply, self.moves, self.samples = 0, [], []
        self.record, self.sample_plies = record, sample_plies
        self.result = None  # winner color, or -1 for an artificial cutoff
        self.reason = None
        self.opening = opening or position

    def to_move(self):
        return self.players[turn(self.position)]

    def tree(self, color=None):
        return 2 * self.id + (turn(self.position) if color is None else color) if self.split else self.id

    def trees(self):
        return [self.tree(0), self.tree(1)] if self.split else [self.id]


def play(engine, model, games, device, simulations, max_plies=100, noise=False, temperature_plies=0,
         rng=None, progress=None, resume=False, checkpoint=None, spawn=None, until=None,
         playout_cap='fixed'):
    """Advance all games to completion; negamax turns use the C++ thread pool.

    With a single network, every game starts its next search as soon as its own move
    is played, so short searches never wait for the slowest game. Negamax replies run
    on a background thread and are applied one loop pass later. Each game draws
    noise and temperature samples from its own stream, so results do not depend on
    this scheduling. `progress(done, total)` is called only when no search is in
    flight: whenever `checkpoint()` returns true, or after every move round when
    `checkpoint` is omitted.

    `spawn(ended)` may return a game to start in place of one that ended; it is
    appended to `games`. Once `until()` is true no new search starts, and play
    returns when none is in flight, leaving unfinished games resumable."""
    rng = rng or random.Random(0)
    active = [g for g in games if g.result is None]
    for game in ([] if resume else active):
        for tree in game.trees():
            engine.drop(tree)
    for g in active:
        if getattr(g, 'rng', None) is None:
            g.rng = random.Random(rng.getrandbits(64))
        for tree in g.trees():
            engine.history(tree, g.history)

    def network(g):
        player = g.to_move()
        return model if isinstance(player, str) else player if isinstance(player, nn.Module) else None

    def job(g):
        job = dict(tree=g.tree(), position=g.position, ply=g.ply, max_plies=max_plies)
        # Under Gumbel search the seed draws the root Gumbels (Dirichlet noise is disabled).
        if noise:
            job['seed'] = g.rng.getrandbits(64)
        return job

    def record(g, policy):
        if g.record:
            if getattr(g, 'value_only_cap', False):
                policy = None  # Low-cap searches train value only; policy targets use the 400-cap cohort.
            if policy is not None:
                g.policy_rows = getattr(g, 'policy_rows', 0) + 1
            else:
                g.value_only_rows = getattr(g, 'value_only_rows', 0) + 1
            g.samples.append([g.position, g.history.get(g.position, 1), (max_plies - g.ply) / max_plies,
                              policy, turn(g.position)])

    def budget(g):
        selected = playout_budget(g.rng, playout_cap, simulations)
        g.value_only_cap = playout_cap == 'randomized-40-400' and selected == 40
        counts = g.playout_counts = getattr(g, 'playout_counts', {'40': 0, '400': 0})
        counts[str(selected)] = counts.get(str(selected), 0) + 1
        return selected

    gumbel = engine.search_settings['search'] == 'gumbel'

    def searched(g, pi, move=-1):
        """Gumbel searches choose `move` themselves: never sample it from the target `pi`."""
        if move >= 0:
            pass
        elif g.ply < temperature_plies and not gumbel:
            move = int(np.searchsorted(np.cumsum(pi, dtype=np.float64), g.rng.random() * float(pi.sum(dtype=np.float64)),
                                       side='right'))
        else:
            move = int(pi.argmax())
        # Canonical and real IDs map onto each other by an involution.
        record(g, pi[engine.canonical[turn(g.position), 0]])
        return move

    def advance(moved):
        """Play each (game, move); returns the games still running."""
        if not moved:
            return []
        after = engine.advance_many([g.tree() for g, _ in moved], [g.position for g, _ in moved], [m for _, m in moved])
        running = []
        for (g, move), position in zip(moved, after):
            if g.split:
                engine.advance(g.position, move, g.tree(turn(g.position) ^ 1))
            g.position = position = tuple(position)
            g.moves.append(move)
            g.ply += 1
            g.history[position] = g.history.get(position, 0) + 1
            w = winner(position)
            if w != -1:
                g.result, g.reason = w, 'goal'
            elif g.history[position] >= 3:
                g.result, g.reason = -1, 'repetition'
            elif g.ply >= max_plies:
                g.result, g.reason = -1, 'plies'
            if g.result is None:
                running.append(g)
            else:
                for tree in g.trees():
                    engine.drop(tree)
        return running

    def replied(waiting, moves):
        """Apply negamax moves; returns the (game, move) pairs to play."""
        for g in waiting:
            record(g, None)
        return list(zip(waiting, moves))

    def done():
        return sum(g.result is not None for g in games)

    networks = {id(n): n for g in active for n in (model if isinstance(p, str) else p for p in g.players)
                if isinstance(n, nn.Module)}
    if len(networks) > 1:
        assert spawn is None and until is None
        # Different networks cannot share a batch; advance all games one move at a time.
        while active:
            groups = {}
            for g in active:
                if network(g) is not None:
                    groups.setdefault(id(network(g)), []).append(g)
            waiting = [g for g in active if network(g) is None]
            still = advance(replied(waiting, engine.negamax([g.position for g in waiting], [g.to_move()[1] for g in waiting])))
            for group in groups.values():
                by_budget = {}
                for g in group:
                    g.current_budget = budget(g)
                    by_budget.setdefault(g.current_budget, []).append(g)
                for b, batch in by_budget.items():
                    policies, _, moves = engine.search(network(batch[0]),
                                                       [job(g) | dict(history=g.history) for g in batch],
                                                       b, device, moves=True)
                    still += advance([(g, searched(g, pi, int(m))) for g, pi, m in zip(batch, policies, moves)])
            order = {g.id: i for i, g in enumerate(games)}
            active = sorted(still, key=lambda g: order[g.id])
            if progress:
                progress(done(), len(games))
        return

    if checkpoint is None:
        checkpoint = (lambda: True) if progress else (lambda: False)
    net = next(iter(networks.values()), None)
    if net is not None:
        engine.prepare(net, device)
    # Two search groups alternate: while one group's batch is on the device, the other
    # group's evaluations are backed up and its next leaves selected.
    idle, slots, size, draining, stopping, group = active, {}, [0, 0], False, False, 0
    replies = None  # (games, future moves) of negamax turns in flight
    while idle or slots or replies:
        if group in engine.inflight:
            engine.collect(group)
        finished = engine.finished()
        moved = []
        policies, _, chosen = engine.take(finished)
        for slot, pi, m in zip(finished, policies, chosen):
            g, member = slots.pop(slot)
            size[member] -= 1
            moved.append((g, searched(g, pi, int(m))))
        if replies:
            moved += replied(replies[0], replies[1].result())
            replies = None
        idle += advance(moved)
        if spawn:
            for g, _ in moved:
                if g.result is not None and (new := spawn(g)) is not None:
                    new.rng = random.Random(rng.getrandbits(64))
                    engine.history(new.tree(), new.history)
                    games.append(new)
                    idle.append(new)
        if not (draining or stopping):
            waiting = [g for g in idle if network(g) is None]
            if waiting:
                replies = waiting, engine.negamax_async([g.position for g in waiting], [g.to_move()[1] for g in waiting])
            for g in idle:
                if network(g) is not None:
                    member = int(size[1] < size[0])
                    g.current_budget = budget(g)
                    slots[engine.begin(job(g), g.current_budget, member)] = g, member
                    size[member] += 1
            idle = []
        engine.launch(group, device)
        group ^= 1
        draining = draining or checkpoint()
        stopping = stopping or (until is not None and until())
        if stopping and not slots and not replies:
            return
        if draining and not slots and not replies:
            draining = False
            if progress:
                progress(done(), len(games))


def samples(game):
    """Replay rows (position, repetitions, remaining, canonical policy|None, value for side to move)."""
    rows = []
    for position, repetitions, remaining, policy, t in game.samples:
        value = 0.0 if game.result == -1 else (1.0 if game.result == t else -1.0)
        rows.append((position, repetitions, remaining, policy, value))
    return rows


def match(engine, model, device, pairs, simulations, nodes=2000, suite='monitor', seed=1, max_plies=100):
    """Paired-color match vs fixed negamax on train.py-compatible openings. No noise, argmax moves."""
    games = []
    for i in range(2 * pairs):
        position, history = evaluation_opening(engine, i, suite, seed)
        learner = i % 2
        players = ['mcts', ('negamax', nodes)] if learner == 0 else [('negamax', nodes), 'mcts']
        g = Game(100000 + i, position, history, players, record=False)
        g.learner = learner
        games.append(g)
    play(engine, model, games, device, simulations, max_plies)
    wins = sum(g.result == g.learner for g in games)
    cutoffs = sum(g.result == -1 for g in games)
    return dict(games=len(games), wins=wins, losses=len(games) - wins - cutoffs, cutoffs=cutoffs,
                score=(wins + 0.5 * cutoffs) / len(games),
                wins_by_color=[sum(g.result == g.learner for g in games if g.learner == c) for c in (0, 1)],
                mean_plies=sum(g.ply for g in games) / len(games))


def head_to_head(engine, a, b, device, pairs, simulations, suite='heldout', seed=1, max_plies=100):
    """Paired-color match between two networks on evaluation openings; scores are from a's side."""
    games = []
    for i in range(2 * pairs):
        position, history = evaluation_opening(engine, i, suite, seed)
        g = Game(100000 + i, position, history, [a, b] if i % 2 == 0 else [b, a], record=False)
        g.learner = i % 2
        games.append(g)
    play(engine, None, games, device, simulations, max_plies)
    wins = sum(g.result == g.learner for g in games)
    cutoffs = sum(g.result == -1 for g in games)
    return dict(games=len(games), wins=wins, losses=len(games) - wins - cutoffs, cutoffs=cutoffs,
                score=(wins + 0.5 * cutoffs) / len(games),
                wins_by_color=[sum(g.result == g.learner for g in games if g.learner == c) for c in (0, 1)],
                mean_plies=sum(g.ply for g in games) / len(games))


# ---------------------------------------------------------------- commands

def settings(args):
    return {k: v for k, v in vars(args).items() if k != 'fn'}


def search_settings(args, mode=None):
    """Engine search settings from parsed arguments; `mode` overrides --search (e.g. --eval-search)."""
    out = {k: getattr(args, k) for k in SEARCH_DEFAULTS if hasattr(args, k)}
    if mode:
        out['search'] = mode
    return out


def cmd_gen(args):
    """Negamax self-play data: random openings, mixed node budgets, occasional random moves.

    Policy target = the negamax move at that position (even when a random move was played)."""
    engine = Engine(args.workers)
    rng = random.Random(f'gen:{args.seed}')
    replay = Replay(args.games * 110)
    budgets = [int(b) for b in args.budgets.split(',')]
    done, start, next_id = 0, time.time(), 0
    checkpoint = Path(str(args.out) + '.state.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for key, value in saved['args'].items():
            if key not in ('fn', 'out', 'workers', 'device'):
                setattr(args, key, value)
        budgets = [int(b) for b in args.budgets.split(',')]
        replay = Replay.restore(saved['replay'])
        done, next_id = saved['done'], saved['next_id']
        rng.setstate(saved['rng'])
    while done < args.games:
        batch = []
        for _ in range(min(args.parallel, args.games - done)):
            position, history = random_opening(engine, rng, rng.randint(0, args.max_opening))
            g = Game(next_id, position, history, [('negamax', rng.choice(budgets)), ('negamax', rng.choice(budgets))])
            next_id += 1
            batch.append(g)
        active = list(batch)
        while active:
            moves = {}
            groups = {}
            for g in active:
                groups.setdefault(g.to_move()[1], []).append(g)
            for nodes, group in groups.items():
                for g, m in zip(group, engine.negamax([g.position for g in group], nodes)):
                    moves[g.id] = m
            still = []
            for g in active:
                target = moves[g.id]
                t = turn(g.position)
                policy = np.zeros(MOVES, np.float32)
                policy[engine.canonical[t, 0, target]] = 1
                g.samples.append([g.position, g.history.get(g.position, 1), (args.max_plies - g.ply) / args.max_plies,
                                  policy, t])
                move = target
                if rng.random() < args.epsilon:
                    legal = engine.legal(g.position)
                    pawn = [m for m in legal if m < 81]
                    move = rng.choice(pawn if rng.random() < 0.5 else legal)
                g.position = engine.advance(g.position, move)
                g.ply += 1
                g.history[g.position] = g.history.get(g.position, 0) + 1
                w = winner(g.position)
                if w != -1:
                    g.result, g.reason = w, 'goal'
                elif g.history[g.position] >= 3 or g.ply >= args.max_plies:
                    g.result, g.reason = -1, 'cutoff'
                if g.result is None:
                    still.append(g)
            active = still
        for g in batch:
            replay.extend(samples(g))
        done += len(batch)
        atomic_save(dict(args=settings(args), replay=replay.snapshot(), done=done, next_id=next_id,
                         rng=rng.getstate()), checkpoint)
        print(f'gen {done}/{args.games} games, {replay.size} positions, {time.time() - start:.0f}s', flush=True)
    replay.save(args.out)


def new_optimizer(model, lr, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def new_rl_optimizer(model, lr, name='adamw', weight_decay=1e-4):
    """Build the RL optimizer. Muon is opt-in; output heads and vector parameters use AdamW."""
    if name == 'adamw':
        return new_optimizer(model, lr, weight_decay)
    if name != 'muon':
        raise ValueError(f'Unknown optimizer {name!r}')
    muon, adam = [], []
    for pname, param in model.named_parameters():
        # Keep both prediction heads on AdamW. Muon's matrix update is for the shared trunk.
        head = pname.startswith(('policy_head.', 'value_head.'))
        (muon if not head and param.ndim >= 2 else adam).append(param)
    # Conv parameters are temporarily represented by the same shaped tensor. PyTorch Muon
    # operates on matrices, so wrap it in a shape-preserving flattened parameter optimizer.
    return MuonAdamW(model, muon, adam, lr, weight_decay)


class MuonAdamW(torch.optim.Optimizer):
    """Muon for trunk matrices (including flattened conv kernels), AdamW otherwise."""
    def __init__(self, model, muon_params, adam_params, lr, weight_decay):
        super().__init__([{'params': muon_params}, {'params': adam_params}], dict(lr=lr, weight_decay=weight_decay))
        self.model_muon_params = list(muon_params)
        self.flat_params = [nn.Parameter(p.detach().reshape(p.shape[0], -1).contiguous()) for p in muon_params]
        self.muon_impl = torch.optim.Muon(self.flat_params, lr=lr, weight_decay=weight_decay,
                                          adjust_lr_fn='match_rms_adamw')
        self.adam_impl = torch.optim.AdamW(adam_params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.muon_impl.param_groups + self.adam_impl.param_groups

    def step(self, closure=None):
        # Scheduler adapters expect a conventional optimizer.step method.
        return self._step(closure)

    def zero_grad(self, *args, **kwargs):
        for p in self.model_muon_params:
            p.grad = None
        self.muon_impl.zero_grad(*args, **kwargs)
        self.adam_impl.zero_grad(*args, **kwargs)

    @torch.no_grad()
    def _step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        for model_p, flat_p in zip(self.model_muon_params, self.flat_params):
            flat_p.copy_(model_p.detach().reshape(model_p.shape[0], -1))
            if model_p.grad is None:
                flat_p.grad = None
            else:
                flat_p.grad = model_p.grad.detach().reshape(model_p.shape[0], -1).contiguous().clone()
        self.muon_impl.step()
        for model_p, flat_p in zip(self.model_muon_params, self.flat_params):
            model_p.copy_(flat_p.reshape_as(model_p))
        self.adam_impl.step()
        return loss

    def state_dict(self):
        return {'muon': self.muon_impl.state_dict(), 'adamw': self.adam_impl.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon_impl.load_state_dict(state_dict['muon'])
        self.adam_impl.load_state_dict(state_dict['adamw'])
        # PyTorch replaces child param group dicts on load; reconnect the scheduler view.
        self.param_groups = self.muon_impl.param_groups + self.adam_impl.param_groups

def make_rl_scheduler(optimizer, kind, total_steps, lr):
    if kind == 'none':
        return None
    if kind == 'cosine':
        horizon = max(1, total_steps)
        floor = 0.05
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(step, horizon) / horizon)))
    if kind == 'cosine-restarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps // 4), T_mult=2,
                                                                    eta_min=lr * 0.05)
    raise ValueError(f'Unknown RL schedule {kind!r}')


def restore_rl_scheduler(scheduler, state, optimizer):
    """Restore scheduler phase and the live optimizer LR before the next gradient update."""
    scheduler.load_state_dict(state)
    last_lrs = state.get('_last_lr')
    if last_lrs is not None:
        if len(last_lrs) != len(optimizer.param_groups):
            raise ValueError('Scheduler checkpoint does not match optimizer parameter groups')
        for group, lr in zip(optimizer.param_groups, last_lrs):
            group['lr'] = lr


def cmd_pretrain(args):
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'pretrain-state.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if saved:
        for key, value in saved['args'].items():
            if key not in ('fn', 'out', 'workers', 'device'):
                setattr(args, key, value)
    device = pick_device(args.device)
    engine = Engine(args.workers)
    data = Replay.load(args.data)
    model = Net(engine.planes, args.channels, args.blocks, args.drop_planes).to(device)
    if device == 'cuda':
        model.to(memory_format=torch.channels_last)
    optimizer = new_optimizer(model, args.lr)
    steps_per_epoch = data.size // args.batch_size
    total = int(steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=total, pct_start=0.05)
    rng = np.random.default_rng(args.seed)
    print(f'{data.size} positions, {sum(p.numel() for p in model.parameters())} parameters, {total} steps', flush=True)
    start, step, log = time.time(), 0, []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if saved:
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, log = saved['step'], saved['log']
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        if device == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])

    def checkpoint_pretrain():
        save_model(model, checkpoint, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   step=step, log=log, args=settings(args), rng=rng.bit_generator.state,
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    checkpoint_pretrain()
    while step < total:
        chunk = min(200, total - step)
        model.train()
        totals = dict(policy=0.0, value=0.0)
        for _ in range(chunk):
            losses = train_steps(model, optimizer, data, engine, 1, args.batch_size, device, rng)
            scheduler.step()
            for k in totals:
                totals[k] += losses[k] / chunk
        step += chunk
        log.append(dict(step=step, **totals, seconds=time.time() - start))
        checkpoint_pretrain()
        print(f"step {step}/{total} policy {totals['policy']:.3f} value {totals['value']:.3f} {time.time() - start:.0f}s",
              flush=True)
    save_model(model, out / 'model.pt', stage='pretrain', args=settings(args))
    result = None
    if args.eval_pairs:
        result = match(engine, model, device, args.eval_pairs, args.simulations)
        print('eval', json.dumps(result), flush=True)
    (out / 'pretrain.json').write_text(json.dumps(dict(args=settings(args), log=log, eval=result), indent=1))


def save_rl_state(path, model, optimizer, replay, iteration, metrics, args, rng, nprng, games=None, engine=None,
                  scheduler=None):
    # One atomic file: a lost VM cannot mix replay from one generation with weights
    # from another. Save only live replay rows, preserving ring cursor and capacity.
    save_model(model, path, optimizer=optimizer.state_dict(), iteration=iteration, metrics=metrics,
               args=settings(args), replay=replay.snapshot(),
               games=[vars(g) for g in games] if games is not None else None,
               trees=engine.snapshot() if games is not None else None,
               rng=rng.getstate(), nprng=nprng.bit_generator.state, torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               scheduler=scheduler.state_dict() if scheduler is not None else None)


def cmd_rl(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    engine = Engine(args.workers, args.cpuct, args.fpu, **search_settings(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'state.pt'
    if not state_path.exists() and args.resume_from:
        # Start a new run directory from an archived snapshot (e.g. state-h012-it00456.pt.gz).
        temporary = state_path.with_name(state_path.name + f'.{os.getpid()}.tmp')
        with (gzip.open if args.resume_from.endswith('.gz') else open)(args.resume_from, 'rb') as source, \
                temporary.open('wb') as target:
            shutil.copyfileobj(source, target, 1 << 24)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, state_path)
        print(f'restored {args.resume_from}', flush=True)
    if state_path.exists():
        model, saved = load_model(state_path, device)
        iteration, metrics = saved['iteration'], saved['metrics']
        if 'args' in saved:
            if '--iterations' not in sys.argv:
                args.iterations = saved['args']['iterations']
            if '--budget-minutes' not in sys.argv:
                args.budget_minutes = saved['args'].get('budget_minutes', 0)
            # Runs saved before Gumbel search existed used PUCT, whatever the command line says.
            for key, value in (SEARCH_DEFAULTS | dict(eval_search='puct') | saved['args']).items():
                if key not in ('fn', 'device', 'workers', 'out', 'iterations', 'checkpoint_seconds', 'budget_minutes',
                               'snapshot_dir', 'snapshot_minutes', 'snapshot_state', 'latest_minutes', 'resume_from'):
                    setattr(args, key, value)
        # Checkpoints predating the RL ablations resolve to historical defaults,
        # regardless of new command-line options supplied during resume.
        saved_args = saved.get('args', {})
        for key, value in {'optimizer': 'adamw', 'schedule': 'none', 'schedule_steps': 25000,
                           'playout_cap': 'fixed'}.items():
            if key not in saved_args:
                setattr(args, key, value)
        for key, value in SEARCH_DEFAULTS.items():
            if key not in saved_args:
                setattr(args, key, value)
        optimizer = new_rl_optimizer(model, args.lr, args.optimizer)
        optimizer.load_state_dict(saved['optimizer'])
        if 'replay' in saved:
            replay = Replay.restore(saved['replay'])
        elif (out / 'replay.npz').exists():
            replay = Replay.load(out / 'replay.npz', args.replay)
        else:
            raise RuntimeError('Legacy checkpoint is missing replay.npz; cannot fully resume')
        engine.configure(args.cpuct, args.fpu, **search_settings(args))
        print(f'resumed at iteration {iteration}', flush=True)
    else:
        if args.init:
            model, _ = load_model(args.init, device)
        else:
            model = Net(engine.planes, args.channels, args.blocks).to(device).eval()
        optimizer = new_rl_optimizer(model, args.lr, args.optimizer)
        iteration, metrics = 0, []
        replay = Replay(args.replay)
        if args.mix_data:
            # Keep some supervised negamax rows so early RL does not forget the bootstrap.
            data = Replay.load(args.mix_data)
            keep = np.random.default_rng(args.seed).permutation(data.size)[:args.replay // 4]
            for i in keep:
                replay.add(data.position[i], data.repetitions[i], data.remaining[i],
                           data.policy[i] if data.has_policy[i] else None, data.value[i])
        (out / 'config.json').write_text(json.dumps(settings(args), indent=1))
        if args.snapshot_dir:
            # Hour-0 anchor (the opponent of the first previous-snapshot match) and the settings, for reports.
            Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.snapshot_dir) / 'config.json').write_text(json.dumps(settings(args), indent=1))
            save_model(model, Path(args.snapshot_dir) / 'model-h000-it00000.pt', iteration=0, spent_seconds=0,
                       totals=dict(games=0, positions=0, steps=0, policy_rows=0))
    if device == 'cuda':
        # About 25% faster training steps on an L4; parameters keep their identity.
        model.to(memory_format=torch.channels_last)
    if not state_path.exists():
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    scheduler = make_rl_scheduler(optimizer, args.schedule, args.schedule_steps, args.lr)
    if state_path.exists() and saved.get('scheduler') is not None and scheduler is not None:
        restore_rl_scheduler(scheduler, saved['scheduler'], optimizer)
    rng = random.Random(f'rl:{args.seed}:{iteration}')
    nprng = np.random.default_rng([args.seed, iteration])
    if state_path.exists() and 'rng' in saved:
        rng.setstate(saved['rng'])
        nprng.bit_generator.state = saved['nprng']
        torch.set_rng_state(saved['torch_rng'])
        if device == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    # Games of the next iteration: carried over from the last one, or restored mid-iteration.
    games = None
    if state_path.exists() and saved.get('games') is not None:
        games = []
        for attributes in saved['games']:
            game = Game.__new__(Game)
            game.__dict__.update(attributes)
            games.append(game)
        engine.restore(saved['trees'])
    else:
        # Persist the starting generation too: interruption during the first rollout
        # replays that iteration with the same weights, replay and random streams.
        save_rl_state(state_path, model, optimizer, replay, iteration, metrics, args, rng, nprng,
                      scheduler=scheduler)

    def new_game(lane):
        """Lanes keep their opponent type, so every batch of games has the configured negamax share."""
        position, history = random_opening(engine, rng, rng.randint(0, args.max_opening))
        if lane < args.games * args.negamax_fraction:
            learner = lane % 2
            nodes = rng.choice(args.negamax_nodes)
            players = ['mcts', ('negamax', nodes)] if learner == 0 else [('negamax', nodes), 'mcts']
        else:
            learner, players = -1, ['mcts', 'mcts']
        g = Game(lane, position, history, players)
        g.learner = learner
        return g

    last_save = [time.monotonic()]
    def spent():
        """Self-play and training seconds of completed iterations (evaluations excluded)."""
        return sum(row['seconds'] for row in metrics)

    while iteration < args.iterations and not (args.budget_minutes and spent() >= 60 * args.budget_minutes):
        iteration += 1
        t0 = time.time()
        if games is None:
            games = [new_game(i) for i in range(args.games)]
        last = [0]

        def due():
            return time.monotonic() - last_save[0] >= args.checkpoint_seconds

        def finished():
            return sum(g.result is not None for g in games)

        def progress(done, total):
            if due():
                save_rl_state(state_path, model, optimizer, replay, iteration - 1, metrics, args, rng, nprng,
                              games=games, engine=engine, scheduler=scheduler)
                last_save[0] = time.monotonic()
            if done - last[0] >= max(1, args.games // 8):
                last[0] = done
                print(f'  iter {iteration}: {done}/{args.games} games done, {time.time() - t0:.0f}s', flush=True)
        # An iteration ends once `--games` games have finished. Every game that ends is
        # replaced at once, so batches stay full; games still running carry over.
        play(engine, model, games, device, args.simulations, args.max_plies, noise=True,
             temperature_plies=args.temperature_plies, rng=rng, progress=progress, resume=True,
             checkpoint=due, spawn=lambda ended: new_game(ended.id), until=lambda: finished() >= args.games,
             playout_cap=args.playout_cap)
        carried = [g for g in games if g.result is None]
        games = [g for g in games if g.result is not None]
        for g in carried:
            for tree in g.trees():
                engine.drop(tree)
        rows = 0
        for g in games:
            for row in samples(g):
                replay.add(*row)
                rows += 1
        played = time.time() - t0
        steps = max(1, int(rows * args.reuse / args.batch_size))
        losses = train_steps(model, optimizer, replay, engine, steps, args.batch_size, device, nprng,
                             scheduler=scheduler)
        vs = [g for g in games if g.learner >= 0]
        selfplay = [g for g in games if g.learner < 0]
        row = dict(iteration=iteration, games=len(games), carried=len(carried), positions=rows, replay=replay.size,
                   steps=steps, **losses,
                   train_wins=sum(g.result == g.learner for g in vs), train_vs_games=len(vs),
                   train_cutoffs=sum(g.result == -1 for g in games),
                   selfplay_first_wins=sum(g.result == 0 for g in selfplay),
                   mean_plies=sum(g.ply for g in games) / len(games),
                   playout_caps={str(cap): sum(getattr(g, 'playout_counts', {}).get(str(cap), 0) for g in games)
                                 for cap in (args.simulations, 40, 400)},
                   policy_rows=sum(getattr(g, 'policy_rows', 0) for g in games),
                   value_only_rows=sum(getattr(g, 'value_only_rows', 0) for g in games),
                   play_seconds=played, seconds=time.time() - t0, search_stats=dict(engine.stats))
        if args.eval_every and iteration % args.eval_every == 0:
            t1 = time.time()
            engine.configure(args.cpuct, args.fpu, **search_settings(args, args.eval_search))
            row['eval'] = match(engine, model, device, args.eval_pairs, args.eval_simulations)
            row['eval']['search'] = args.eval_search
            engine.configure(args.cpuct, args.fpu, **search_settings(args))
            row['eval_seconds'] = time.time() - t1
        metrics.append(row)
        print(json.dumps(row), flush=True)
        games = carried
        def crossed(minutes):
            period = 60 * minutes
            return bool(period) and int(spent() // period) > int((spent() - row['seconds']) // period)
        hourly, latest = crossed(args.snapshot_minutes), args.snapshot_state and crossed(args.latest_minutes)
        if args.snapshot_dir and (hourly or latest):
            # Written before state.pt: a crash in between replays this iteration and rewrites them.
            snapshots = Path(args.snapshot_dir)
            snapshots.mkdir(parents=True, exist_ok=True)
            name = f'h{int(spent() // 3600):03d}-it{iteration:05d}'
            if args.snapshot_state:
                # state-latest is never older than the newest kept state, so a new VM can always resume from it.
                save_rl_state(snapshots / 'state-latest.pt.gz', model, optimizer, replay, iteration, metrics, args,
                              rng, nprng, games=games, engine=engine, scheduler=scheduler)
            if hourly:
                totals = {key: sum(r.get(key, 0) for r in metrics) for key in ('games', 'positions', 'steps', 'policy_rows')}
                save_model(model, snapshots / f'model-{name}.pt', iteration=iteration, spent_seconds=spent(), totals=totals)
                previous = sorted(p for p in snapshots.glob('model-h*.pt') if p.stem < f'model-{name}')
                if args.prev_eval_pairs and previous:
                    # Head-to-head against the previous kept model, once negamax evals saturate.
                    t1 = time.time()
                    opponent, _ = load_model(previous[-1], device)
                    engine.configure(args.cpuct, args.fpu, **search_settings(args, args.eval_search))
                    row['prev_eval'] = head_to_head(engine, model, opponent, device, args.prev_eval_pairs,
                                                    args.eval_simulations)
                    engine.configure(args.cpuct, args.fpu, **search_settings(args))
                    row['prev_eval'].update(opponent=previous[-1].stem, search=args.eval_search,
                                            seconds=time.time() - t1)
                    print('prev_eval', json.dumps(row['prev_eval']), flush=True)
                if args.snapshot_state:
                    temporary = snapshots / f'state-{name}.pt.gz.{os.getpid()}.tmp'
                    shutil.copyfile(snapshots / 'state-latest.pt.gz', temporary)
                    os.replace(temporary, snapshots / f'state-{name}.pt.gz')
            (snapshots / 'metrics.json').write_text(json.dumps(metrics, indent=1))
            print(f'snapshot {name}' + ('' if hourly else ' (latest only)'), flush=True)
        save_model(model, out / 'model.pt', iteration=iteration)
        save_rl_state(state_path, model, optimizer, replay, iteration, metrics, args, rng, nprng,
                      games=games, engine=engine, scheduler=scheduler)
        last_save[0] = time.monotonic()
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))


def cmd_eval(args):
    device = pick_device(args.device)
    engine = Engine(args.workers, args.cpuct, args.fpu, **search_settings(args))
    model, _ = load_model(args.model, device)
    results = {}
    for suite in args.suites:
        t = time.time()
        results[suite] = match(engine, model, device, args.pairs, args.simulations, args.nodes, suite, args.seed,
                               args.max_plies)
        results[suite]['seconds'] = time.time() - t
        print(suite, json.dumps(results[suite]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(args=settings(args), results=results), indent=1))


def cmd_h2h(args):
    """Round robin between checkpoints (named by --names, else by file name)."""
    device = pick_device(args.device)
    engine = Engine(args.workers, args.cpuct, args.fpu, **search_settings(args))
    names = args.names or [Path(m).stem for m in args.models]
    models = [load_model(m, device)[0] for m in args.models]
    results = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            t = time.time()
            result = head_to_head(engine, models[i], models[j], device, args.pairs, args.simulations, args.suite,
                                  args.seed, args.max_plies)
            result['seconds'] = time.time() - t
            results[f'{names[i]} vs {names[j]}'] = result
            print(f'{names[i]} vs {names[j]}', json.dumps(result), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(args=settings(args), results=results), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--workers', type=int, default=None)
        p.add_argument('--device', default='auto')
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--max-plies', type=int, default=100)

    def search(p):
        p.add_argument('--search', choices=('puct', 'gumbel'), default='puct')
        p.add_argument('--gumbel-interior', choices=('puct', 'completed-q'), default='puct',
                       help='Below the Gumbel root: PUCT, or the full algorithm\'s completed-Q selector')
        p.add_argument('--gumbel-candidates', type=int, default=16, help='Root actions considered (m)')
        p.add_argument('--gumbel-scale', type=float, default=1.0, help='Gumbel noise scale for noisy (training) searches')
        p.add_argument('--gumbel-value-scale', type=float, default=0.1, help='c_scale of the Q transform')
        p.add_argument('--gumbel-maxvisit-init', type=float, default=50.0, help='c_visit of the Q transform')
        p.add_argument('--gumbel-rescale-values', action=argparse.BooleanOptionalAction, default=True,
                       help='Rescale completed Q-values to [0, 1] (mctx default)')
        p.add_argument('--race-solver', choices=('legacy', 'exact', 'tablebase'), default='legacy',
                       help='Zero-wall bounded proof minimax; unknown proofs continue through MCTS')
        p.add_argument('--race-depth', type=int, default=100, help='Maximum zero-wall proof plies')
        p.add_argument('--race-nodes', type=int, default=2048, help='Node cap per zero-wall proof query')

    p = sub.add_parser('gen')
    common(p)
    p.add_argument('--out', required=True)
    p.add_argument('--games', type=int, default=20000)
    p.add_argument('--parallel', type=int, default=512)
    p.add_argument('--budgets', default='500,1000,2000,4000')
    p.add_argument('--epsilon', type=float, default=0.05)
    p.add_argument('--max-opening', type=int, default=10)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser('pretrain')
    common(p)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--epochs', type=float, default=4)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--eval-pairs', type=int, default=20)
    p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--drop-planes', type=int, nargs='*', default=[],
                   help='Ablation: zero these input planes (path features are 4-11 and 14)')
    p.set_defaults(fn=cmd_pretrain)

    p = sub.add_parser('rl')
    common(p)
    p.add_argument('--out', required=True)
    p.add_argument('--init', help='Pretrained model.pt; omit to start from random weights')
    p.add_argument('--mix-data', help='Supervised .npz to seed a quarter of the replay buffer')
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--games', type=int, default=1024)
    p.add_argument('--negamax-fraction', type=float, default=0.5)
    p.add_argument('--negamax-nodes', type=int, nargs='+', default=[2000])
    p.add_argument('--simulations', type=int, default=200)
    p.add_argument('--temperature-plies', type=int, default=8)
    p.add_argument('--max-opening', type=int, default=8)
    p.add_argument('--cpuct', type=float, default=1.5)
    p.add_argument('--fpu', type=float, default=0.25)
    p.add_argument('--replay', type=int, default=400000)
    p.add_argument('--reuse', type=float, default=4.0, help='Average times each new position is sampled')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw',
                   help='RL optimizer; Muon uses the shared trunk matrices and AdamW for heads/vectors')
    p.add_argument('--schedule', choices=('none', 'cosine', 'cosine-restarts'), default='none',
                   help='Optional RL learning-rate schedule, stepped once per optimizer update')
    p.add_argument('--schedule-steps', type=int, default=25000,
                   help='Optimizer-update horizon for cosine schedules')
    p.add_argument('--playout-cap', choices=('fixed', 'randomized-40-400'), default='fixed',
                   help='Randomized ablation draws 40 sims (75%%) or 400 (25%%); low-cap rows train value only')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-pairs', type=int, default=20)
    p.add_argument('--eval-simulations', type=int, default=200)
    p.add_argument('--save-every', type=int, default=5, help='Legacy option; state now saves every iteration')
    p.add_argument('--checkpoint-seconds', type=float, default=60, help='In-flight move-boundary checkpoint cadence')
    p.add_argument('--budget-minutes', type=float, default=0,
                   help='Also stop once self-play and training (not evaluation) have taken this long')
    p.add_argument('--snapshot-dir', help='Keep a copy of the model here every --snapshot-minutes')
    p.add_argument('--snapshot-minutes', type=float, default=60,
                   help='Self-play and training minutes (evaluations excluded) between snapshots')
    p.add_argument('--snapshot-state', action='store_true',
                   help='Snapshots also keep the full gzipped resumable state (replay, optimizer, games, trees)')
    p.add_argument('--latest-minutes', type=float, default=20,
                   help='With --snapshot-state, also refresh state-latest.pt.gz this often (overwritten, not kept)')
    p.add_argument('--prev-eval-pairs', type=int, default=0,
                   help='At each kept snapshot, paired games against the previous kept model (heldout openings)')
    p.add_argument('--resume-from', help='Archived state(.pt.gz) to start --out from when it has no state.pt')
    search(p)
    p.add_argument('--eval-search', choices=('puct', 'gumbel'), default='puct', help='Search for in-run evaluations')
    p.set_defaults(fn=cmd_rl)

    p = sub.add_parser('eval')
    common(p)
    p.add_argument('model')
    p.add_argument('--pairs', type=int, default=20)
    p.add_argument('--simulations', type=int, default=200)
    p.add_argument('--nodes', type=int, default=2000)
    p.add_argument('--suites', nargs='+', default=['monitor', 'heldout'])
    p.add_argument('--cpuct', type=float, default=1.5)
    p.add_argument('--fpu', type=float, default=0.25)
    p.add_argument('--out')
    search(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser('h2h')
    common(p)
    p.add_argument('models', nargs='+')
    p.add_argument('--names', nargs='+')
    p.add_argument('--pairs', type=int, default=20)
    p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--suite', default='heldout')
    p.add_argument('--cpuct', type=float, default=1.5)
    p.add_argument('--fpu', type=float, default=0.25)
    p.add_argument('--out')
    search(p)
    p.set_defaults(fn=cmd_h2h)

    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command in ('gen', 'pretrain', 'rl'):
        lock_path = Path(str(args.out) + '.lock') if args.command == 'gen' else Path(args.out) / 'run.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Another process is writing {args.out}') from None
            args.fn(args)
    else:
        args.fn(args)


if __name__ == '__main__':
    main()
