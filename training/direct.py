"""In-process native search with zero-copy CPU tensors and shared MPS inference."""
import ctypes as C
from functools import lru_cache
import json
import time

import numpy as np
import torch

from bridge import ROOT
from native.build_direct import OUTPUT, build


@lru_cache(maxsize=32768)
def packed_key(key):
    return pack(json.loads(key))


def pack(position):
    a, b = position['pawns']
    x, y = position['wallsLeft']
    turn = position['turn']
    if not (0 <= a < 81 and 0 <= b < 81 and a != b and 0 <= x <= 10 and 0 <= y <= 10 and turn in (0, 1)):
        raise ValueError('Invalid position')
    masks = []
    for name in ('hWalls', 'vWalls'):
        mask = 0
        for i in position[name]:
            if not isinstance(i, int) or not 0 <= i < 64 or mask & (1 << i):
                raise ValueError('Invalid wall anchor')
            mask |= 1 << i
        masks.append(mask)
    return (*masks, a | (b << 7) | (x << 14) | (y << 18) | (turn << 22))


def unpack(key):
    h, v, p = map(int, key)
    return {'pawns': [p & 127, (p >> 7) & 127], 'wallsLeft': [(p >> 14) & 15, (p >> 18) & 15],
            'turn': (p >> 22) & 1, 'hWalls': [i for i in range(64) if h & (1 << i)],
            'vWalls': [i for i in range(64) if v & (1 << i)]}


def ptr(array):
    assert array.flags.c_contiguous
    return C.c_void_p(array.ctypes.data)


def library():
    if not OUTPUT.exists() or any((ROOT / 'training/native' / name).stat().st_mtime > OUTPUT.stat().st_mtime
                                  for name in ('direct.cpp', 'engine.hpp', 'build_direct.py')):
        build()
    lib = C.CDLL(str(OUTPUT))  # ctypes releases the GIL during native search.
    pointer, integer = C.c_void_p, C.c_int
    signatures = {
        'q_create': (pointer, [integer]), 'q_destroy': (None, [pointer]), 'q_error': (C.c_char_p, []),
        'q_clear': (integer, [pointer]), 'q_drop': (integer, [pointer, integer]),
        'q_describe': (integer, [pointer, pointer]),
        'q_advance': (integer, [pointer, integer, pointer, integer, pointer]),
        'q_start': (integer, [pointer]),
        'q_begin': (integer, [pointer, integer, pointer, pointer, pointer, integer, integer, integer, integer, pointer, integer]),
        'q_wave': (integer, [pointer, pointer, pointer, pointer, pointer]),
        'q_evaluate': (integer, [pointer, pointer, pointer, pointer, integer]),
        'q_policy': (integer, [pointer, pointer, pointer]),
        'q_negamax': (integer, [pointer, pointer, integer, integer, pointer]),
        'q_root': (pointer, [pointer, integer]),
        'q_inspect': (integer, [pointer, pointer, pointer, pointer, pointer, pointer]),
        'q_restore_node': (pointer, [pointer, integer, pointer, integer, pointer, integer, C.c_double, integer, pointer, pointer, integer]),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args
    return lib


class DirectBridge:
    def __init__(self, workers=1, mps_inference="bucketed"):
        self.lib = library()
        self.pool = self.lib.q_create(workers)
        if not self.pool:
            raise RuntimeError(self.lib.q_error().decode())
        self.inference_mode = mps_inference
        self.runners = {}
        self.ids = {}
        self.next_id = 0
        self.stats = {'requests': 0, 'bridge_seconds': 0.0, 'inference_seconds': 0.0,
                      'inference_batches': 0, 'inference_positions': 0, 'reused_visits': 0}

    def check(self, result):
        if result == -1:
            raise RuntimeError(self.lib.q_error().decode())
        return result

    def identifier(self, name):
        if name not in self.ids:
            self.ids[name] = self.next_id
            self.next_id += 1
        return self.ids[name]

    def describe(self, position):
        packed = np.asarray(pack(position), dtype=np.uint64)
        legal = np.empty(136, dtype=np.int32)
        n = self.check(self.lib.q_describe(ptr(packed), ptr(legal)))
        winner = 0 if position['pawns'][0] < 9 else 1 if position['pawns'][1] >= 72 else -1
        return {'position': position, 'winner': winner, 'legal': legal[:n].tolist()}

    def advance(self, identifier, position, move):
        source = np.asarray(pack(position), dtype=np.uint64)
        result = np.empty(3, dtype=np.uint64)
        self.check(self.lib.q_advance(self.pool, identifier, ptr(source), move, ptr(result)))
        return self.describe(unpack(result))

    def snapshot_node(self, node):
        position, stats = np.empty(3, np.uint64), np.empty(3, np.float64)
        moves, priors = np.empty(136, np.int32), np.empty(136, np.float64)
        children = (C.c_void_p * 136)()
        n = self.check(self.lib.q_inspect(node, ptr(position), ptr(stats), ptr(moves), ptr(priors), children))
        board = unpack(position)
        return {'position': board, 'winner': 0 if board['pawns'][0] < 9 else 1 if board['pawns'][1] >= 72 else -1,
                'visits': int(stats[0]), 'total': float(stats[1]),
                'priors': [[int(moves[i]), float(priors[i])] for i in range(n)] if stats[2] else None,
                'children': {str(moves[i]): self.snapshot_node(children[i]) for i in range(n) if children[i]}}

    def restore_node(self, identifier, parent, move, saved):
        position = np.asarray(pack(saved['position']), np.uint64)
        pairs = saved['priors'] or []
        moves = np.asarray([p[0] for p in pairs], np.int32)
        priors = np.asarray([p[1] for p in pairs], np.float64)
        node = self.lib.q_restore_node(self.pool, identifier, parent, move, ptr(position), saved['visits'], saved['total'],
                                      int(saved['priors'] is not None), ptr(moves), ptr(priors), len(pairs))
        if not node:
            raise RuntimeError(self.lib.q_error().decode())
        for child_move, child in saved['children'].items():
            self.restore_node(identifier, node, int(child_move), child)

    def ask(self, **request):
        start = time.perf_counter()
        op = request.get('op')
        if op == 'snapshot':
            trees = []
            for name, identifier in self.ids.items():
                node = self.lib.q_root(self.pool, identifier)
                if node:
                    trees.append([name, self.snapshot_node(node)])
            result = {'trees': trees}
        elif op in ('clear', 'restore'):
            self.check(self.lib.q_clear(self.pool))
            self.ids.clear()
            self.next_id = 0
            if op == 'restore':
                for name, saved in request['snapshot']:
                    self.restore_node(self.identifier(name), None, -1, saved)
            result = {}
        elif op == 'drop':
            for name in request['ids']:
                identifier = self.ids.pop(name, None)
                if identifier is not None:
                    self.check(self.lib.q_drop(self.pool, identifier))
            result = {}
        elif op == 'advance':
            result = {'states': [self.advance(self.identifier(row['id']), row['position'], row['move']) for row in request['moves']]}
        elif op in ('negamax', 'negamax_batch'):
            positions = request['positions'] if op == 'negamax_batch' else [request['position']]
            packed = np.asarray([pack(p) for p in positions], dtype=np.uint64).reshape(-1, 3)
            output = np.empty((len(positions), 3), np.int64)
            self.check(self.lib.q_negamax(self.pool, ptr(packed), len(positions), request.get('nodes', 2000), ptr(output)))
            results = [{'move': int(row[0]), 'nodes': int(row[1]), 'score': int(row[2])} for row in output]
            result = {'results': results} if op == 'negamax_batch' else results[0]
        elif op is None:
            position = request.get('position', {'pawns': [76, 4], 'wallsLeft': [10, 10], 'hWalls': [], 'vWalls': [], 'turn': 0})
            result = self.advance(-1, position, request['move']) if 'move' in request else self.describe(position)
        else:
            raise ValueError(f'Unknown direct-engine operation: {op}')
        category = 'negamax_seconds' if (op or '').startswith('negamax') else 'other_bridge_seconds'
        elapsed = time.perf_counter() - start
        self.stats[category] = self.stats.get(category, 0) + elapsed
        self.stats['bridge_seconds'] += elapsed
        self.stats['requests'] += 1
        return result

    def search(self, model, requests, device, progress=None):
        start = time.perf_counter()
        if len({row['id'] for row in requests}) != len(requests):
            raise ValueError('Duplicate search id')
        self.check(self.lib.q_start(self.pool))
        for row in requests:
            position = np.asarray(pack(row['position']), np.uint64)
            history = np.asarray([packed_key(k) for k in row['history']], np.uint64).reshape(-1, 3)
            counts = np.asarray(list(row['history'].values()), np.int32)
            noise = np.asarray(row.get('noise', []), np.float64)
            self.check(self.lib.q_begin(self.pool, self.identifier(row['id']), ptr(position), ptr(history), ptr(counts), len(history),
                                       row['ply'], row['maxPlies'], row['simulations'], ptr(noise), len(noise)))
        capacity = len(requests)
        features = np.empty((capacity, 9, 9, 9), np.float32)
        legal = np.empty((capacity, 136), np.int32)
        counts = np.empty(capacity, np.int32)
        indices = np.empty(capacity, np.int32)
        search_seconds = time.perf_counter() - start
        wave = 0
        while True:
            start = time.perf_counter()
            n = self.check(self.lib.q_wave(self.pool, ptr(features), ptr(legal), ptr(counts), ptr(indices)))
            search_seconds += time.perf_counter() - start
            if not n:
                break
            start = time.perf_counter()
            if device == 'mps':
                if model not in self.runners:
                    from inference import MpsInference
                    self.runners[model] = MpsInference(model, self.inference_mode)
                runner = self.runners[model]
                outputs = runner(torch.from_numpy(features[:n]))
                self.stats['cpu_inference_batches'] = sum(r.cpu_batches for r in self.runners.values())
                self.stats['mps_inference_batches'] = sum(r.mps_batches for r in self.runners.values())
            else:
                logits, values = model(torch.from_numpy(features[:n]).to(device))
                outputs = torch.cat((logits, values[:, None]), dim=1).cpu()
            forwarded = time.perf_counter()
            # One batched gather and softmax replaces per-leaf Python tensor dispatch.
            selected = outputs[:, :209].gather(1, torch.from_numpy(legal[:n]).long())
            selected.masked_fill_(torch.arange(136)[None, :] >= torch.from_numpy(counts[:n])[:, None], -torch.inf)
            probabilities = selected.softmax(1).numpy()
            values = outputs[:, -1].contiguous().numpy()
            finished = time.perf_counter()
            self.stats['inference_seconds'] += finished - start
            self.stats['forward_seconds'] = self.stats.get('forward_seconds', 0) + forwarded - start
            self.stats['policy_seconds'] = self.stats.get('policy_seconds', 0) + finished - forwarded
            self.stats['inference_batches'] += 1
            self.stats['inference_positions'] += n
            start = time.perf_counter()
            self.check(self.lib.q_evaluate(self.pool, ptr(probabilities), ptr(values), ptr(indices), n))
            search_seconds += time.perf_counter() - start
            wave += 1
            if progress:
                progress(wave)
        policies = np.empty((capacity, 209), np.float32)
        reused = np.zeros(1, np.int64)
        start = time.perf_counter()
        self.check(self.lib.q_policy(self.pool, ptr(policies), ptr(reused)))
        search_seconds += time.perf_counter() - start
        self.stats['reused_visits'] += int(reused[0])
        self.stats['search_bridge_seconds'] = self.stats.get('search_bridge_seconds', 0) + search_seconds
        self.stats['bridge_seconds'] += search_seconds
        self.stats['requests'] += wave + 1
        return {row['id']: torch.from_numpy(policies[i].copy()) for i, row in enumerate(requests)}

    def close(self):
        if self.pool:
            self.lib.q_destroy(self.pool)
            self.pool = None
