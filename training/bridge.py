"""CPU engine processes sharing one deterministic, centrally evaluated leaf wave."""
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


class EngineProcess:
    def __init__(self, engine):
        self.process = subprocess.Popen(
            ['node', '--import', 'tsx', 'training/bridge.ts'], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True,
            env={**os.environ, 'QUORIDOR_ENGINE': engine})

    def ask(self, request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Game engine worker exited unexpectedly')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


class Bridge:
    def __init__(self, engine='typescript', workers=1):
        if workers < 1:
            raise ValueError('workers must be positive')
        self.stats = {'requests': 0, 'bridge_seconds': 0.0, 'inference_seconds': 0.0,
                      'inference_batches': 0, 'inference_positions': 0, 'reused_visits': 0}
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.owners = {}
        self.next_owner = 0
        self.order = {}
        self.search_results = {}
        try:
            for _ in range(workers):
                self.workers.append(EngineProcess(engine))
        except BaseException:
            self.close()
            raise

    def _owner(self, identifier):
        if identifier not in self.owners:
            self.owners[identifier] = self.next_owner % len(self.workers)
            self.next_owner += 1
        return self.owners[identifier]

    def _exchange(self, requests):
        """At most one outstanding request per process; drain all on errors."""
        if not self.executor:
            return {i: self.workers[i].ask(req) for i, req in requests.items()}
        pending = {i: self.executor.submit(self.workers[i].ask, req) for i, req in requests.items()}
        if pending:
            wait(pending.values())
        return {i: future.result() for i, future in pending.items()}

    def _partition(self, rows, identifier):
        groups = {}
        for row in rows:
            groups.setdefault(self._owner(identifier(row)), []).append(row)
        return groups

    def _dispatch(self, request):
        op = request.get('op')
        if len(self.workers) == 1:
            return self.workers[0].ask(request)
        if op in ('mcts_begin', 'mcts_step'):
            field = 'requests' if op == 'mcts_begin' else 'evaluations'
            rows = request[field]
            if op == 'mcts_begin':
                self.order = {row['id']: i for i, row in enumerate(rows)}
                if len(self.order) != len(rows):
                    raise ValueError('Duplicate search id')
                self.search_results = {}
            groups = self._partition(rows, lambda row: row['id'])
            self.search_results.update(self._exchange({i: {'op': op, field: group} for i, group in groups.items()}))
            # Completed workers retain their policies while other workers finish.
            results = list(self.search_results.values())
            done = all(result['done'] for result in results)
            leaves = sorted((leaf for result in results for leaf in result['leaves']),
                            key=lambda leaf: self.order[leaf['id']])
            policies = {identifier: policy for result in results for identifier, policy in result['policies'].items()}
            return {'leaves': leaves, 'done': done,
                    'policies': {identifier: policies[identifier] for identifier in self.order} if done else {},
                    'reused': sum(result['reused'] for result in results)}
        if op == 'advance':
            groups = self._partition(request['moves'], lambda row: row['id'])
            results = self._exchange({i: {'op': op, 'moves': rows} for i, rows in groups.items()})
            states = {row['id']: state for i, rows in groups.items()
                      for row, state in zip(rows, results[i]['states'])}
            return {'states': [states[row['id']] for row in request['moves']]}
        if op == 'negamax_batch':
            groups = {}
            for index, position in enumerate(request['positions']):
                groups.setdefault(index % len(self.workers), []).append((index, position))
            results = self._exchange({i: {**request, 'positions': [p for _, p in rows]} for i, rows in groups.items()})
            ordered = [None] * len(request['positions'])
            for i, rows in groups.items():
                for (index, _), result in zip(rows, results[i]['results']):
                    ordered[index] = result
            return {'results': ordered}
        if op == 'snapshot':
            results = self._exchange({i: request for i in range(len(self.workers))})
            trees = {identifier: tree for result in results.values() for identifier, tree in result['trees']}
            return {'trees': [[identifier, trees[identifier]] for identifier in self.owners if identifier in trees]}
        if op in ('clear', 'restore'):
            self._exchange({i: {'op': 'clear'} for i in range(len(self.workers))})
            self.owners.clear()
            self.next_owner = 0
            self.search_results = {}
            self.order = {}
            if op == 'restore':
                groups = self._partition(request['snapshot'], lambda row: row[0])
                self._exchange({i: {'op': op, 'snapshot': rows} for i, rows in groups.items()})
            return {}
        if op == 'drop':
            groups = self._partition(request['ids'], lambda identifier: identifier)
            self._exchange({i: {'op': op, 'ids': rows} for i, rows in groups.items()})
            for identifier in request['ids']:
                self.owners.pop(identifier, None)
            return {}
        # Initial positions, openings and single-game bot requests are small.
        return self.workers[0].ask(request)

    def ask(self, **request):
        start = time.perf_counter()
        result = self._dispatch(request)
        self.stats['requests'] += 1
        elapsed = time.perf_counter() - start
        self.stats['bridge_seconds'] += elapsed
        category = 'negamax_seconds' if request.get('op', '').startswith('negamax') else (
            'search_bridge_seconds' if request.get('op', '').startswith('mcts_') else 'other_bridge_seconds')
        self.stats[category] = self.stats.get(category, 0) + elapsed
        return result

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        for worker in self.workers:
            worker.close()


def create_bridge(engine='typescript', workers=1, mps_inference='bucketed'):
    if engine == 'direct':
        from direct import DirectBridge
        return DirectBridge(workers=workers, mps_inference=mps_inference)
    return Bridge(engine, workers=workers)
