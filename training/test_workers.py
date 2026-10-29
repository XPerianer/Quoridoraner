"""Exercise real engine processes, ordered inference and portable tree snapshots."""
import random
import unittest

import torch

from core import Bridge, PolicyValue, key, search_batch


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parallel_search_batches_match_and_restore_with_different_worker_count(self):
        single, parallel, restored = Bridge(), Bridge(workers=3), Bridge(workers=2)
        try:
            class RecordingModel(PolicyValue):
                def __init__(self):
                    super().__init__()
                    self.batches = []

                def forward(self, x):
                    self.batches.append(len(x))
                    return super().forward(x)

            torch.manual_seed(9)
            model = RecordingModel().eval()
            initial = single.ask()
            games = [{'id': f'game-{i}', 'state': initial,
                      'history': {key(initial['position']): 1}, 'ply': 0} for i in range(7)]
            expected = search_batch(model, single, games, 100, 12, random.Random(4), 'cpu', explore=True)
            model.batches.clear()
            actual = search_batch(model, parallel, games, 100, 12, random.Random(4), 'cpu', explore=True)
            self.assertTrue(all(n == 7 for n in model.batches))
            self.assertEqual(list(actual), list(expected))
            for identifier in actual:
                self.assertTrue(torch.equal(actual[identifier], expected[identifier]))
            self.assertEqual(len({w.process.pid for w in parallel.workers}), 3)
            snapshot = parallel.ask(op='snapshot')['trees']
            self.assertEqual(dict(snapshot), dict(single.ask(op='snapshot')['trees']))
            restored.ask(op='restore', snapshot=snapshot)
            moves = [{'id': g['id'], 'position': initial['position'], 'move': int(actual[g['id']].argmax())}
                     for g in reversed(games)]
            expected_states = single.ask(op='advance', moves=moves)
            self.assertEqual(parallel.ask(op='advance', moves=moves), expected_states)
            self.assertEqual(restored.ask(op='advance', moves=moves), expected_states)
            next_games = []
            for move, state in zip(moves, expected_states['states']):
                next_games.append({'id': move['id'], 'state': state, 'ply': 1,
                                   'history': {key(initial['position']): 1, key(state['position']): 1}})
            expected = search_batch(model, single, next_games, 100, 8, random.Random(5), 'cpu')
            actual = search_batch(model, restored, next_games, 100, 8, random.Random(5), 'cpu')
            for identifier in expected:
                self.assertTrue(torch.equal(actual[identifier], expected[identifier]))
            self.assertEqual(restored.ask(op='negamax_batch', positions=[g['state']['position'] for g in next_games], nodes=100),
                             single.ask(op='negamax_batch', positions=[g['state']['position'] for g in next_games], nodes=100))
            restored.ask(op='drop', ids=[next_games[0]['id']])
            self.assertNotIn(next_games[0]['id'], dict(restored.ask(op='snapshot')['trees']))
            restored.ask(op='clear')
            self.assertEqual(restored.ask(op='snapshot')['trees'], [])
        finally:
            single.close()
            parallel.close()
            restored.close()

    def test_finished_worker_is_retained_until_other_searches_finish(self):
        single, parallel = Bridge(), Bridge(workers=4)
        try:
            position = {'pawns': [76, 4], 'wallsLeft': [0, 0], 'hWalls': [], 'vWalls': [], 'turn': 0}
            requests = [dict(id=f'g{i}', position=position, history={key(position): 1},
                             ply=0, maxPlies=1 if i == 0 else 100, simulations=9) for i in range(3)]
            a = single.ask(op='mcts_begin', requests=requests)
            b = parallel.ask(op='mcts_begin', requests=requests)
            while True:
                self.assertEqual(a, b)
                if a['done']:
                    break
                evaluations = [{'id': leaf['id'], 'policy': [1] * len(leaf['legal']), 'value': 0.25}
                               for leaf in a['leaves']]
                a = single.ask(op='mcts_step', evaluations=evaluations)
                b = parallel.ask(op='mcts_step', evaluations=evaluations)
            self.assertEqual(dict(single.ask(op='snapshot')['trees']), dict(parallel.ask(op='snapshot')['trees']))
        finally:
            single.close()
            parallel.close()

    def test_worker_failure_surfaces_and_all_processes_close(self):
        bridge = Bridge(workers=2)
        victim = bridge.workers[1].process
        victim.kill()
        victim.wait(timeout=5)
        try:
            with self.assertRaises((RuntimeError, BrokenPipeError)):
                bridge.ask(op='clear')
        finally:
            bridge.close()
        self.assertTrue(all(w.process.poll() is not None for w in bridge.workers))


if __name__ == '__main__':
    unittest.main()
