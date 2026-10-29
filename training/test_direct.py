import json
import random
import unittest

import torch

from core import Bridge, PolicyValue, encode, key, search_batch
from direct import DirectBridge, pack, unpack
import test_training


class DirectTrainingTests(test_training.TrainingTests):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bridge = DirectBridge(workers=4)


class DirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reachable_rules_negamax_and_tree_portability(self):
        reference, native = Bridge(), DirectBridge(workers=3)
        try:
            state = reference.ask()
            rng = random.Random(93)
            positions = []
            for _ in range(100):
                self.assertEqual(native.ask(position=state['position']), state)
                self.assertEqual(unpack(pack(state['position'])), state['position'])
                positions.append(state['position'])
                if state['winner'] != -1:
                    state = reference.ask()
                    continue
                move = rng.choice(state['legal'])
                actual = native.ask(position=state['position'], move=move)
                state = reference.ask(position=state['position'], move=move)
                self.assertEqual(actual, state)
            positions = [p for p in positions[::10] if p['pawns'][0] >= 9 and p['pawns'][1] < 72]
            self.assertEqual(native.ask(op='negamax_batch', positions=positions, nodes=1000),
                             reference.ask(op='negamax_batch', positions=positions, nodes=1000))
            torch.manual_seed(12)
            model = PolicyValue()
            state = reference.ask()
            games = [{'id': str(i), 'state': state, 'history': {key(state['position']): 1}, 'ply': 0} for i in range(7)]
            expected = search_batch(model, reference, games, 100, 32, random.Random(1), 'cpu', True)
            actual = search_batch(model, native, games, 100, 32, random.Random(1), 'cpu', True)
            for identifier in expected:
                torch.testing.assert_close(expected[identifier], actual[identifier], rtol=0, atol=0)
            snapshot = reference.ask(op='snapshot')['trees']
            native.ask(op='restore', snapshot=snapshot)
            self.assertEqual(native.ask(op='snapshot')['trees'], snapshot)
            moves = [{'id': g['id'], 'position': state['position'], 'move': int(expected[g['id']].argmax())} for g in games]
            states = reference.ask(op='advance', moves=moves)['states']
            self.assertEqual(native.ask(op='advance', moves=moves)['states'], states)
            for game, next_state in zip(games, states):
                game['state'] = next_state
                game['ply'] = 1
                game['history'][key(next_state['position'])] = 1
            expected = search_batch(model, reference, games, 100, 16, random.Random(3), 'cpu')
            actual = search_batch(model, native, games, 100, 16, random.Random(3), 'cpu')
            for identifier in expected:
                torch.testing.assert_close(expected[identifier], actual[identifier], rtol=0, atol=0)
            snapshot = native.ask(op='snapshot')['trees']
            reference.ask(op='restore', snapshot=snapshot)
            self.assertEqual(reference.ask(op='snapshot')['trees'], snapshot)
        finally:
            reference.close()
            native.close()

    def test_native_features_match_reference_planes(self):
        class Recorder(PolicyValue):
            def __init__(self):
                super().__init__()
                self.inputs = []
            def forward(self, x):
                self.inputs.append(x.clone())
                return super().forward(x)
        native = DirectBridge(workers=2)
        try:
            position = {'pawns': [43, 24], 'wallsLeft': [7, 9], 'hWalls': [0, 10, 20], 'vWalls': [30], 'turn': 1}
            state = native.ask(position=position)
            game = {'id': 'features', 'state': state, 'history': {key(position): 2}, 'ply': 17}
            model = Recorder()
            search_batch(model, native, [game], 100, 4, random.Random(3), 'cpu')
            self.assertTrue(torch.equal(model.inputs[0][0], encode(position, 2, .83)))
            native.ask(op='drop', ids=['features'])
            self.assertEqual(native.ask(op='snapshot')['trees'], [])
        finally:
            native.close()

    def test_rejects_invalid_positions_and_budgets(self):
        native = DirectBridge()
        try:
            state = native.ask()
            invalid = json.loads(json.dumps(state['position']))
            invalid['hWalls'] = [0, 1]
            with self.assertRaisesRegex(RuntimeError, 'Overlapping'):
                native.ask(position=invalid)
            game = {'id': 'x', 'state': state, 'history': {key(state['position']): 1}, 'ply': 0}
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                search_batch(PolicyValue(), native, [game], 100, 0, random.Random(1), 'cpu')
        finally:
            native.close()


@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS hardware is unavailable')
class MpsInferenceTests(unittest.TestCase):
    def test_residual_hybrid_cpu_gpu_switch_preserves_outputs(self):
        from inference import MpsInference
        torch.set_num_threads(1)
        for architecture in ('residual32', 'residual64'):
            model = PolicyValue(architecture).eval().to('mps')
            runner = MpsInference(model, 'hybrid')
            with torch.inference_mode():
                for n in (2, 8):
                    x = torch.rand(n, 9, 9, 9)
                    logits, values = model(x.to('mps'))
                    expected = torch.cat((logits, values[:, None]), 1).cpu()
                    torch.testing.assert_close(runner(x), expected, rtol=1e-4, atol=1e-5)
                self.assertEqual(runner.cpu_batches, 1)
                self.assertEqual(runner.mps_batches, 1)

    def test_bucketed_and_hybrid_outputs_and_weight_refresh(self):
        from inference import MpsInference
        torch.set_num_threads(1)
        torch.manual_seed(5)
        model = PolicyValue().eval().to('mps')
        with torch.inference_mode():
            runner = MpsInference(model, 'bucketed')
            for n in (1, 17, 33):
                x = torch.rand(n, 9, 9, 9)
                logits, values = model(x.to('mps'))
                expected = torch.cat((logits, values[:, None]), 1).cpu()
                torch.testing.assert_close(runner(x), expected, rtol=1e-4, atol=1e-5)
            hybrid = MpsInference(model, 'hybrid')
            x = torch.rand(2, 9, 9, 9)
            before = hybrid(x).clone()
            # Match an optimizer update: source parameter version changes.
            model.policy.bias.add_(1)
            after = hybrid(x)
            torch.testing.assert_close(after[:, :209], before[:, :209] + 1, rtol=1e-5, atol=1e-6)
            self.assertEqual(hybrid.cpu_batches, 2)
            hybrid(torch.rand(32, 9, 9, 9))
            self.assertEqual(hybrid.mps_batches, 1)


if __name__ == '__main__':
    unittest.main()
