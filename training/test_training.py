from pathlib import Path
import random
import tempfile
import unittest

import torch

from core import Bridge, PolicyValue, choose, encode, encode_batch, key, search, search_batch, update
from train import advance, atomic_save, finish_game, migrate


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bridge = Bridge()

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()

    def test_batch_encoding_matches_scalar_reference(self):
        rows, expected = [], []
        rng = random.Random(23)
        for i in range(32):
            position = {'pawns': rng.sample(range(81), 2), 'wallsLeft': [i % 11, (i + 3) % 11],
                        'hWalls': rng.sample(range(64), i % 11),
                        'vWalls': rng.sample(range(64), (i + 5) % 11), 'turn': i % 2}
            repetitions, remaining = i % 4 + 1, (100 - i) / 100
            planes = torch.zeros(9, 9, 9)
            for player, cell in enumerate(position['pawns']):
                planes[player, cell // 9, cell % 9] = 1
                planes[4 + player].fill_(position['wallsLeft'][player] / 10)
            for channel, name in [(2, 'hWalls'), (3, 'vWalls')]:
                for anchor in position[name]:
                    planes[channel, anchor // 8, anchor % 8] = 1
            planes[6].fill_(position['turn'])
            planes[7].fill_(min(repetitions, 3) / 3)
            planes[8].fill_(remaining)
            rows.append((position, repetitions, remaining))
            expected.append(planes)
        actual = encode_batch(rows)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual, torch.stack(expected)))
        self.assertTrue(torch.equal(encode(*rows[0]), expected[0]))

    def test_batches_neural_positions_in_one_forward_pass(self):
        class RecordingModel(PolicyValue):
            def __init__(self):
                super().__init__()
                self.batches = []

            def forward(self, features):
                self.batches.append(len(features))
                return super().forward(features)

        self.bridge.ask(op='clear')
        model = RecordingModel()
        state = self.bridge.ask()
        games = [{'id': str(i), 'state': state, 'history': {key(state['position']): 1}, 'ply': 0} for i in range(4)]
        policies = search_batch(model, self.bridge, games, 100, 4, random.Random(1), 'cpu')
        self.assertTrue(all(batch == 4 for batch in model.batches))
        self.assertEqual(len(policies), 4)
        for policy in policies.values():
            self.assertAlmostEqual(policy.sum().item(), 1)
        self.bridge.ask(op='clear')

    def test_migrates_legacy_in_progress_checkpoint(self):
        game = {'ply': 3, 'samples': [], 'moves': [67, 13, 58]}
        state = {'version': 1, 'mode': 'selfplay', 'phase': 'train', 'iteration': 2,
                 'game_index': 1, 'game': game, 'config': {}}
        migrate(state)
        self.assertEqual(state['version'], 2)
        self.assertIs(state['active_games'][0], game)
        self.assertEqual(game['id'], 'selfplay:train:2:1')
        self.assertEqual(state['config']['parallel_games'], 1)
        self.assertEqual(state['config']['checkpoint_seconds'], 15)
        self.assertEqual(state['config']['negamax_value_turns'], 'learner')
        self.assertEqual(game['moves'], [67, 13, 58])

    def test_rules_and_illegal_move(self):
        state = self.bridge.ask()
        self.assertEqual(len(state['legal']), 131)
        with self.assertRaisesRegex(RuntimeError, 'Illegal move'):
            self.bridge.ask(position=state['position'], move=0)
        self.assertIn(self.bridge.ask(op='negamax', position=state['position'], nodes=100)['move'], state['legal'])

    def test_mcts_value_sign_takes_win_for_either_player(self):
        model = PolicyValue()
        for param in model.parameters():
            param.data.zero_()
        for turn, pawns, win in [(0, [9, 40], 0), (1, [40, 63], 72)]:
            position = {'pawns': pawns, 'wallsLeft': [0, 0], 'hWalls': [], 'vWalls': [], 'turn': turn}
            state = self.bridge.ask(position=position)
            policy = search(model, self.bridge, state, {key(position): 1}, 0, 100, 32, random.Random(1), 'cpu')
            self.assertEqual(choose(policy, random.Random(1), False), win)
            self.assertAlmostEqual(policy.sum().item(), 1)
            self.assertTrue(all(policy[m] == 0 for m in range(209) if m not in state['legal']))

    def test_repetition_and_move_limit_are_draws(self):
        model = PolicyValue()
        position = {'pawns': [76, 4], 'wallsLeft': [0, 0], 'hWalls': [], 'vWalls': [], 'turn': 0}
        state = self.bridge.ask(position=position)
        # All children trigger a third repetition. Values cannot favor a losing/winning leaf.
        history = {key(position): 1}
        for move in state['legal']:
            child = self.bridge.ask(position=position, move=move)
            history[key(child['position'])] = 2
        policy = search(model, self.bridge, state, history, 0, 100, 8, random.Random(3), 'cpu')
        self.assertAlmostEqual(policy.sum().item(), 1)
        with self.assertRaisesRegex(ValueError, 'finished'):
            search(model, self.bridge, state, {key(position): 3}, 0, 100, 8, random.Random(3), 'cpu')
        with self.assertRaisesRegex(ValueError, 'finished'):
            search(model, self.bridge, state, {key(position): 1}, 100, 100, 8, random.Random(3), 'cpu')

    def test_checkpoint_restores_exact_next_update(self):
        torch.manual_seed(12)
        model = PolicyValue()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        rng = random.Random(7)
        initial = self.bridge.ask()
        policy = torch.zeros(209)
        policy[67] = 1
        replay = [(encode(initial['position']), policy, float(i % 3 - 1)) for i in range(8)]
        update(model, optimizer, replay, 4, rng, 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'rng': rng.getstate(), 'replay': replay}, path)
            expected = update(model, optimizer, replay, 4, rng, 'cpu')
            saved = torch.load(path, weights_only=True)
            restored = PolicyValue()
            restored.load_state_dict(saved['model'])
            opt = torch.optim.AdamW(restored.parameters(), lr=0.001)
            opt.load_state_dict(saved['optimizer'])
            rng.setstate(saved['rng'])
            actual = update(restored, opt, saved['replay'], 4, rng, 'cpu')
            self.assertEqual(expected, actual)
            for a, b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_outcomes_use_sample_player_and_replay_is_bounded(self):
        features, policy = torch.zeros(9, 9, 9), torch.zeros(209)
        state = {'phase': 'train', 'mode': 'selfplay', 'iteration': 1, 'game_index': 0,
                 'records': [], 'arms': {'selfplay': {'replay': []}}, 'config': {'replay_size': 2},
                 'game': {'learner': 0, 'ply': 3, 'opening': {}, 'moves': [],
                          'samples': [(features, policy, p) for p in (0, 1, 0)]}}
        finish_game(state, 1, 'goal')
        self.assertEqual([row[2] for row in state['arms']['selfplay']['replay']], [1.0, -1.0])
        self.assertIsNone(state['game'])
        self.assertEqual(state['game_index'], 1)

    def test_opponent_rows_train_value_without_policy_gradients(self):
        model = PolicyValue()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.value.weight.detach().clone()
        loss = update(model, optimizer, [(torch.ones(9, 9, 9), None, 1.0)],
                      1, random.Random(1), 'cpu')
        self.assertEqual(loss['policy'], 0.0)
        self.assertIsNone(model.policy.weight.grad)
        self.assertFalse(torch.equal(before, model.value.weight))

    def test_mixed_batch_policy_loss_uses_only_learner_rows(self):
        model = PolicyValue()
        x = torch.ones(9, 9, 9)
        target = torch.zeros(209)
        target[67] = 1
        with torch.no_grad():
            expected = -model(x.unsqueeze(0))[0].log_softmax(1)[0, 67].item()
        loss = update(model, torch.optim.SGD(model.parameters(), lr=0),
                      [(x, target, -1.0), (x * 2, None, 1.0)], 2, random.Random(1), 'cpu')
        self.assertAlmostEqual(loss['policy'], expected, places=5)

    def test_opponent_value_targets_survive_checkpoint(self):
        state = {'phase': 'train', 'mode': 'negamax', 'iteration': 1, 'game_index': 0,
                 'records': [], 'arms': {'negamax': {'replay': []}}, 'config': {'replay_size': 10},
                 'game': {'learner': 0, 'ply': 2, 'opening': {}, 'moves': [],
                          'samples': [(torch.zeros(9, 9, 9), torch.ones(209) / 209, 0),
                                      (torch.ones(9, 9, 9), None, 1)]}}
        finish_game(state, 1, 'goal')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            atomic_save(state, path)
            replay = torch.load(path, weights_only=True)['arms']['negamax']['replay']
        self.assertEqual([r[2] for r in replay], [-1.0, 1.0])
        self.assertIsNone(replay[1][1])

    def test_schedule_evaluates_both_initial_models_then_trains_independently(self):
        state = {'phase': 'eval', 'mode': 'selfplay', 'iteration': 0, 'config': {'iterations': 1}}
        phases = []
        while state['phase'] != 'done':
            phases.append((state['mode'], state['phase'], state['iteration']))
            advance(state)
        self.assertEqual(phases, [('selfplay', 'eval', 0), ('negamax', 'eval', 0),
                                 ('selfplay', 'train', 1), ('selfplay', 'update', 1), ('selfplay', 'eval', 1),
                                 ('negamax', 'train', 1), ('negamax', 'update', 1), ('negamax', 'eval', 1)])


if __name__ == '__main__':
    unittest.main()
