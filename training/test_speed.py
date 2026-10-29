"""Scheduling and budget regressions, including low-budget target semantics."""
import random
import unittest
from unittest.mock import patch

import torch
from core import PolicyValue, key, search_batch, update
from direct import DirectBridge
from train import fill_games, search_budgets


class SpeedTests(unittest.TestCase):
    def test_refill_never_duplicates_or_skips_games(self):
        state = dict(config=dict(parallel_games=3, game_scheduler='rolling'),
                     game_index=0, active_games=[])
        with patch('train.fresh_game', side_effect=lambda s, t, i: dict(index=i)):
            fill_games(state, None, 7)
            self.assertEqual([g['index'] for g in state['active_games']], [0, 1, 2])
            state['active_games'].pop(1)  # Game 1 finishes ahead of game 0.
            state['game_index'] += 1
            fill_games(state, None, 7)
            self.assertEqual([g['index'] for g in state['active_games']], [0, 2, 3])
            state['active_games'] = [dict(index=0)]
            state['game_index'] = 5
            fill_games(state, None, 7)
            self.assertEqual([g['index'] for g in state['active_games']], [0, 6])
            state['config']['game_scheduler'] = 'cohort'
            state['active_games'] = [dict(index=0)]
            fill_games(state, None, 7)
            self.assertEqual(state['active_games'], [dict(index=0)])

    def test_budgets_are_reproducible_and_eval_is_full(self):
        state = dict(config=dict(seed=1, simulations=64, fast_simulations=16,
                     full_search_probability=.25, search_budget='randomized'),
                     phase='train', mode='selfplay', iteration=3)
        games = [dict(id=str(i), index=i, ply=5) for i in range(1000)]
        budgets = search_budgets(state, games)
        self.assertEqual(budgets, search_budgets(state, list(reversed(games))))
        self.assertTrue(200 < sum(b == 64 for b in budgets.values()) < 300)
        state['phase'] = 'eval'
        self.assertEqual(set(search_budgets(state, games).values()), {64})
        state['phase'] = 'train'
        state['config']['search_budget'] = 'fixed'
        self.assertEqual(set(search_budgets(state, games).values()), {64})

    def test_unequal_budgets_match_separate_searches(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = PolicyValue()
        bridge = DirectBridge(workers=2)
        try:
            state = bridge.ask()
            games = [dict(id=str(i), state=state, history={key(state['position']): 1}, ply=0) for i in range(2)]
            actual = search_batch(model, bridge, games, 100, {'0': 4, '1': 16}, random.Random(1), 'cpu')
            bridge.ask(op='clear')
            for game, budget in zip(games, [4, 16]):
                expected = search_batch(model, bridge, [game], 100, budget, random.Random(1), 'cpu')
                torch.testing.assert_close(actual[game['id']], expected[game['id']], rtol=0, atol=0)
        finally:
            bridge.close()

    def test_all_shallow_batch_trains_value_without_policy(self):
        torch.set_num_threads(1)
        model = PolicyValue()
        before = model.policy.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        loss = update(model, optimizer, [(torch.zeros(9, 9, 9), None, 1.)], 1, random.Random(1), 'cpu')
        self.assertEqual(loss['policy'], 0)
        self.assertTrue(torch.equal(before, model.policy.weight))
        self.assertGreater(loss['value'], 0)
