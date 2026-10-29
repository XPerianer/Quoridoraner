import random
import unittest
from collections import Counter

from replay import retain, sample, task


class ReplayTests(unittest.TestCase):
    def test_long_games_cannot_evict_short_tasks(self):
        replay = [(i, None, 1, 'race-0') for i in range(10)]
        replay += [(i, None, -1, 'standard') for i in range(1000)]
        retain(replay, 20, True)
        self.assertEqual(Counter(map(task, replay)), {'race-0': 10, 'standard': 10})
        self.assertEqual([r[0] for r in replay if task(r) == 'standard'], list(range(990, 1000)))

    def test_minibatch_balances_tasks_even_when_sparse(self):
        replay = [(0, None, 1, 'race-0')] + [(i, None, -1, 'standard') for i in range(1000)]
        batch = sample(replay, 64, random.Random(1), True)
        self.assertEqual(Counter(map(task, batch)), {'race-0': 32, 'standard': 32})
        self.assertEqual(batch, sample(replay, 64, random.Random(1), True))

    def test_capacity_and_batch_remainders_and_legacy(self):
        replay = [(i, None, 0, str(t)) for t in range(5) for i in range(10)]
        retain(replay, 17, True)
        self.assertEqual(len(replay), 17)
        counts = Counter(map(task, sample(replay, 13, random.Random(2), True)))
        self.assertEqual(sorted(counts.values()), [2, 2, 3, 3, 3])
        legacy = [(i, None, 0) for i in range(10)]
        retain(legacy, 5)
        self.assertEqual([r[0] for r in legacy], list(range(5, 10)))
        self.assertEqual(sample(legacy, 3, random.Random(1)), random.Random(1).sample(legacy, 3))
