"""Gumbel AlphaZero search: halving schedule, budgets, improved targets, determinism and resume."""
import copy
import math
import random
import unittest

import numpy as np
import torch

from az import MOVES, Engine, Game, Net, START, head_to_head, match, play, turn, winner


def reference_schedule(considered, simulations):
    """mctx seq_halving.get_sequence_of_considered_visits (google-deepmind/mctx@428bfb7, Apache 2.0)."""
    if considered <= 1:
        return tuple(range(simulations))
    log2max = int(math.ceil(math.log2(considered)))
    sequence, visits, n = [], [0] * considered, considered
    while len(sequence) < simulations:
        extra = max(1, int(simulations / (log2max * n)))
        for _ in range(extra):
            sequence.extend(visits[:n])
            for i in range(n):
                visits[i] += 1
        n = max(2, n // 2)
    return tuple(sequence[:simulations])


GUMBEL = dict(search='gumbel', gumbel_candidates=16)


class GumbelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5)
        cls.model = Net(19, 8, 1).eval()

    def engine(self, **search):
        return Engine(2, **(GUMBEL | search))

    def positions(self, engine, n=6, seed=3):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            position = START
            for _ in range(rng.randint(0, 30)):
                child = engine.advance(position, rng.choice(engine.legal(position)))
                if winner(child) != -1:
                    break
                position = child
            out.append(position)
        return out

    def search(self, engine, positions, simulations, seeds=None, first_tree=0):
        jobs = [dict(tree=first_tree + i, position=p, history={p: 1}, ply=0, max_plies=100)
                | ({} if seeds is None else dict(seed=seeds[i])) for i, p in enumerate(positions)]
        return engine.search(self.model, jobs, simulations, 'cpu', moves=True)

    def root_visits(self, engine, tree):
        out = np.empty(MOVES, np.int32)
        root = engine.check(engine.lib.az_root_visits(engine.pool, tree, out.ctypes.data))
        return out, root

    def test_schedule_matches_reference(self):
        e = Engine(1)
        for m in (0, 1, 2, 3, 5, 7, 16, 17, 32, 131):
            for b in (1, 2, 7, 16, 63, 64, 100, 200, 800):
                out = np.empty(b, np.int32)
                e.check(e.lib.az_considered_visits(m, b, out.ctypes.data))
                self.assertEqual(tuple(out.tolist()), reference_schedule(m, b), (m, b))

    def test_exact_budgets_and_targets(self):
        e = self.engine()
        positions = self.positions(e)
        for budget in (1, 2, 7, 16, 63, 200):
            e.clear()
            policies, _, moves = self.search(e, positions, budget, seeds=list(range(len(positions))))
            for i, p in enumerate(positions):
                legal = e.legal(p)
                visits, root = self.root_visits(e, i)
                self.assertEqual(visits.sum(), budget)
                self.assertEqual(root, budget + 1)  # The root's own expansion is outside the budget.
                self.assertIn(int(moves[i]), legal)
                self.assertEqual(visits[moves[i]], visits.max())
                np.testing.assert_allclose(policies[i].sum(), 1, rtol=1e-5)
                # Every legal move, visited or not, keeps target mass; illegal moves have none.
                self.assertTrue((policies[i][legal] > 0).all())
                self.assertEqual(np.count_nonzero(policies[i]), len(legal))
                # The schedule never visits more than min(m, legal) distinct moves.
                self.assertLessEqual(np.count_nonzero(visits), min(16, len(legal), budget))

    def test_one_legal_action(self):
        # White in the bottom-left corner, no walls left, a vertical wall to its right: only "up".
        e = self.engine()
        p = (0, 1 << 56, 72 | (4 << 7))
        self.assertEqual(e.legal(p), [63])
        policies, _, moves = self.search(e, [p], 7, seeds=[1])
        self.assertEqual(int(moves[0]), 63)
        self.assertEqual(float(policies[0][63]), 1)
        self.assertEqual(self.root_visits(e, 0)[0].sum(), 7)

    def test_finds_immediate_win(self):
        # White steps to row 0 now, or black (one step from its goal) wins next.
        position = (0, 0, 13 | (67 << 7))
        for interior in ('puct', 'completed-q'):
            e = self.engine(gumbel_interior=interior)
            _, _, moves = self.search(e, [position], 16)
            self.assertEqual(int(moves[0]), 4)

    def test_noise_changes_choices_and_is_reproducible(self):
        e = self.engine()
        positions = self.positions(e, 8)
        a = self.search(e, positions, 32, seeds=list(range(8)))
        e.clear()
        b = self.search(e, positions, 32, seeds=list(range(8)))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        e.clear()
        c = self.search(e, positions, 32, seeds=list(range(100, 108)))
        self.assertFalse(np.array_equal(a[0], c[0]))

    def test_cache_does_not_change_decisions(self):
        positions = self.positions(Engine(1), 6)
        results = []
        for limit in (0, 1 << 16):
            e = self.engine()
            e.cache_limit = limit
            results.append(self.search(e, positions * 2, 24, seeds=list(range(12))))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(results[0][2], results[1][2])

    def test_worker_counts_do_not_change_games(self):
        out = []
        for workers in (1, 3):
            e = Engine(workers, **GUMBEL, gumbel_interior='completed-q')
            games = [Game(i, START, {START: 1}, ['mcts', 'mcts']) for i in range(4)]
            play(e, self.model, games, 'cpu', 12, max_plies=10, noise=True, rng=random.Random(3))
            out.append([(g.moves, [s[3] for s in g.samples]) for g in games])
        for (ma, sa), (mb, sb) in zip(*out):
            self.assertEqual(ma, mb)
            for x, y in zip(sa, sb):
                np.testing.assert_array_equal(x, y)

    def test_targets_are_canonical_and_legal(self):
        e = self.engine()
        games = [Game(i, START, {START: 1}, ['mcts', 'mcts']) for i in range(3)]
        play(e, self.model, games, 'cpu', 8, max_plies=12, noise=True, rng=random.Random(1))
        for g in games:
            for position, _, _, policy, t in g.samples:
                legal = e.legal(position)
                self.assertEqual(sorted(np.flatnonzero(policy).tolist()), sorted(e.canonical[t, 0, legal].tolist()))
                self.assertEqual(t, turn(position))

    def test_resume_at_move_boundary(self):
        e = self.engine()
        games = [Game(i, START, {START: 1}, ['mcts', 'mcts']) for i in range(4)]
        rng = random.Random(4)
        saved = []

        class Paused(Exception):
            pass

        def pause(done, total):
            if games[0].ply == 3:
                saved.extend([copy.deepcopy(games), e.snapshot(), rng.getstate()])
                raise Paused()
        with self.assertRaises(Paused):
            play(e, self.model, games, 'cpu', 8, max_plies=8, noise=True, rng=rng, progress=pause)
        play(e, self.model, games, 'cpu', 8, max_plies=8, noise=True, rng=rng, resume=True)
        fresh = self.engine()
        fresh.restore(saved[1])
        resumed = random.Random()
        resumed.setstate(saved[2])
        play(fresh, self.model, saved[0], 'cpu', 8, max_plies=8, noise=True, rng=resumed, resume=True)
        for a, b in zip(games, saved[0]):
            self.assertEqual((a.moves, a.result), (b.moves, b.result))
            for x, y in zip(a.samples, b.samples):
                np.testing.assert_array_equal(x[3], y[3])

    def test_matches_and_negamax_replies(self):
        e = self.engine(gumbel_interior='completed-q')
        self.assertEqual(match(e, self.model, 'cpu', 1, 8, nodes=50, max_plies=20)['games'], 2)
        other = Net(19, 8, 1).eval()
        self.assertEqual(head_to_head(e, self.model, other, 'cpu', 1, 8, max_plies=20)['games'], 2)

    def test_puct_unchanged_interface(self):
        e = Engine(2)
        policy, values, moves = self.search(e, [START], 16)
        self.assertEqual(int(moves[0]), -1)
        self.assertAlmostEqual(float(policy[0].sum()), 1, places=5)


if __name__ == '__main__':
    unittest.main()
