"""Reference checks for repetition-aware bounded zero-wall proof minimax."""
import unittest

from az import Engine, Net


def position(a, b, turn=0, h=0, v=0):
    return (h, v, a | (b << 7) | (turn << 22))


def winner(p):
    if (p[2] & 127) < 9:
        return 0
    if ((p[2] >> 7) & 127) >= 72:
        return 1
    return -1


def side(p):
    return (p[2] >> 22) & 1


def reference(engine, p, history, ply, max_plies):
    """Independent Python minimax for the small cutoff positions used here."""
    w = winner(p)
    if w >= 0:
        return 1 if w == side(p) else -1
    if history.get(p, 0) >= 3 or ply >= max_plies:
        return 0
    best = -1
    for move in engine.legal(p):
        child = tuple(engine.advance(p, move))
        history[child] = history.get(child, 0) + 1
        value = -reference(engine, child, history, ply + 1, max_plies)
        history[child] -= 1
        if history[child] == 0:
            del history[child]
        best = max(best, value)
        if best == 1:
            break
    return best


class ZeroWallRaceTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(1, race_solver='exact', race_depth=100, race_nodes=100000)

    def tearDown(self):
        self.engine.close()

    def test_jump_and_small_cutoff_match_independent_minimax(self):
        p = position(40, 31)
        self.assertIn(22, self.engine.legal(p))  # straight jump over the opposing pawn
        history = {p: 1}
        expected = reference(self.engine, p, dict(history), 0, 4)
        actual, nodes, memo_hits = self.engine.solve_race(p, history, 0, 4, depth=4, node_cap=100000)
        self.assertEqual(actual, expected)
        self.assertGreater(nodes, 0)
        self.assertGreaterEqual(memo_hits, 0)

    def test_placed_walls_block_straight_and_one_diagonal_jump(self):
        # Horizontal wall blocks the straight jump; vertical wall blocks its east diagonal.
        p = position(40, 31, h=1 << 20, v=1 << 28)
        moves = self.engine.legal(p)
        self.assertIn(30, moves)
        self.assertNotIn(22, moves)
        self.assertNotIn(32, moves)
        expected = reference(self.engine, p, {p: 1}, 0, 3)
        self.assertEqual(self.engine.solve_race(p, {p: 1}, 0, 3, depth=3, node_cap=100000)[0], expected)

    def test_repetition_history_and_cutoff_draws(self):
        p = position(40, 4)
        history = {p: 3}
        self.assertEqual(self.engine.solve_race(p, history, 0, 8, depth=8)[0], 0)
        history = {p: 1}
        expected = reference(self.engine, p, dict(history), 0, 3)
        self.assertEqual(self.engine.solve_race(p, history, 0, 3, depth=8)[0], expected)
        self.assertEqual(self.engine.solve_race(p, history, 3, 3, depth=8)[0], 0)

    def test_second_prior_occurrence_makes_a_future_child_repetition(self):
        p = position(40, 4)
        q = tuple(self.engine.advance(p, 39))
        self.assertEqual(q[0:2], p[0:2])
        history = {p: 1, q: 2}
        expected = reference(self.engine, p, dict(history), 0, 5)
        self.assertEqual(self.engine.solve_race(p, history, 0, 5, depth=5, node_cap=100000)[0], expected)

    def test_horizon_and_node_cap_are_unknown_not_draws(self):
        p = position(40, 4)
        self.assertEqual(self.engine.solve_race(p, {p: 1}, 0, 12, depth=1, node_cap=100000)[0], 2)
        value, nodes, _ = self.engine.solve_race(p, {p: 1}, 0, 12, depth=12, node_cap=1)
        self.assertEqual((value, nodes), (2, 1))
        won = position(0, 40, turn=1)
        self.assertEqual(self.engine.solve_race(won, {won: 1}, 0, 0, depth=1)[0], -1)

    def test_wall_history_cannot_recur_after_wall_reserves_empty(self):
        p = position(40, 4)
        stale = (0, 1 << 5, p[2] | (1 << 14))
        value, _, _ = self.engine.solve_race(p, {stale: 3, p: 1}, 0, 2, depth=2, node_cap=10000)
        self.assertEqual(value, reference(self.engine, p, {p: 1}, 0, 2))

    def test_tablebase_search_matches_proof_search_at_short_cutoff(self):
        p = position(40, 31)
        model = Net(self.engine.planes, 8, 1).eval()
        job = dict(tree=9, position=p, history={p: 1}, ply=0, max_plies=4)
        expected = self.engine.search(model, [job], 32, 'cpu')
        other = Engine(1, race_solver='tablebase')
        try:
            actual = other.search(model, [job], 32, 'cpu')
            import numpy as np
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        finally:
            other.close()

    def test_native_search_uses_opt_in_proofs_and_reports_coverage(self):
        p = position(40, 31)
        model = Net(self.engine.planes, 8, 1).eval()
        self.engine.search(model, [dict(tree=4, position=p, history={p: 1}, ply=0, max_plies=12)],
                           8, 'cpu')
        self.assertGreater(self.engine.stats['race_queries'], 0)
        self.assertEqual(self.engine.stats['race_queries'],
                         self.engine.stats['race_solved'] + self.engine.stats['race_unknown'])
        self.assertGreater(self.engine.stats['race_nodes'], 0)
        self.assertGreaterEqual(self.engine.stats['race_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
