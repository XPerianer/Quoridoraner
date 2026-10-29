"""Checks for the AlphaZero pipeline: openings, perspective symmetry, canonical moves, search."""
import random
import unittest

import numpy as np
import torch

from az import Engine, Net, Replay, START, evaluation_opening, head_to_head, match, train_steps, turn, winner


def flip_position(position):
    """Swap the players and reflect the board vertically (same game from the other side)."""
    h, v, p = position
    a, b = p & 127, (p >> 7) & 127
    x, y, t = (p >> 14) & 15, (p >> 18) & 15, (p >> 22) & 1

    def cell(c):
        return (8 - c // 9) * 9 + c % 9

    def walls(mask):
        out = 0
        for i in range(64):
            if mask >> i & 1:
                out |= 1 << ((7 - i // 8) * 8 + i % 8)
        return out
    return (walls(h), walls(v), cell(b) | (cell(a) << 7) | (y << 14) | (x << 18) | ((t ^ 1) << 22))


class AzTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(2)

    def positions(self, n=40, seed=3):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            position = START
            for _ in range(rng.randint(0, 30)):
                legal = self.engine.legal(position)
                child = self.engine.advance(position, rng.choice(legal))
                if winner(child) != -1:
                    break
                position = child
            out.append(position)
        return out

    def test_openings_match_trainer(self):
        from curriculum import evaluation_opening as reference
        from direct import DirectBridge, pack
        bridge = DirectBridge(1)
        config = {'seed': 1, 'walls': 10, 'varied_eval': True}
        for suite in ('monitor', 'heldout'):
            for i in range(0, 20, 3):
                expected, _ = reference(bridge, config, i, suite)
                position, _ = evaluation_opening(self.engine, i, suite)
                self.assertEqual(position, tuple(pack(expected['position'])))

    def test_features_are_perspective_invariant(self):
        positions = self.positions()
        flipped = [flip_position(p) for p in positions]
        n = len(positions)
        ones, rem, zero = np.ones(n, np.int32), np.ones(n, np.float32), np.zeros(n, np.uint8)
        a = self.engine.features(positions, ones, rem, zero)
        b = self.engine.features(flipped, ones, rem, zero)
        keep = [c for c in range(self.engine.planes) if c != 15]  # plane 15 is the absolute turn
        np.testing.assert_array_equal(a[:, keep], b[:, keep])

    def test_mirror_features_match_mirrored_moves(self):
        # Mirrored features of a position: pawn plane reflects left-right.
        position = self.positions(1, seed=9)[0]
        f = self.engine.features([position, position], np.ones(2, np.int32), np.ones(2, np.float32),
                                 np.array([0, 1], np.uint8))
        np.testing.assert_array_equal(f[0, :2], f[1, :2, :, ::-1])
        np.testing.assert_array_equal(f[0, 8:12], f[1, 8:12, :, ::-1])

    def test_canonical_moves(self):
        for position in self.positions(20):
            legal = self.engine.legal(position)
            other = self.engine.legal(flip_position(position))
            t = turn(position)
            mapped = sorted(self.engine.canonical[t, 0, legal].tolist())
            self.assertEqual(mapped, sorted(self.engine.canonical[t ^ 1, 0, other].tolist()))
            for m in range(2):
                perm = self.engine.canonical[t, m]
                np.testing.assert_array_equal(perm[perm], np.arange(209))

    def test_search_finds_immediate_win(self):
        # White pawn one step from row 0, black far away: MCTS must step to the goal.
        position = (0, 0, 13 | (40 << 7) | (0 << 14) | (0 << 18))
        model = Net(self.engine.planes, 8, 1).eval()
        job = dict(tree=7, position=position, history={position: 1}, ply=0, max_plies=100)
        policy, _ = self.engine.search(model, [job], 64, 'cpu')
        self.assertEqual(int(policy[0].argmax()), 4)

    def test_training_and_match_smoke(self):
        replay = Replay(100)
        for p in self.positions(30):
            policy = np.zeros(209, np.float32)
            policy[self.engine.canonical[turn(p), 0, self.engine.legal(p)[0]]] = 1
            replay.add(p, 1, 1.0, policy, 1.0)
        model = Net(self.engine.planes, 8, 1)
        optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
        losses = train_steps(model, optimizer, replay, self.engine, 3, 16, 'cpu', np.random.default_rng(0))
        self.assertTrue(np.isfinite(losses['policy']) and np.isfinite(losses['value']))
        result = match(self.engine, model, 'cpu', 1, 4, nodes=50)
        self.assertEqual(result['games'], 2)

    def test_head_to_head_uses_both_networks(self):
        torch.manual_seed(0)
        a, b = Net(self.engine.planes, 8, 1).eval(), Net(self.engine.planes, 8, 1).eval()
        result = head_to_head(self.engine, a, b, 'cpu', 2, 4)
        self.assertEqual(result['games'], 4)
        self.assertEqual(result['wins'] + result['losses'] + result['cutoffs'], 4)


if __name__ == '__main__':
    unittest.main()
