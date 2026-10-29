"""Optional integration tests; download opponents with fetch_external_bots.py first."""
import unittest

import torch
from az import Engine, START, winner
from external_arena import BOTS, Goris, Sigma, check


@unittest.skipUnless((BOTS / 'SigmaQuoridor/game.py').exists(), 'external bots not downloaded')
class ExternalAdaptersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.engine = Engine(1)

    def test_jump_diagonal_and_goal(self):
        cases = [
            # Adjacent pawns jump straight; the second player eventually reaches its goal.
            ([67, 13, 58, 22, 49, 31, 40, 49, 31, 58, 22, 67, 13, 76], 1),
            # Block the square behind the opposing pawn, enabling a diagonal go-around.
            ([67, 13, 58, 22, 49, 31, 40, 101, 30], -1),
        ]
        for adapter in (Goris, Sigma):
            for sequence, expected_winner in cases:
                with self.subTest(adapter=adapter.__name__, sequence=sequence):
                    opponent = adapter(1, 7)
                    pos = START
                    try:
                        for move in sequence:
                            check(self.engine, pos, opponent)
                            self.assertIn(move, self.engine.legal(pos))
                            opponent.advance(move)
                            pos = self.engine.advance(pos, move)
                        self.assertEqual(winner(pos), expected_winner)
                        if expected_winner == -1:
                            check(self.engine, pos, opponent)
                    finally:
                        opponent.close()


if __name__ == '__main__':
    unittest.main()
