import io
import unittest
import torch
from adaptive import opponent_nodes, promotion_check
from curriculum import evaluation_target


class AdaptiveTests(unittest.TestCase):
    def test_promotion_failure_cutoffs_cap_and_serialization(self):
        state = dict(config=dict(adaptive_negamax=True, negamax_start_nodes=32, negamax_nodes=100,
                                 promotion_games=4, promotion_win_rate=.75, promotion_streak=2,
                                 eval_games=2, heldout_games=2, iterations=8),
                     mode='negamax', phase='train', iteration=0, records=[])
        self.assertEqual(opponent_nodes(state, 'monitor'), 32)
        state['phase'] = 'eval'
        self.assertEqual(opponent_nodes(state, 'monitor'), 100)
        self.assertEqual(opponent_nodes(state, 'promotion'), 32)
        self.assertEqual(evaluation_target(state), 6)
        for iteration, wins, expected in [(0, 3, 32), (1, 2, 32), (2, 3, 32),
                                           (3, 4, 64), (4, 4, 64), (5, 4, 100)]:
            state['iteration'] = iteration
            nodes = state['adaptive']['nodes']
            state['records'] += [dict(mode='negamax', phase='eval', suite='promotion',
                                     iteration=iteration, opponent_nodes=nodes,
                                     score=1 if i < wins else .5,
                                     termination='goal' if i < wins else 'maxPlies') for i in range(4)]
            promotion_check(state)
            self.assertEqual(state['adaptive']['nodes'], expected)
            n = len(state['adaptive']['history'])
            promotion_check(state)
            self.assertEqual(len(state['adaptive']['history']), n)
            stream = io.BytesIO()
            torch.save(state, stream)
            stream.seek(0)
            state = torch.load(stream, weights_only=True)
        state['iteration'] = 8
        self.assertEqual(evaluation_target(state), 8)


if __name__ == '__main__':
    unittest.main()
