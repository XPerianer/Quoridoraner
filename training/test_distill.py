import unittest
from unittest.mock import patch

import numpy as np
import torch

from distill import distillation_loss, replay_split
from evaluate_transformer import SearchBudgetEngine
from az import Engine
from az import START, TransformerNet, winner
from evaluate_timed import TimedEngine


class DistillationTest(unittest.TestCase):
    def test_timed_puct_returns_legal_moves_with_usable_trees(self):
        torch.set_num_threads(1)
        model = TransformerNet(width=24, blocks=1, heads=3).eval()
        engine = TimedEngine(1)
        position, history = START, {START: 1}
        try:
            for ply in range(6):
                move, timing = engine.timed_search(model, dict(tree=0, position=position,
                    history=history, ply=ply, max_plies=100), 3, 'cpu')
                self.assertIn(move, engine.legal(position))
                self.assertGreater(timing['root_completed'], 0)
                self.assertLess(timing['milliseconds'], 1000)
                position = engine.advance(position, move, 0)
                history[position] = history.get(position, 0)+1
                if winner(position) != -1:
                    break
        finally:
            engine.close()

    def test_search_budget_routes_by_model_and_records_moves(self):
        engine = object.__new__(SearchBudgetEngine)
        a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        engine.model_budgets = {id(a): ('a', 800), id(b): ('b', 200)}
        engine.side_timings = {}
        jobs = [{'tree': 1}, {'tree': 2}]
        with patch.object(Engine, 'search', return_value='result') as search:
            self.assertEqual(engine.search(a, jobs, 100, 'cpu', moves=True), 'result')
            search.assert_called_with(a, jobs, 800, 'cpu', moves=True)
            engine.search(b, jobs[:1], 100, 'cpu')
            search.assert_called_with(b, jobs[:1], 200, 'cpu', moves=False)
        self.assertEqual(engine.side_timings['a']['moves'], 2)
        self.assertEqual(engine.side_timings['b']['moves'], 1)

    def test_teacher_detached_and_student_learns(self):
        student = torch.randn(4, 209, requires_grad=True)
        value = torch.randn(4, requires_grad=True)
        teacher = torch.randn(4, 209, requires_grad=True)
        teacher_value = torch.randn(4, requires_grad=True)
        p, v = distillation_loss(student, value, teacher, teacher_value)
        (p + v).backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_value.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())
        p, v = distillation_loss(teacher, teacher_value, teacher, teacher_value)
        self.assertAlmostEqual(p.item(), 0, places=5)
        self.assertEqual(v.item(), 0)

    def test_duplicate_states_do_not_cross_split(self):
        states = np.random.default_rng(1).integers(0, 100000, (200, 3), dtype=np.uint64)
        states = np.concatenate([states, states])
        train, test = replay_split(states)
        self.assertGreater(len(test), 0)
        self.assertGreater(len(train), 0)
        self.assertFalse(set(map(tuple, states[train])) & set(map(tuple, states[test])))


if __name__ == '__main__':
    unittest.main()
