"""Kickstarting: teacher targets in the loss, the decaying weight, and the RL command with an anchor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from az import Engine, Net, Replay, START, TransformerNet, backward_batch, kickstart_weight, save_model, train_steps


class KickstartTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_teacher_terms_match_explicit_loss_and_microbatches(self):
        model = TransformerNet(width=24, heads=3, blocks=1)
        x = torch.randn(5, 19, 9, 9)
        target = torch.randn(5, 209).softmax(1)
        mask = torch.tensor([True, False, True, False, False])
        value = torch.randn(5)
        guide = (torch.randn(5, 209).softmax(1), torch.rand(5) * 2 - 1, 0.7, 0.3)
        model.zero_grad(set_to_none=True)
        totals = backward_batch(model, x, target, mask, value, 'cpu', 1.0, 0, guide)
        gradients = [p.grad.clone() for p in model.parameters()]

        model.zero_grad(set_to_none=True)
        logits, prediction = model(x)
        log_policy = logits.log_softmax(1)
        policy_loss = (-(target * log_policy).sum(1) * mask).sum() / mask.sum()
        value_loss = (prediction - value).square().mean()
        teacher_policy = -(guide[0] * log_policy).sum(1).mean()
        teacher_value = (prediction - guide[1]).square().mean()
        (policy_loss + value_loss + 0.7 * teacher_policy + 0.3 * teacher_value).backward()
        for actual, expected in zip(totals, (policy_loss, value_loss, teacher_policy, teacher_value)):
            torch.testing.assert_close(actual, expected.detach())
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)

        model.zero_grad(set_to_none=True)
        chunked = backward_batch(model, x, target, mask, value, 'cpu', 1.0, 2, guide)
        for a, b in zip(chunked, totals):
            torch.testing.assert_close(a, b)
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)

    def test_weight_decays_linearly_to_zero(self):
        args = argparse.Namespace(teacher='teacher.pt', teacher_weight=1.0, teacher_hours=4.0)
        self.assertEqual([kickstart_weight(args, 3600 * h) for h in (0, 1, 2, 4, 9)], [1.0, 0.75, 0.5, 0.0, 0.0])
        self.assertEqual(kickstart_weight(argparse.Namespace(teacher=None), 0), 0.0)

    def test_student_approaches_teacher(self):
        engine = Engine(1)
        teacher, student = Net(engine.planes, 8, 1).eval(), Net(engine.planes, 8, 1)
        replay = Replay(64)
        for _ in range(64):
            replay.add(START, 1, 1.0, None, 0.0)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
        rng = np.random.default_rng(0)
        first = train_steps(student, optimizer, replay, engine, 1, 16, 'cpu', rng, teacher=teacher, teacher_weight=1.0)
        last = train_steps(student, optimizer, replay, engine, 60, 16, 'cpu', rng, teacher=teacher, teacher_weight=1.0)
        self.assertLess(last['teacher_kl'], first['teacher_kl'] / 4)
        self.assertGreaterEqual(last['teacher_kl'], -1e-5)
        plain = train_steps(student, optimizer, replay, engine, 1, 16, 'cpu', rng, teacher=teacher, teacher_weight=0.0)
        self.assertNotIn('teacher_kl', plain)
        engine.close()

    def test_rl_records_weight_and_anchor_matches(self):
        script = Path(__file__).with_name('az.py')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            teacher = Net(Engine(1).planes, 8, 1)
            save_model(teacher, directory / 'teacher.pt')
            out, snapshots = directory / 'rl', directory / 'snapshots'
            command = [sys.executable, str(script), 'rl', '--device', 'cpu', '--workers', '1',
                       '--architecture', 'transformer', '--width', '16', '--heads', '2', '--blocks', '1',
                       '--games', '2', '--simulations', '4', '--max-plies', '4', '--replay', '32', '--batch-size', '4',
                       '--reuse', '1', '--negamax-fraction', '0', '--eval-every', '0', '--max-opening', '0',
                       '--eval-simulations', '4', '--out', str(out), '--teacher', str(directory / 'teacher.pt'),
                       '--teacher-hours', '1e-9', '--anchor', str(directory / 'teacher.pt'), '--anchor-pairs', '1',
                       '--snapshot-dir', str(snapshots), '--snapshot-minutes', '1e-6']
            subprocess.run(command + ['--iterations', '2'], check=True, capture_output=True, text=True)
            # Resuming keeps the saved teacher settings.
            start = command.index('--teacher')
            subprocess.run(command[:start] + command[start + 4:] + ['--iterations', '3'],
                           check=True, capture_output=True, text=True)
            metrics = json.loads((out / 'metrics.json').read_text())
            self.assertEqual([row['teacher_weight'] for row in metrics], [1.0, 0.0, 0.0])
            self.assertIn('teacher_kl', metrics[0])
            self.assertNotIn('teacher_kl', metrics[1])
            self.assertEqual([row['anchor_eval']['opponent'] for row in metrics], ['teacher'] * 3)
            self.assertEqual(metrics[0]['anchor_eval']['games'], 2)


if __name__ == '__main__':
    unittest.main()
