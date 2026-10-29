"""Exercise real Ctrl-C, an in-flight game, and deterministic CPU resume end to end."""
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch

from core import ROOT


class ResumeTest(unittest.TestCase):
    def test_interrupted_run_matches_uninterrupted_run(self):
        self.check_resume(1, 1)

    def test_parallel_run_can_resume_with_one_worker(self):
        self.check_resume(2, 1)

    def test_direct_run_resumes_across_worker_counts(self):
        self.check_resume(3, 1, engine='direct')

    def test_bridge_checkpoint_can_migrate_to_direct(self):
        self.check_resume(2, 2, engine='typescript', resume_engine='direct')

    def test_curriculum_and_final_heldout_resume_exactly(self):
        self.check_resume(2, 1, engine='direct', curriculum='races')

    def test_adaptive_opponent_resumes_exactly(self):
        self.check_resume(2, 1, engine='direct', adaptive=True)

    def test_gated_curriculum_and_balanced_replay_resume_exactly(self):
        self.check_resume(2, 1, engine='direct', curriculum='gated', adaptive=True)

    def test_rolling_randomized_run_resumes_exactly(self):
        self.check_resume(2, 1, engine='direct', randomized=True)

    def test_residual_architecture_resumes_exactly(self):
        self.check_resume(2, 1, engine='direct', architecture='residual32')

    def check_resume(self, workers, resume_workers, engine='typescript', resume_engine=None, curriculum='none', adaptive=False, randomized=False, architecture='small'):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'interrupted'
            control = Path(directory) / 'control'
            command = [sys.executable, str(ROOT / 'training/train.py')]
            options = ['--engine', engine, '--device', 'cpu', '--plain', '--iterations',
                       '3' if curriculum == 'gated' else '1', '--games', '2',
                       '--eval-games', '2', '--heldout-games', '2', '--curriculum', curriculum,
                       '--simulations', '8', '--max-plies', '40',
                       '--negamax-nodes', '100', '--updates', '2', '--checkpoint-seconds', '0.01',
                       '--architecture', architecture]
            if adaptive:
                options += ['--adaptive-negamax', '--negamax-start-nodes', '16', '--promotion-games', '2']
            if randomized:
                options += ['--games', '8', '--parallel-games', '2', '--game-scheduler', 'rolling',
                            '--search-budget', 'randomized', '--fast-simulations', '2']
            if curriculum == 'gated':
                options += ['--curriculum-games', '4', '--replay-sampling', 'task',
                            '--eval-every', '1', '--curriculum-streak', '1', '--curriculum-win-rate', '0.25']
            with (Path(directory) / 'log').open('w') as log:
                process = subprocess.Popen(command + options + ['--out', str(run), '--workers', str(workers)], stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if (run / 'latest.pt').exists():
                            state = torch.load(run / 'latest.pt', weights_only=True)
                            if state['phase'] == 'train' and state['game'] is not None and state['game']['ply'] >= 2:
                                break
                        if process.poll() is not None:
                            self.fail('Training exited before interrupt')
                        time.sleep(0.02)
                    else:
                        self.fail('Timed out waiting for an in-progress game')
                    process.send_signal(signal.SIGINT)
                    self.assertEqual(process.wait(timeout=30), 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                state = torch.load(run / 'latest.pt', weights_only=True)
                if curriculum == 'none':
                    self.assertIsNotNone(state['game'])
                    self.assertTrue(state['trees'])
                    self.assertEqual(len(state['active_games']), 2)
                    self.assertGreaterEqual(state['game']['ply'], 2)
                subprocess.run(command + ['--resume', str(run), '--device', 'cpu', '--plain', '--workers', str(resume_workers), '--engine', resume_engine or engine],
                               stdout=log, stderr=log, check=True, timeout=60)
                subprocess.run(command + options + ['--out', str(control)],
                               stdout=log, stderr=log, check=True, timeout=60)
            a = torch.load(run / 'latest.pt', weights_only=True)
            b = torch.load(control / 'latest.pt', weights_only=True)
            self.assertEqual(a['phase'], 'done')
            self.assertEqual(a['records'], b['records'])
            self.assertEqual(a.get('adaptive'), b.get('adaptive'))
            self.assertEqual(a.get('curriculum_progress'), b.get('curriculum_progress'))
            if curriculum == 'gated':
                self.assertGreaterEqual(a['curriculum_progress']['negamax']['stage'], 1)
            self.assertEqual(a['rng'], b['rng'])
            self.assertEqual(a['loss_history'], b['loss_history'])
            for mode in ('selfplay', 'negamax'):
                for name in a['arms'][mode]['model']:
                    self.assertTrue(torch.equal(a['arms'][mode]['model'][name], b['arms'][mode]['model'][name]))
                self.assertEqual(len(a['arms'][mode]['replay']), len(b['arms'][mode]['replay']))
                for actual, expected in zip(a['arms'][mode]['replay'], b['arms'][mode]['replay']):
                    self.assertTrue(torch.equal(actual[0], expected[0]))
                    self.assertEqual(actual[2], expected[2])
                    self.assertEqual(actual[3:], expected[3:])
                    if actual[1] is None:
                        self.assertIsNone(expected[1])
                    else:
                        self.assertTrue(torch.equal(actual[1], expected[1]))
            self.assertTrue(any(row[1] is None for row in a['arms']['negamax']['replay']))
            training = [r for r in a['records'] if r['mode'] == 'negamax' and r['phase'] == 'train']
            self.assertEqual(len(a['arms']['negamax']['replay']), sum(r['plies'] for r in training))
