"""Pipeline cache isolation, retained-tree recovery and CPU resume regressions."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

from az import Engine, Game, Net, Replay, START, play, with_walls


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_shared_cache_and_weight_invalidation(self):
        e = Engine(2)
        model = Net(e.planes, 8, 1).eval()
        jobs = [dict(tree=i, position=START, history={START: 1}, ply=0, max_plies=20) for i in range(8)]
        e.search(model, jobs, 4, 'cpu')
        self.assertGreater(e.stats['cache_hits'], 0)
        first = e.stats['unique_evaluations']
        e.clear()
        e.search(model, jobs, 4, 'cpu')
        self.assertEqual(first, e.stats['unique_evaluations'])
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
        e.clear()
        e.search(model, jobs, 4, 'cpu')
        self.assertGreater(e.stats['unique_evaluations'], first)

    def test_tree_and_active_game_resume(self):
        e = Engine(2)
        model = Net(e.planes, 8, 1).eval()
        games = [Game(i, START, {START: 1}, ['mcts', 'mcts']) for i in range(4)]
        rng = random.Random(4)
        saved = []

        class Paused(Exception):
            pass

        def pause(done, total):
            if games[0].ply == 3:
                import copy
                saved.extend([copy.deepcopy(games), e.snapshot(), rng.getstate()])
                raise Paused()
        with self.assertRaises(Paused):
            play(e, model, games, 'cpu', 8, max_plies=8, noise=True, temperature_plies=8,
                 rng=rng, progress=pause)
        play(e, model, games, 'cpu', 8, max_plies=8, noise=True, temperature_plies=8, rng=rng, resume=True)
        fresh = Engine(1)
        fresh.restore(saved[1])
        resumed_rng = random.Random()
        resumed_rng.setstate(saved[2])
        play(fresh, model, saved[0], 'cpu', 8, max_plies=8, noise=True, temperature_plies=8,
             rng=resumed_rng, resume=True)
        for a, b in zip(games, saved[0]):
            self.assertEqual((a.moves, a.result, a.history), (b.moves, b.result, b.history))
            for x, y in zip(a.samples, b.samples):
                np.testing.assert_array_equal(x[3], y[3])

    def test_exact_cpu_iteration_resume_in_new_process(self):
        script = Path(__file__).with_name('az.py')
        with tempfile.TemporaryDirectory() as directory:
            full, split = Path(directory) / 'full', Path(directory) / 'split'
            base = [sys.executable, str(script), 'rl', '--device', 'cpu', '--workers', '1',
                    '--channels', '8', '--blocks', '1', '--games', '2', '--simulations', '4',
                    '--max-plies', '4', '--replay', '32', '--batch-size', '4', '--reuse', '1',
                    '--negamax-fraction', '0', '--eval-every', '0', '--max-opening', '0']
            for out, iterations in ((full, 2), (split, 1), (split, 2)):
                subprocess.run(base + ['--out', str(out), '--iterations', str(iterations)],
                               check=True, capture_output=True, text=True)
            a, b = [torch.load(p / 'state.pt', weights_only=False) for p in (full, split)]
            for name in a['state']:
                torch.testing.assert_close(a['state'][name], b['state'][name], rtol=0, atol=0)
            self.assertEqual(a['rng'], b['rng'])
            self.assertEqual(a['replay']['next'], b['replay']['next'])
            for name in a['replay']['data']:
                np.testing.assert_array_equal(a['replay']['data'][name], b['replay']['data'][name])

    def test_snapshots_play_the_previous_model(self):
        script = Path(__file__).with_name('az.py')
        with tempfile.TemporaryDirectory() as directory:
            out, snapshots = Path(directory) / 'rl', Path(directory) / 'snapshots'
            subprocess.run([sys.executable, str(script), 'rl', '--device', 'cpu', '--workers', '1',
                            '--channels', '8', '--blocks', '1', '--games', '2', '--simulations', '4',
                            '--max-plies', '4', '--replay', '32', '--batch-size', '4', '--reuse', '1',
                            '--negamax-fraction', '0', '--eval-every', '0', '--max-opening', '0',
                            '--iterations', '2', '--eval-simulations', '4', '--out', str(out),
                            '--snapshot-dir', str(snapshots), '--snapshot-minutes', '1e-6', '--prev-eval-pairs', '1'],
                           check=True, capture_output=True, text=True)
            self.assertEqual(sorted(p.name for p in snapshots.glob('model-*.pt')),
                             ['model-h000-it00000.pt', 'model-h000-it00001.pt', 'model-h000-it00002.pt'])
            metrics = json.loads((snapshots / 'metrics.json').read_text())
            self.assertEqual([row['prev_eval']['opponent'] for row in metrics],
                             ['model-h000-it00000', 'model-h000-it00001'])
            self.assertEqual(metrics[0]['prev_eval']['games'], 2)

    def test_lost_process_recovers_inflight_checkpoint(self):
        script = Path(__file__).with_name('az.py')
        with tempfile.TemporaryDirectory() as directory:
            full, resumed = Path(directory) / 'full', Path(directory) / 'resumed'
            base = [sys.executable, str(script), 'rl', '--device', 'cpu', '--workers', '1',
                    '--channels', '8', '--blocks', '1', '--games', '4', '--simulations', '32',
                    '--max-plies', '12', '--replay', '128', '--batch-size', '8', '--reuse', '1',
                    '--negamax-fraction', '0', '--eval-every', '0', '--max-opening', '0',
                    '--iterations', '1', '--checkpoint-seconds', '0.001']
            subprocess.run(base + ['--out', str(full)], check=True, capture_output=True)
            process = subprocess.Popen(base + ['--out', str(resumed)], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 20
                found = False
                while time.monotonic() < deadline and process.poll() is None:
                    path = resumed / 'state.pt'
                    if path.exists():
                        state = torch.load(path, weights_only=False)
                        if state.get('games'):
                            found = True
                            process.kill()  # No graceful signal handler: simulate VM loss.
                            break
                    time.sleep(0.01)
                self.assertTrue(found, 'Did not observe an in-flight checkpoint')
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            subprocess.run(base + ['--out', str(resumed)], check=True, capture_output=True)
            a, b = [torch.load(p / 'state.pt', weights_only=False) for p in (full, resumed)]
            for name in a['state']:
                torch.testing.assert_close(a['state'][name], b['state'][name], rtol=0, atol=0)
            for name in a['replay']['data']:
                np.testing.assert_array_equal(a['replay']['data'][name], b['replay']['data'][name])

    def test_generation_and_pretrain_stage_resume(self):
        script = Path(__file__).with_name('az.py')
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'gen.npz'
            command = [sys.executable, str(script), 'gen', '--out', str(data), '--games', '4',
                       '--parallel', '2', '--max-plies', '4', '--budgets', '10', '--workers', '1']
            subprocess.run(command, check=True, capture_output=True)
            original = data.read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            with np.load(data) as regenerated:
                import io
                with np.load(io.BytesIO(original)) as first:
                    for name in first.files:
                        np.testing.assert_array_equal(first[name], regenerated[name])
            out = Path(directory) / 'pre'
            command = [sys.executable, str(script), 'pretrain', '--out', str(out), '--data', str(data),
                       '--device', 'cpu', '--workers', '1', '--channels', '8', '--blocks', '1',
                       '--epochs', '4', '--batch-size', '4', '--eval-pairs', '0']
            subprocess.run(command, check=True, capture_output=True)
            first = torch.load(out / 'pretrain-state.pt', weights_only=False)
            subprocess.run(command, check=True, capture_output=True)
            second = torch.load(out / 'pretrain-state.pt', weights_only=False)
            self.assertEqual(first['step'], second['step'])
            for name in first['state']:
                torch.testing.assert_close(first['state'][name], second['state'][name], rtol=0, atol=0)

    def test_replay_ring_snapshot(self):
        replay = Replay(3)
        for i in range(8):
            replay.add(START, 1, 1, None, float(i))
        restored = Replay.restore(replay.snapshot())
        self.assertEqual((restored.size, restored.next), (replay.size, replay.next))
        np.testing.assert_array_equal(restored.value, replay.value)

    def test_pawn_endgame_needs_no_neural_leaf(self):
        e = Engine(1)
        model = Net(e.planes, 8, 1).eval()
        p = (0, 0, 13 | (67 << 7))
        jobs = [dict(tree=0, position=p, history={p: 1}, ply=0, max_plies=10)]
        pi, _ = e.search(model, jobs, 32, 'cpu')
        self.assertEqual(int(pi[0].argmax()), 4)
        self.assertLess(e.stats['unique_evaluations'], 10)


if __name__ == '__main__':
    unittest.main()
