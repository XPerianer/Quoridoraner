"""Opt-in RL optimizer, schedule and playout-cap ablation invariants."""
import random
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from az import (Net, make_rl_scheduler, new_optimizer, new_rl_optimizer, playout_budget,
                restore_rl_scheduler)


class AblationTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_adamw_default_keeps_legacy_optimizer(self):
        model = Net(planes=19, channels=8, blocks=1)
        old = new_optimizer(model, 5e-4)
        new = new_rl_optimizer(model, 5e-4, 'adamw')
        self.assertIs(type(old), type(new))
        self.assertEqual(old.param_groups[0]['params'], new.param_groups[0]['params'])

    def test_muon_flattens_conv_weights_and_adamw_owns_heads_vectors(self):
        model = Net(planes=19, channels=8, blocks=1)
        optimizer = new_rl_optimizer(model, 1e-3, 'muon')
        adam_params = optimizer.param_groups[1]['params']
        self.assertTrue(any(p is model.stem[0].weight for p in optimizer.model_muon_params))
        self.assertTrue(any(p is model.policy_head[-1].weight for p in adam_params))
        self.assertTrue(any(p is model.stem[1].weight for p in adam_params))
        before = model.stem[0].weight.detach().clone()
        model(torch.randn(2, 19, 9, 9))[0].sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.stem[0].weight))
        self.assertEqual(tuple(model.stem[0].weight.shape), (8, 19, 3, 3))
        state = optimizer.state_dict()
        restored = new_rl_optimizer(model, 1e-3, 'muon')
        restored.load_state_dict(state)
        self.assertEqual(len(restored.muon_impl.state), len(optimizer.muon_impl.state))

    def test_channels_last_muon_resume_matches_uninterrupted_updates(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        a, b = Net(planes=19, channels=8, blocks=1), Net(planes=19, channels=8, blocks=1)
        b.load_state_dict(a.state_dict())
        a.to(memory_format=torch.channels_last); b.to(memory_format=torch.channels_last)
        oa, ob = new_rl_optimizer(a, 3e-4, 'muon'), new_rl_optimizer(b, 3e-4, 'muon')
        sa, sb = make_rl_scheduler(oa, 'cosine-restarts', 8, 3e-4), make_rl_scheduler(ob, 'cosine-restarts', 8, 3e-4)
        batches = [torch.randn(2, 19, 9, 9).to(memory_format=torch.channels_last) for _ in range(5)]
        for x in batches[:3]:
            oa.zero_grad(); a(x)[0].square().mean().backward(); oa.step(); sa.step()
            ob.zero_grad(); b(x)[0].square().mean().backward(); ob.step(); sb.step()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        c = Net(planes=19, channels=8, blocks=1).to(memory_format=torch.channels_last)
        c.load_state_dict(a.state_dict())
        self.assertEqual((a.training, c.training), (True, True))
        for p, q in zip(a.parameters(), c.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for p, q in zip(a.buffers(), c.buffers()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(a(batches[3])[0], c(batches[3])[0], rtol=0, atol=0)
        oc = new_rl_optimizer(c, 3e-4, 'muon'); oc.load_state_dict(copy.deepcopy(oa.state_dict()))
        sc = make_rl_scheduler(oc, 'cosine-restarts', 8, 3e-4)
        restore_rl_scheduler(sc, copy.deepcopy(sa.state_dict()), oc)
        for p, q in zip(a.parameters(), c.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for p, q in zip(a.buffers(), c.buffers()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(oa.muon_impl.state[oa.flat_params[0]]['momentum_buffer'],
                                   oc.muon_impl.state[oc.flat_params[0]]['momentum_buffer'], rtol=0, atol=0)
        for x in batches[3:]:
            oa.zero_grad(); a(x)[0].square().mean().backward()
            oc.zero_grad(); c(x)[0].square().mean().backward()
            for pa, pc in zip(a.parameters(), c.parameters()):
                if pa.grad is not None:
                    torch.testing.assert_close(pa.grad, pc.grad, rtol=0, atol=0)
            for model, optimizer, scheduler in ((a, oa, sa), (c, oc, sc)):
                optimizer.step(); scheduler.step()
            self.assertEqual([g['lr'] for g in oa.param_groups], [g['lr'] for g in oc.param_groups])
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, c.state_dict()[name], rtol=0, atol=0)
        self.assertEqual([g['lr'] for g in oa.param_groups], [g['lr'] for g in oc.param_groups])

    def test_new_process_resume_muon_schedule_and_random_caps(self):
        script = Path(__file__).with_name('az.py')
        with tempfile.TemporaryDirectory() as directory:
            full, split = Path(directory) / 'full', Path(directory) / 'split'
            base = [sys.executable, str(script), 'rl', '--device', 'cpu', '--workers', '1',
                    '--channels', '8', '--blocks', '1', '--games', '2', '--simulations', '4',
                    '--max-plies', '1', '--replay', '32', '--batch-size', '2', '--reuse', '1',
                    '--negamax-fraction', '0', '--eval-every', '0', '--max-opening', '0', '--seed', '23',
                    '--optimizer', 'muon', '--schedule', 'cosine-restarts', '--schedule-steps', '37',
                    '--playout-cap', 'randomized-40-400']
            for out, iters in ((full, 2), (split, 1), (split, 2)):
                subprocess.run(base + ['--out', str(out), '--iterations', str(iters)],
                               check=True, capture_output=True, text=True)
            a, b = [torch.load(p / 'state.pt', weights_only=False) for p in (full, split)]
            for name in a['state']:
                torch.testing.assert_close(a['state'][name], b['state'][name], rtol=0, atol=0)
            self.assert_nested_equal(a['optimizer'], b['optimizer'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertEqual(a['rng'], b['rng'])
            row = a['metrics'][-1]
            self.assertGreater(row['playout_caps']['40'] + row['playout_caps']['400'], 0)
            self.assertEqual(row['policy_rows'] + row['value_only_rows'], row['positions'])

    def test_cosine_schedules_restore_live_lr_before_optimizer_step(self):
        for kind in ('cosine', 'cosine-restarts'):
            with self.subTest(kind=kind):
                torch.manual_seed(4)
                a = Net(planes=19, channels=8, blocks=1)
                b = Net(planes=19, channels=8, blocks=1)
                b.load_state_dict(a.state_dict())
                oa = new_rl_optimizer(a, 1e-3, 'muon')
                sa = make_rl_scheduler(oa, kind, 37, 1e-3)
                for i in range(7):
                    oa.zero_grad(); a(torch.full((2, 19, 9, 9), float(i)))[0].square().mean().backward()
                    oa.step(); sa.step()
                b.load_state_dict(a.state_dict())
                ob = new_rl_optimizer(b, 1e-3, 'muon')
                ob.load_state_dict(copy.deepcopy(oa.state_dict()))
                sb = make_rl_scheduler(ob, kind, 37, 1e-3)
                restore_rl_scheduler(sb, copy.deepcopy(sa.state_dict()), ob)
                # This check is before either next optimizer.step: loading scheduler state
                # alone does not apply the saved phase to live optimizer groups.
                self.assertEqual([g['lr'] for g in oa.param_groups], [g['lr'] for g in ob.param_groups])
                oa.zero_grad(); a(torch.ones(2, 19, 9, 9))[0].square().mean().backward()
                ob.zero_grad(); b(torch.ones(2, 19, 9, 9))[0].square().mean().backward()
                oa.step(); ob.step()
                self.assertEqual([g['lr'] for g in oa.param_groups], [g['lr'] for g in ob.param_groups])
                sa.step(); sb.step()
                self.assertEqual([g['lr'] for g in oa.param_groups], [g['lr'] for g in ob.param_groups])

    def test_cosine_schedule_stays_at_floor_after_horizon(self):
        model = Net(planes=19, channels=8, blocks=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = make_rl_scheduler(optimizer, 'cosine', 3, 1e-3)
        for _ in range(7):
            optimizer.zero_grad(); sum(p.sum() for p in model.parameters()).backward()
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-5, places=10)

    def test_randomized_caps_are_deterministic_and_fixed_mode_preserves_rng(self):
        a, b = random.Random(31), random.Random(31)
        self.assertEqual([playout_budget(a, 'randomized-40-400', 200) for _ in range(100)],
                         [playout_budget(b, 'randomized-40-400', 200) for _ in range(100)])
        a = random.Random(8); expected = a.getstate()
        self.assertEqual(playout_budget(a, 'fixed', 200), 200)
        self.assertEqual(a.getstate(), expected)
        self.assertEqual(set(playout_budget(random.Random(i), 'randomized-40-400', 200)
                             for i in range(40)), {40, 400})


if __name__ == '__main__':
    unittest.main()
