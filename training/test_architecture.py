"""Architecture compatibility, gradients, and persisted configuration."""
import random
import tempfile
from pathlib import Path
import unittest

import torch
from core import ARCHITECTURES, PolicyValue, board_features, encode, model_from_config, update
from train import atomic_save, new_arm


class ArchitectureTests(unittest.TestCase):
    def test_legacy_and_residual_exports_roundtrip(self):
        torch.set_num_threads(1)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                torch.manual_seed(11)
                model = PolicyValue(architecture)
                cfg = {} if architecture == 'small' else dict(architecture=architecture)
                x = torch.randn(3, 9, 9, 9)
                before = model(x)
                self.assertEqual(before[0].shape, (3, 209))
                self.assertEqual(before[1].shape, (3,))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'bot.pt'
                    atomic_save(dict(config=cfg, model=model.state_dict()), path)
                    saved = torch.load(path, weights_only=True)
                    restored = model_from_config(saved['config'])
                    restored.load_state_dict(saved['model'])
                    for actual, expected in zip(restored(x), before):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                loaded, optimizer = new_arm(model.state_dict(), 'cpu', cfg)
                policy = torch.zeros(209); policy[76] = 1
                loss = update(loaded, optimizer, [(x[0], policy, 1.), (x[1], None, -1.)],
                              2, random.Random(1), 'cpu')
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss.values()))
                for name, parameter in loaded.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(any(not torch.equal(a, b) for a, b in
                                    zip(loaded.parameters(), model.parameters())))

    def test_precision_config_and_cpu_fallback(self):
        torch.set_num_threads(1)
        fp32 = model_from_config({'architecture': 'residual32'})
        bf16 = model_from_config({'architecture': 'residual32', 'precision': 'bf16'})
        bf16.load_state_dict(fp32.state_dict())
        x = torch.randn(2, 9, 9, 9)
        for actual, expected in zip(bf16(x), fp32(x)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(bf16.precision, 'bf16')
        with self.assertRaises(ValueError):
            PolicyValue(precision='fp16')

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA required')
    def test_cuda_bf16_update_and_outputs(self):
        model = model_from_config({'architecture': 'residual32', 'precision': 'bf16'}).cuda()
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.rand(2, 9, 9, 9)
        policy = torch.ones(209) / 209
        before = model.policy.weight.detach().clone()
        loss = update(model, optimizer, [(x[0], policy, 1.), (x[1], None, -1.)],
                      2, random.Random(1), 'cuda')
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss.values()))
        self.assertFalse(torch.equal(before, model.policy.weight))
        self.assertTrue(all(t.dtype == torch.float32 for t in model(x.cuda())))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_small_checkpoint_layout_is_unchanged(self):
        model = PolicyValue()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 370866)
        self.assertEqual(tuple(model.state_dict()['trunk.5.weight'].shape), (128, 2592))
        with self.assertRaises(ValueError):
            PolicyValue('unknown')


    def test_wall_geometry_and_metadata(self):
        position = dict(pawns=[0, 80], wallsLeft=[7, 3], turn=1,
                        hWalls=[0, 63], vWalls=[7, 56])
        x = encode(position, 2, .25).unsqueeze(0)
        edges = board_features(x, 'edges')
        torch.testing.assert_close(edges[:, :9], x)
        self.assertEqual(edges[0, 10, 0, :2].tolist(), [1, 1])
        self.assertEqual(edges[0, 9, 1, :2].tolist(), [1, 1])
        self.assertEqual(edges[0, 12, :2, 7].tolist(), [1, 1])
        self.assertEqual(edges[0, 11, :2, 8].tolist(), [1, 1])
        self.assertEqual(int(edges[:, 9:].sum()), 52)
        lattice = board_features(x, 'lattice')
        self.assertEqual(lattice.shape, (1, 10, 17, 17))
        self.assertEqual(lattice[0, 0, 0, 0], 1)
        self.assertEqual(lattice[0, 1, 16, 16], 1)
        self.assertEqual(lattice[0, 2, 1, :3].tolist(), [1, 1, 1])
        self.assertEqual(lattice[0, 3, :3, 15].tolist(), [1, 1, 1])
        self.assertEqual(int(lattice[0, 2:4].sum()), 12)
        self.assertEqual(int(lattice[0, 9].sum()), 81)
        for channel in range(4, 9):
            torch.testing.assert_close(lattice[:, channel],
                                       x[:, channel, :1, :1].expand(1, 17, 17))
