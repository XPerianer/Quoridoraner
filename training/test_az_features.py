"""Compact inputs, legacy checkpoints, and mixed-encoding search compatibility."""
import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import az

from az import (Engine, Net, SIMPLIFIED_PLANES, START, head_to_head,
                load_model, model_from_args, save_model)


class FeatureSetTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_selection_and_excluded_channels(self):
        x = torch.randn(2, 19, 9, 9)
        models = [Net(10, 8, 1)]
        if hasattr(az, 'TransformerNet'):
            models.append(az.TransformerNet(10, width=24, heads=3, blocks=1))
        for model in models:
            model.eval()
            compact = x[:, SIMPLIFIED_PLANES]
            changed = x.clone()
            changed[:, [4, 5, 6, 7, 14, 15, 16, 17, 18]] += 100
            for expected, direct, perturbed in zip(model(x), model(compact), model(changed)):
                torch.testing.assert_close(expected, direct)
                torch.testing.assert_close(expected, perturbed)
            model(x)[0].square().mean().backward()
            self.assertIsNotNone(model.stem[0].weight.grad if isinstance(model, Net) else model.stem.weight.grad)

    def test_checkpoint_encoding_and_cli_factory(self):
        for feature_set, planes in [('full', 19), ('simplified', 10)]:
            args = argparse.Namespace(feature_set=feature_set, channels=8, blocks=1)
            model = model_from_args(args, 19).eval()
            self.assertEqual(model.stem[0].in_channels, planes)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                save_model(model, path)
                restored, _ = load_model(path, 'cpu')
                self.assertEqual(restored.config['planes'], planes)
                x = torch.randn(2, 19, 9, 9)
                for a, b in zip(model(x), restored(x)):
                    torch.testing.assert_close(a, b)

    def test_native_features_and_mixed_model_search(self):
        engine = Engine(1)
        features = engine.features([START], np.ones(1, np.int32), np.ones(1, np.float32),
                                   np.zeros(1, np.uint8))
        compact = features[:, SIMPLIFIED_PLANES]
        self.assertEqual(compact.shape, (1, 10, 9, 9))
        self.assertEqual(compact[0, 0].sum(), 1)
        self.assertEqual(compact[0, 1].sum(), 1)
        np.testing.assert_array_equal(compact[0, 8:], 1)
        self.assertTrue(np.isin(compact[0, 6:8], [0, 1]).all())
        result = head_to_head(engine, Net(19, 8, 1).eval(), Net(10, 8, 1).eval(),
                              'cpu', 1, 4)
        self.assertEqual(result['games'], 2)


if __name__ == '__main__':
    unittest.main()
