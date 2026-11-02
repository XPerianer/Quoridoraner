"""The browser export keeps full native features for both ResNet input variants."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import az
from export_web import export_model


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                     'Optional web export dependencies not installed')
class ExportWebTest(unittest.TestCase):
    def test_both_plane_variants_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for planes in (19, 10):
                checkpoint = root / f'{planes}.pt'
                az.save_model(az.Net(planes=planes, channels=8, blocks=1).eval(), checkpoint)
                result = export_model(checkpoint, root / 'models', f'test-{planes}', 'Test', fixtures=4)
                self.assertEqual(result['planes'], 19)
                self.assertLess(result['parityMaxAbsoluteError']['logits'], 2e-5)
                fixture = json.loads((root / 'models' / f'test-{planes}-fixtures.json').read_text())
                self.assertEqual({r['position']['turn'] for r in fixture['rows']}, {0, 1})
                self.assertEqual(len(fixture['rows'][0]['features']), 19 * 81)
            manifest = json.loads((root / 'bots.json').read_text())
            self.assertEqual(len(manifest['bots']), 2)


if __name__ == '__main__':
    unittest.main()
