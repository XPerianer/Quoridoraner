import random
import unittest
from unittest.mock import patch

import torch

from core import Bridge, PolicyValue, encode, key, search_batch, update
from device import benchmark_devices, select_device


class DeviceTests(unittest.TestCase):
    def test_cuda_selection_and_unavailable_device_fallback(self):
        with patch('device.available_devices', return_value=['cpu', 'cuda']):
            self.assertEqual(select_device('cuda', 8), ('cuda', []))
            self.assertEqual(select_device('auto', 8, saved='cuda'), ('cuda', []))
        rows = [{'device': 'cpu', 'batch': 8, 'median_ms': 1.0}]
        with patch('device.available_devices', return_value=['cpu']), patch('device.benchmark_devices', return_value=rows):
            with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
                select_device('cuda', 8)
            self.assertEqual(select_device('auto', 8, saved='cuda'), ('cpu', rows))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is unavailable')
    def test_cuda_benchmark_rng_and_shared_inference_update(self):
        torch.set_num_threads(1)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        self.assertTrue(any(row['device'] == 'cuda' for row in benchmark_devices((2,), repeats=1)))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())))
        bridge = Bridge(workers=2)
        try:
            model = PolicyValue().to('cuda')
            state = bridge.ask()
            games = [{'id': str(i), 'state': state, 'ply': 0, 'history': {key(state['position']): 1}} for i in range(2)]
            policies = search_batch(model, bridge, games, 100, 4, random.Random(1), 'cuda')
            replay = [(encode(state['position']), policies[str(i)], 0.0) for i in range(2)]
            loss = update(model, torch.optim.AdamW(model.parameters()), replay, 2, random.Random(1), 'cuda')
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss.values()))
        finally:
            bridge.close()


if __name__ == '__main__':
    unittest.main()
