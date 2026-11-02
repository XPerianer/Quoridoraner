"""Architecture integration: action ordering, gradients, checkpoint compatibility, compilation."""
import tempfile
import unittest
from pathlib import Path

import torch

from az import Microbatch, Net, TransformerNet, backward_batch, load_model, save_model


class TransformerTest(unittest.TestCase):
    def test_accumulated_gradients_sparse_policy_and_uneven_chunks(self):
        torch.manual_seed(9)
        model = TransformerNet(width=24, heads=3, blocks=1)
        x = torch.randn(5, 19, 9, 9)
        target = torch.randn(5, 209).softmax(1)
        value = torch.randn(5)
        for mask in [torch.tensor([False, False, True, False, True]), torch.zeros(5, dtype=torch.bool)]:
            model.zero_grad(set_to_none=True)
            expected = backward_batch(model, x, target, mask, value, 'cpu', value_weight=1.7)
            gradients = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            actual = backward_batch(model, x, target, mask, value, 'cpu', value_weight=1.7, microbatch=2)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            for parameter, gradient in zip(model.parameters(), gradients):
                torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)

    def test_microbatch_preserves_reused_outputs_and_remainder(self):
        buffer = torch.empty(3, 2)

        def reused(x):
            out = buffer[:len(x)]
            out.copy_(x * 2)
            return out, out[:, 0]

        x = torch.arange(22).reshape(11, 2).float()
        policy, value = Microbatch(reused, 3)(x)
        torch.testing.assert_close(policy, x * 2)
        torch.testing.assert_close(value, x[:, 0] * 2)
        for size in [1, 3]:
            policy, value = Microbatch(reused, 3)(x[:size])
            torch.testing.assert_close(policy, x[:size] * 2)

    def test_shapes_action_order_and_gradient(self):
        torch.manual_seed(7)
        model = TransformerNet(width=24, heads=3, blocks=2)
        x = torch.randn(3, 19, 9, 9)
        captured = []
        handle = model.policy_head.register_forward_hook(lambda module, args, output: captured.append(output))
        policy, value = model(x)
        handle.remove()
        board = captured[0].reshape(3, 9, 9, 3)
        expected = torch.cat((board[..., 0].flatten(1), board[:, :8, :8, 1].flatten(1),
                              board[:, :8, :8, 2].flatten(1)), dim=1)
        torch.testing.assert_close(policy, expected)
        self.assertEqual(policy.shape, (3, 209))
        self.assertEqual(value.shape, (3,))
        self.assertTrue((value.abs() <= 1).all())
        (policy.square().mean() + value.square().mean()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_roundtrip_and_legacy(self):
        x = torch.randn(2, 19, 9, 9)
        with tempfile.TemporaryDirectory() as directory:
            for model in (TransformerNet(width=24, heads=3, blocks=1, drop_planes=[4]),
                          Net(channels=8, blocks=1)):
                model.eval()
                path = Path(directory) / 'model.pt'
                save_model(model, path, iteration=2)
                restored, metadata = load_model(path, 'cpu')
                self.assertEqual(type(model), type(restored))
                self.assertEqual(metadata['iteration'], 2)
                for a, b in zip(model(x), restored(x)):
                    torch.testing.assert_close(a, b)

    def test_compiled_bfloat16_channels_last(self):
        model = TransformerNet(width=24, heads=3, blocks=1).eval().to(
            dtype=torch.bfloat16, memory_format=torch.channels_last)
        x = torch.randn(2, 19, 9, 9).to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        # Fullgraph catches graph breaks without relying on a local CUDA toolchain.
        compiled = torch.compile(model, backend='eager', fullgraph=True, dynamic=True)
        with torch.inference_mode():
            for a, b in zip(model(x), compiled(x)):
                torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
