"""Stable MPS shapes and an optional CPU lane for latency-bound small batches."""
import copy

import torch


class MpsInference:
    def __init__(self, model, mode='bucketed'):
        if mode not in ('eager', 'bucketed', 'hybrid'):
            raise ValueError('Unknown MPS inference mode')
        self.model = model
        self.mode = mode
        # Measured crossovers for the original CNNs. New variants stay on MPS
        # until their CPU/GPU crossover has been measured.
        # Keep the legacy cutoff unchanged for old small-network checkpoints.
        self.cpu_batch_limit = {'small': 16, 'residual32': 4, 'residual64': 2,
                                'edges': 0, 'lattice': 0, 'mlp': 0}.get(
            getattr(model, 'architecture', 'small'), 0)
        self.buffers = {}
        self.cpu_model = None
        self.cpu_versions = None
        self.cpu_batches = 0
        self.mps_batches = 0

    def __call__(self, features):
        n = len(features)
        if self.mode == 'hybrid' and n <= self.cpu_batch_limit:
            # The measured small-batch CPU latency is lower than an MPS round trip.
            # Refresh after every optimizer/load_state_dict change, never during
            # a search. This is a float32 copy, not quantization or stale weights.
            versions = tuple(p._version for p in self.model.parameters())
            if self.cpu_model is None:
                self.cpu_model = copy.deepcopy(self.model).to('cpu').eval()
                self.cpu_versions = versions
            elif versions != self.cpu_versions:
                self.cpu_model.load_state_dict({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
                self.cpu_versions = versions
            logits, values = self.cpu_model(features)
            self.cpu_batches += 1
            return torch.cat((logits, values[:, None]), dim=1)
        if self.mode != 'eager':
            size = max(8, 1 << (n - 1).bit_length())
            if size not in self.buffers:
                self.buffers[size] = torch.zeros(size, 9, 9, 9)
            buffer = self.buffers[size]
            buffer[:n].copy_(features)
            features = buffer
        # The CPU input remains alive until the blocking output copy below.
        # Queue H2D asynchronously to avoid a separate host-side fence.
        logits, values = self.model(features.to('mps', non_blocking=self.mode != 'eager'))
        self.mps_batches += 1
        return torch.cat((logits, values[:, None]), dim=1).cpu()[:n]
