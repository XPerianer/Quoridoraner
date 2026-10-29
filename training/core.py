"""Policy/value network and PUCT search. Values are always for the side to move."""
import json
import time

import numpy as np
import torch
from torch import nn
from replay import sample as sample_replay

from bridge import Bridge, ROOT  # Re-export for bots and existing callers.


def key(position):
    return json.dumps(position, sort_keys=True, separators=(',', ':'))


def encode(position, repetitions=1, remaining=1.0):
    return encode_batch([(position, repetitions, remaining)])[0]


def encode_batch(rows):
    """Build a contiguous CPU batch without dispatching scalar tensor operations."""
    # Absolute coordinates; the turn plane identifies the player and its goal.
    batch = np.zeros((len(rows), 9, 9, 9), dtype=np.float32)
    for planes, (position, repetitions, remaining) in zip(batch, rows):
        for player, cell in enumerate(position['pawns']):
            planes[player, cell // 9, cell % 9] = 1
            planes[4 + player].fill(position['wallsLeft'][player] / 10)
        for channel, name in [(2, 'hWalls'), (3, 'vWalls')]:
            for anchor in position[name]:
                planes[channel, anchor // 8, anchor % 8] = 1
        planes[6].fill(position['turn'])
        planes[7].fill(min(repetitions, 3) / 3)
        planes[8].fill(remaining)
    return torch.from_numpy(batch)


RESIDUAL_SPECS = {'residual32': (32, 4), 'residual64': (64, 4),
                  'residual128': (128, 8), 'residual256': (256, 12),
                  'residual512': (512, 16)}
ARCHITECTURES = ('small', *RESIDUAL_SPECS, 'edges', 'lattice', 'mlp')


def board_features(x, architecture):
    """Derive geometry inside the model; preserve replay and engine formats."""
    if architecture == 'edges':
        edges = x.new_zeros((x.shape[0], 4, 9, 9))
        h, v = x[:, 2, :8, :8], x[:, 3, :8, :8]
        down = x.new_zeros((x.shape[0], 8, 9))
        down[:, :, :8] += h
        down[:, :, 1:] += h
        right = x.new_zeros((x.shape[0], 9, 8))
        right[:, :8, :] += v
        right[:, 1:, :] += v
        edges[:, 0, 0, :] = 1
        edges[:, 0, 1:, :] = down
        edges[:, 1, -1, :] = 1
        edges[:, 1, :8, :] = down
        edges[:, 2, :, 0] = 1
        edges[:, 2, :, 1:] = right
        edges[:, 3, :, -1] = 1
        edges[:, 3, :, :8] = right
        return torch.cat((x, edges), dim=1)
    if architecture == 'lattice':
        # Squares at even coordinates; each wall spans three intervening sites.
        board = x.new_zeros((x.shape[0], 10, 17, 17))
        board[:, :2, ::2, ::2] = x[:, :2]
        for offset in range(3):
            board[:, 2, 1::2, offset:offset+15:2] += x[:, 2, :8, :8]
            board[:, 3, offset:offset+15:2, 1::2] += x[:, 3, :8, :8]
        board[:, 4:9] = x[:, 4:9, :1, :1]
        board[:, 9, ::2, ::2] = 1
        return board
    return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return (x + 0.5 * self.conv2(self.conv1(x).relu())).relu()


class PolicyValue(nn.Module):
    def __init__(self, architecture='small', precision='fp32'):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unknown architecture: {architecture}')
        if precision not in ('fp32', 'bf16'):
            raise ValueError(f'Unknown precision: {precision}')
        self.precision = precision
        self.architecture = architecture
        if architecture == 'mlp':
            self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(729, 384), nn.ReLU(),
                                       nn.Linear(384, 128), nn.ReLU())
        elif architecture in ('edges', 'lattice'):
            inputs = 13 if architecture == 'edges' else 10
            stride = 1 if architecture == 'edges' else 2
            self.trunk = nn.Sequential(nn.Conv2d(inputs, 32, 3, stride=stride, padding=1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                       nn.Linear(32 * 81, 128), nn.ReLU())
        elif architecture == 'small':
            # Keep the original module names/shapes for old checkpoints.
            self.trunk = nn.Sequential(nn.Conv2d(9, 32, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                       nn.Linear(32 * 81, 128), nn.ReLU())
        else:
            channels, blocks = RESIDUAL_SPECS[architecture]
            self.trunk = nn.Sequential(nn.Conv2d(9, channels, 3, padding=1), nn.ReLU(),
                                       *(ResidualBlock(channels) for _ in range(blocks)), nn.Flatten(),
                                       nn.Linear(channels * 81, 128), nn.ReLU())
        self.policy = nn.Linear(128, 209)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        # Keep parameters/checkpoints FP32. BF16 applies only to CUDA forwards;
        # CPU/MPS exports remain usable. Losses and search softmax stay FP32.
        enabled = x.device.type == 'cuda' and self.precision == 'bf16'
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
            hidden = self.trunk(board_features(x, self.architecture))
            logits, value = self.policy(hidden), self.value(hidden)
        return logits.float(), value.float().tanh().squeeze(-1)


def model_from_config(config=None):
    config = config or {}
    return PolicyValue(config.get('architecture', 'small'), config.get('precision', 'fp32'))


@torch.inference_mode()
def search_batch(model, bridge, games, max_plies, simulations, rng, device,
                 explore=False, progress=None):
    """One neural batch per leaf wave, with one pending leaf per concurrent game.

    Mutable trees and legal move generation live in the CPU engine processes.
    Each request performs `simulations` NEW visits, retaining previous visits.
    A mapping from game IDs to budgets permits randomized training searches.
    """
    requests = []
    for game in games:
        state = game['state']
        position = state['position']
        if (state['winner'] != -1 or game['history'].get(key(position), 0) >= 3
                or game['ply'] >= max_plies):
            raise ValueError('Cannot search a finished game')
        request = {'id': game['id'], 'position': position, 'history': game['history'],
                   'ply': game['ply'], 'maxPlies': max_plies,
                   'simulations': simulations[game['id']] if isinstance(simulations, dict) else simulations}
        if explore:
            noise = [rng.gammavariate(0.3, 1) for _ in state['legal']]
            total = sum(noise)
            request['noise'] = [value / total for value in noise]
        requests.append(request)
    if hasattr(bridge, 'search'):
        return bridge.search(model, requests, device, progress)
    response = bridge.ask(op='mcts_begin', requests=requests)
    wave = 0
    while not response['done']:
        leaves = response['leaves']
        if not leaves:
            raise RuntimeError('MCTS made no progress')
        start = time.perf_counter()
        features = encode_batch([(leaf['position'], leaf['repetitions'], leaf['remaining']) for leaf in leaves])
        encoded = time.perf_counter()
        logits, values = model(features.to(device))
        # A single device-to-host transfer per batch; tiny legal softmaxes run on CPU.
        outputs = torch.cat((logits, values.unsqueeze(1)), dim=1).cpu()
        forwarded = time.perf_counter()
        evaluations = []
        for index, leaf in enumerate(leaves):
            probabilities = outputs[index, leaf['legal']].softmax(0).tolist()
            evaluations.append({'id': leaf['id'], 'policy': probabilities, 'value': outputs[index, -1].item()})
        finished = time.perf_counter()
        for category, elapsed in [('encoding_seconds', encoded - start),
                                  ('forward_seconds', forwarded - encoded),
                                  ('policy_seconds', finished - forwarded)]:
            bridge.stats[category] = bridge.stats.get(category, 0) + elapsed
        bridge.stats['inference_seconds'] += finished - start
        bridge.stats['inference_batches'] += 1
        bridge.stats['inference_positions'] += len(leaves)
        response = bridge.ask(op='mcts_step', evaluations=evaluations)
        wave += 1
        if progress:
            progress(wave)
    bridge.stats['reused_visits'] += response['reused']
    return {identifier: torch.tensor(policy) for identifier, policy in response['policies'].items()}


def search(model, bridge, state, history, ply, max_plies, simulations, rng, device,
           explore=False, progress=None, tree_id=None):
    identifier = tree_id or 'single'
    if tree_id is None:
        bridge.ask(op='drop', ids=[identifier])
    game = {'id': identifier, 'state': state, 'history': history, 'ply': ply}
    return search_batch(model, bridge, [game], max_plies, simulations, rng, device,
                        explore, progress)[identifier]


def choose(policy, rng, sample):
    if sample:
        return rng.choices(range(209), weights=policy.tolist(), k=1)[0]
    return int(policy.argmax())


def update(model, optimizer, replay, batch_size, rng, device, sampling='uniform'):
    batch = sample_replay(replay, batch_size, rng, sampling == 'task')
    x = torch.stack([row[0] for row in batch]).to(device)
    values = torch.tensor([row[2] for row in batch], device=device)
    logits, predictions = model(x)
    # Opponent turns provide value supervision only (None policy). Average
    # policy loss over learner rows so adding value rows does not halve it.
    policy_indices = [i for i, row in enumerate(batch) if row[1] is not None]
    if policy_indices:
        targets = torch.stack([batch[i][1] for i in policy_indices]).to(device)
        policy_loss = -(targets * logits[policy_indices].log_softmax(1)).sum(1).mean()
    else:
        policy_loss = predictions.new_zeros(())
    value_loss = (predictions - values).square().mean()
    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return {'policy': policy_loss.item(), 'value': value_loss.item()}
