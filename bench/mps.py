"""Optional batched shortest paths. Run after npm run bench:kernels.

Inputs are resident on the device during timing; packing/transfers are excluded.
80 synchronous BFS waves cover every simple path on an 81-cell board.
"""
import json
import statistics
import time
from pathlib import Path
import torch


def pack(positions, device):
    horizontal, vertical, starts, goals = [], [], [], []
    for p in positions:
        down = [[True] * 9 for _ in range(8)]
        right = [[True] * 8 for _ in range(9)]
        for w in p['hWalls']:
            y, x = divmod(w, 8)
            down[y][x] = down[y][x + 1] = False
        for w in p['vWalls']:
            y, x = divmod(w, 8)
            right[y][x] = right[y + 1][x] = False
        for player in (0, 1):
            horizontal.append(right)
            vertical.append(down)
            starts.append(p['pawns'][player])
            goals.append(0 if player == 0 else 8)
    return (torch.tensor(horizontal, device=device), torch.tensor(vertical, device=device),
            torch.tensor(starts, device=device), torch.tensor(goals, device=device))


@torch.inference_mode()
def distances(data):
    right, down, starts, goals = data
    count = starts.shape[0]
    wave = torch.zeros((count, 81), dtype=torch.bool, device=starts.device)
    wave.scatter_(1, starts[:, None], True)
    wave = wave.reshape(count, 9, 9)
    result = torch.full((count,), -1, dtype=torch.int32, device=starts.device)
    index = torch.arange(count, device=starts.device)
    for step in range(81):
        reached = wave[index, goals].any(dim=1)
        result = torch.where((result < 0) & reached, step, result)
        if step == 80:
            break
        nxt = wave.clone()
        nxt[:, :, 1:] |= wave[:, :, :-1] & right
        nxt[:, :, :-1] |= wave[:, :, 1:] & right
        nxt[:, 1:, :] |= wave[:, :-1, :] & down
        nxt[:, :-1, :] |= wave[:, 1:, :] & down
        wave = nxt
    return result


def main():
    source = json.loads(Path('bench/positions.json').read_text())
    # Include unreachable and terminal positions in validation.
    terminal = dict(source[0], pawns=[4, 76])
    validation = pack(source + [terminal], 'cpu')
    cpu_expected = distances(validation).tolist()
    assert cpu_expected[-2:] == [0, 0]
    assert cpu_expected[:-2] == [d for p in source for d in p['expected']]
    blocked = pack([source[0]], 'cpu')
    blocked[1].fill_(False)
    assert distances(blocked).tolist() == [-1, -1]
    available = torch.backends.mps.is_available()
    report = {'torch': torch.__version__, 'mpsAvailable': available, 'rows': []}
    for count in (1, 81, 1024):
        positions = [source[i % len(source)] for i in range(count)]
        expected = distances(pack(positions, 'cpu')).tolist()
        for device in (['cpu', 'mps'] if available else ['cpu']):
            data = pack(positions, device)
            assert distances(data).cpu().tolist() == expected
            if device == 'mps':
                assert distances(pack(source + [terminal], device)).cpu().tolist() == cpu_expected
            samples = []
            for repeat in range(7):
                if device == 'mps': torch.mps.synchronize()
                start = time.perf_counter()
                distances(data)
                if device == 'mps': torch.mps.synchronize()
                ms = (time.perf_counter() - start) * 1000
                if repeat >= 2: samples.append(ms)
            row = {'device': device, 'positions': count, 'medianMs': statistics.median(samples), 'samplesMs': samples}
            report['rows'].append(row)
            print(row, flush=True)
    Path('bench/mps-results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
