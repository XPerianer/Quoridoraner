"""Export a trusted AZ ResNet checkpoint and native-reference browser fixtures.

Install optional dependencies with pip install -r training/requirements-web.txt.
Checkpoint loading uses pickle: only export checkpoints you trust.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

import az


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_position(p):
    h, v, state = map(int, p)
    return dict(pawns=[state & 127, (state >> 7) & 127],
                wallsLeft=[(state >> 14) & 15, (state >> 18) & 15],
                hWalls=[i for i in range(64) if h >> i & 1],
                vWalls=[i for i in range(64) if v >> i & 1], turn=(state >> 22) & 1)


def reference_rows(count=16):
    """Deterministic legal positions, including both turns and varied wall layouts."""
    rng = random.Random(731)
    engine = az.Engine(1)
    rows = []
    try:
        position = az.START
        for i in range(count):
            if az.winner(position) != -1:
                position = az.START
            repetitions = i % 3 + 1
            remaining = max(0., 1. - i / max(count, 1))
            features = engine.features([position], [repetitions], [remaining], [False])[0]
            legal = engine.legal(position)
            turn = unpack_position(position)['turn']
            rows.append(dict(position=unpack_position(position), repetitions=repetitions,
                             remaining=remaining, features=features.reshape(-1).tolist(),
                             legal=legal, canonicalLegal=engine.canonical[turn, 0, legal].tolist()))
            position = engine.advance(position, rng.choice(legal))
        return rows
    finally:
        engine.close()


def export_model(checkpoint, destination, bot_id, name, simulations=64, fixtures=16):
    import onnx
    import onnxruntime as ort

    if not bot_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in bot_id):
        raise ValueError('Bot ID must contain only lowercase letters, digits, hyphens or underscores')
    if simulations not in (8, 16, 32, 64, 128, 256, 400, 512) or fixtures < 2:
        raise ValueError('Use a supported arena simulation budget (8/16/32/64/128/256/400/512) and at least two fixtures')
    torch.set_num_threads(1)
    model, _ = az.load_model(checkpoint, 'cpu')
    if not isinstance(model, az.Net):
        raise ValueError('Browser exporter currently supports AZ ResNets only')
    if model.config['planes'] not in (10, 19):
        raise ValueError('Only 10-plane and 19-plane AZ feature contracts are supported')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = reference_rows(fixtures)
    errors = dict(logits=0., value=0.)
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / 'model.onnx'
        # Fixed batch one and full native input contract. model_inputs includes
        # the compact-plane selection in the graph for 10-plane checkpoints.
        with torch.no_grad():
            torch.onnx.export(model, torch.zeros(1, 19, 9, 9), str(staged),
                              input_names=['features'], output_names=['logits', 'value'],
                              opset_version=17, dynamo=False, external_data=False)
        onnx.checker.check_model(str(staged))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(staged), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        for row in rows:
            x = np.asarray(row['features'], np.float32).reshape(1, 19, 9, 9)
            with torch.no_grad():
                expected = [a.numpy() for a in model(torch.from_numpy(x))]
            actual = session.run(['logits', 'value'], {'features': x})
            for key, a, b in zip(('logits', 'value'), expected, actual):
                np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5)
                errors[key] = max(errors[key], float(np.abs(a - b).max()))
            row['logits'] = expected[0][0].tolist()
            row['value'] = float(expected[1][0])
        digest = checksum(staged)
        filename = f'{bot_id}-{digest[:12]}.onnx'
        target = destination / filename
        target.write_bytes(staged.read_bytes())
    metadata = dict(id=bot_id, name=name, url=f'models/{filename}', sha256=digest,
                    bytes=target.stat().st_size, planes=19, simulations=simulations,
                    config=model.config, contract='az-v1', input='features',
                    outputs=dict(logits='logits', value='value'))
    report = dict(**metadata, sourceSha256=checksum(checkpoint), fixtureCount=len(rows),
                  parityMaxAbsoluteError=errors, pytorch=torch.__version__,
                  onnx=onnx.__version__, onnxruntime=ort.__version__)
    (destination / f'{bot_id}.json').write_text(json.dumps(report, indent=2) + '\n')
    (destination / f'{bot_id}-fixtures.json').write_text(json.dumps(dict(contract='az-v1', rows=rows)) + '\n')
    manifest = destination.parent / 'bots.json'
    data = json.loads(manifest.read_text()) if manifest.exists() else dict(version=1, bots=[])
    data['bots'] = [b for b in data['bots'] if b['id'] != bot_id] + [metadata]
    manifest.write_text(json.dumps(data, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, default=Path('public/models'))
    parser.add_argument('--id', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--fixtures', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(export_model(args.checkpoint, args.out, args.id, args.name,
                                 args.simulations, args.fixtures), indent=2))


if __name__ == '__main__':
    main()
