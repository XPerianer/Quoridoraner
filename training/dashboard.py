"""Local dashboard API and built frontend: npm run dashboard."""
import argparse
import atexit
from collections import Counter, OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import threading
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'training' / 'runs'
LOCK = threading.Lock()
MODELS = OrderedDict()
BRIDGE = None
SUMMARIES = {}


def run_path(name):
    path = (RUNS / name).resolve()
    if not path.is_relative_to(RUNS.resolve()) or not path.is_dir():
        raise ValueError('Unknown run')
    return path


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def describe(path, detail=False):
    files = [p for p in path.iterdir() if p.is_file()]
    summary = read(path / 'summary.json', None)
    if summary is None:
        from curriculum import summarize
        metrics = path / 'metrics.json'
        stamp = metrics.stat().st_mtime_ns if metrics.exists() else 0
        cache_key = str(path)
        if SUMMARIES.get(cache_key, (None,))[0] != stamp:
            SUMMARIES[cache_key] = (stamp, summarize(read(metrics, [])))
        summary = SUMMARIES[cache_key][1]
    result = dict(id=str(path.relative_to(RUNS)), config=read(path / 'config.json', {}),
                  summary=summary, status=read(path / 'status.json', None),
                  bots=[m for m in ('selfplay', 'negamax') if (path / f'{m}.pt').exists()],
                  updated=max((p.stat().st_mtime for p in files), default=0),
                  performance=read(path / 'performance.json', {}))
    if detail:
        result['losses'] = read(path / 'losses.json', [])
    return result


def bot_move(data):
    global BRIDGE
    import torch
    from bridge import create_bridge
    from core import model_from_config, search, choose, key
    path = run_path(data['run'])
    mode = data['mode']
    if mode not in ('selfplay', 'negamax'):
        raise ValueError('Unknown bot')
    simulations = int(data.get('simulations', 64))
    if not 1 <= simulations <= 512:
        raise ValueError('Simulations must be between 1 and 512')
    positions = data['positions']
    if not isinstance(positions, list) or not 1 <= len(positions) <= 201:
        raise ValueError('Invalid game history')
    with LOCK:
        torch.set_num_threads(1)
        checkpoint = path / f'{mode}.pt'
        stamp = checkpoint.stat().st_mtime_ns
        cache_key = (str(checkpoint), stamp)
        if cache_key not in MODELS:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = model_from_config(saved.get('config'))
            model.load_state_dict(saved['model'])
            model.eval()
            MODELS[cache_key] = model
            while len(MODELS) > 4:
                MODELS.popitem(last=False)
        MODELS.move_to_end(cache_key)
        if BRIDGE is None:
            BRIDGE = create_bridge('typescript')
            atexit.register(BRIDGE.close)
        state = BRIDGE.ask(position=positions[-1])
        if state['winner'] != -1:
            raise ValueError('Game has ended')
        counts = dict(Counter(key(p) for p in positions))
        policy = search(MODELS[cache_key], BRIDGE, state, counts, len(positions)-1,
                        200, simulations, random.Random(1), 'cpu')
        return {'move': choose(policy, random.Random(1), False)}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'dist'), **kwargs)

    def respond(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith('/api/'):
            return super().do_GET()
        try:
            if parsed.path == '/api/runs':
                paths = sorted(RUNS.rglob('config.json')) if RUNS.exists() else []
                return self.respond(sorted([describe(p.parent) for p in paths], key=lambda r: r['updated'], reverse=True))
            if parsed.path == '/api/run':
                return self.respond(describe(run_path(parse_qs(parsed.query)['id'][0]), True))
            self.respond({'error': 'Not found'}, 404)
        except (ValueError, KeyError, OSError) as exc:
            self.respond({'error': str(exc)}, 400)

    def do_POST(self):
        if self.path != '/api/move':
            return self.respond({'error': 'Not found'}, 404)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 100000:
                raise ValueError('Invalid request size')
            self.respond(bot_move(json.loads(self.rfile.read(length))))
        except Exception as exc:
            self.respond({'error': str(exc)}, 400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'Dashboard: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
