"""Download pinned opponent code/weights without installing or executing them."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'experiments/training/external-bots'
SOURCES = {
    'borowski': {
        'repo': 'danielborowski/quoridor-ai',
        'commit': '4723b0eae6e524c232bcfeb9fc49eb8742daafb6',
        'files': ['js/path/graph.js', 'js/path/astar.js', 'js/board.js',
                  'js/ai.js', 'js/placewall.js', 'display.html', 'README.md'],
    },
    'quoridor-ai': {
        'repo': 'gorisanson/quoridor-ai',
        'commit': 'e15d6e7825e6f7c2f3a41de0ecef1af7aa290f1e',
        'files': ['src/js/game.js', 'src/js/ai.js', 'LICENSE', 'README.md'],
    },
    'SigmaQuoridor': {
        'repo': 'bartolomeo3000/SigmaQuoridor',
        'commit': '751186344fc52ad0c29bc65922e62c6fa915f006',
        'files': ['game.py', 'mcts.py', 'dual_network.py', 'LICENSE', 'README.md',
                  'runs/models_9x9_scratch/best.pt'],
    },
}


def download(job):
    name, source, file = job
    dest = ROOT / name / file
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{file}"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    tmp = dest.with_suffix(dest.suffix + '.download')
    tmp.write_bytes(data)
    tmp.replace(dest)
    return name, file, hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    manifests = {name: dict(source, sha256={}) for name, source in SOURCES.items()}
    jobs = [(n, s, f) for n, s in SOURCES.items() for f in s['files']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name, file, digest in pool.map(download, jobs):
            manifests[name]['sha256'][file] = digest
            print(f'{name}/{file}: {digest}', flush=True)
    for name, manifest in manifests.items():
        (ROOT / name / 'SOURCE.json').write_text(json.dumps(manifest, indent=2))
