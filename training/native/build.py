"""Build the optional Node-API C++ training engine; no downloaded dependencies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug', action='store_true', help='Build with assertions and debug symbols')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    node = shutil.which('node')
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not node or not compiler:
        raise SystemExit('Install Node and a C++17 compiler (macOS: Xcode Command Line Tools).')
    version = subprocess.check_output([node, '-p', 'process.versions.node'], text=True).strip()
    candidates = [Path(os.environ.get('NODE_INCLUDE', '/nonexistent')),
                  Path(node).resolve().parent.parent / 'include/node',
                  Path.home() / f'Library/Caches/node-gyp/{version}/include/node',
                  Path.home() / f'.cache/node-gyp/{version}/include/node',
                  Path('/usr/include/node'), Path('/usr/local/include/node')]
    headers = next((p for p in candidates if (p / 'node_api.h').exists()), None)
    if not headers:
        raise SystemExit('Node headers were not found. Set NODE_INCLUDE to the directory containing node_api.h.')
    output = root / 'build'
    output.mkdir(exist_ok=True)
    flags = ['-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-DNAPI_VERSION=8']
    flags += ['-O0', '-g'] if args.debug else ['-O3', '-DNDEBUG']
    if sys.platform == 'darwin':
        flags += ['-undefined', 'dynamic_lookup']
    command = [compiler, *flags, '-I', str(headers), str(root / 'addon.cpp'), '-o', str(output / 'engine.node')]
    subprocess.run(command, check=True)
    (output / 'build.json').write_text(json.dumps({'node': version, 'compiler': compiler, 'flags': flags}, indent=2)+'\n')
    print(f'Built {output / "engine.node"}')


if __name__ == '__main__':
    main()
