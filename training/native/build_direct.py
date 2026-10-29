"""Build the in-process C++ training library (no Node/Python headers needed)."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'build' / ('direct.dylib' if sys.platform == 'darwin' else 'direct.so')


def build(debug=False):
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('Install a C++17 compiler (macOS: Xcode Command Line Tools).')
    OUTPUT.parent.mkdir(exist_ok=True)
    flags = ['-std=c++17', '-shared', '-fPIC', '-pthread', '-Wall', '-Wextra', '-ffp-contract=off']
    flags += ['-O0', '-g'] if debug else ['-O3', '-DNDEBUG']
    temporary = OUTPUT.with_name(OUTPUT.name + f'.{os.getpid()}.tmp')
    try:
        subprocess.run([compiler, *flags, str(ROOT / 'direct.cpp'), '-o', str(temporary)], check=True)
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    return OUTPUT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug', action='store_true')
    print(build(parser.parse_args().debug))
