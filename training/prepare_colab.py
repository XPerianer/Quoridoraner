"""Bundle current training source (including uncommitted files), without runs or secrets."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def bundle(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / 'training').glob('*.py'))
    paths += sorted((ROOT / 'training' / 'native').glob('*.py'))
    paths += sorted((ROOT / 'training' / 'native').glob('*.cpp'))
    paths += sorted((ROOT / 'training' / 'native').glob('*.hpp'))
    paths += sorted((ROOT / 'training').glob('*.md'))
    paths += sorted((ROOT / 'training').glob('*.sh'))
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'training/runs/colab-source.zip')
    print(bundle(parser.parse_args().out))
