"""Export source, frozen inputs, caches and run records to a separate private ZIP."""
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_qa.io import file_hash
import json


def export(root, destination):
    root, destination = root.resolve(), destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError('Backup directory must be outside the project')
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    target = destination / f'adaptiveqa-stage03-private-backup-{stamp}.zip'
    allowed = ('src', 'tests', 'scripts', 'configs', 'notebooks', 'data/fixtures', 'data/processed',
               'data/cache_stage03_answer_v2', 'results')
    files = set()
    for name in allowed:
        files.update(p for p in (root / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts)
    for name in ('README.md', '.gitignore', 'pyproject.toml', 'requirements.txt', 'requirements-tested.txt',
                 'data/raw/evidence_corpus_qanta2021.jsonl', 'data/raw/wikipedia_pilot/corpus_manifest.json',
                 'data/raw/wikipedia_pilot/title_selection.json', 'data/raw/qanta2021/download_manifest.json'):
        if (root / name).is_file():
            files.add(root / name)
    hashes = {str(p.relative_to(root)).replace('\\', '/'): file_hash(p) for p in sorted(files)}
    with ZipFile(target, 'x', ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, 'adaptive-qa/' + p.relative_to(root).as_posix())
        archive.writestr('backup_inventory.json', json.dumps({'created_utc': stamp, 'files': hashes,
            'scope': 'Private Stage 3 source, frozen inputs, generated outputs and caches. Original Stage 2 backup remains separate.'}, indent=2))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(export(Path(__file__).resolve().parents[1], args.destination))
