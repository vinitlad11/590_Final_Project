"""Reuse reviewed, frozen pilot data without selecting new questions."""
from pathlib import Path
import argparse
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_qa.io import file_hash, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path('data/processed/qanta2021')
CORPUS = Path('data/raw/evidence_corpus_qanta2021.jsonl')
CORPUS_MANIFEST = Path('data/raw/wikipedia_pilot/corpus_manifest.json')
FROZEN_FILES = ('pilot10.jsonl', 'pilot10_splits.json', 'pilot50.jsonl', 'pilot50_splits.json')
IMMUTABLE_PROCESSED = (*FROZEN_FILES, 'frozen_manifest.json', 'answer_review.csv', 'selection.json')


def validate_frozen(root):
    root = Path(root)
    folder = root / PROCESSED
    frozen = json.loads((folder / 'frozen_manifest.json').read_text(encoding='utf-8'))
    if set(frozen['files']) != set(FROZEN_FILES):
        raise ValueError('Frozen manifest must identify exactly the 10/50-question data and split files')
    for name in FROZEN_FILES:
        if file_hash(folder / name) != frozen['files'][name]:
            raise ValueError(f'Frozen file changed: {name}')
    for name, key in [('answer_review.csv', 'review_sha256'), ('selection.json', 'selection_sha256')]:
        if file_hash(folder / name) != frozen[key]:
            raise ValueError(f'Review/selection changed after freezing: {name}')
    records = {}
    splits = {}
    for size in (10, 50):
        rows = read_jsonl(folder / f'pilot{size}.jsonl')
        records[size] = {r['question_id']: r for r in rows}
        splits[size] = json.loads((folder / f'pilot{size}_splits.json').read_text())
        if len(rows) != size or len(records[size]) != size or set(splits[size]) != set(records[size]):
            raise ValueError(f'Expected {size} unique questions and matching splits')
    for qid, row in records[10].items():
        if records[50].get(qid) != row or splits[10][qid] != splits[50].get(qid):
            raise ValueError(f'10-question subset changed: {qid}')
    corpus_manifest = json.loads((root / CORPUS_MANIFEST).read_text())
    if file_hash(root / CORPUS) != corpus_manifest['corpus_sha256']:
        raise ValueError('Frozen evidence corpus changed')
    return {'status': 'PASSED', 'questions': 50, 'setup_subset': 10,
            'dataset_hashes': frozen['files'], 'corpus_sha256': corpus_manifest['corpus_sha256'],
            'passages': corpus_manifest['passages'],
            'note': 'All 50 questions are development pilot data; keep a separate final holdout.'}


def import_frozen(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Stage 3 uses a separate Drive project directory')
    result = validate_frozen(source)
    immutable_paths = [PROCESSED / name for name in IMMUTABLE_PROCESSED]
    corpus_paths = [CORPUS, CORPUS_MANIFEST]
    for optional in ('data/raw/wikipedia_pilot/title_selection.json', 'data/raw/qanta2021/download_manifest.json'):
        if (source / optional).exists():
            corpus_paths.append(Path(optional))
    # Immutable processed question/split files must always match Stage 2 exactly.
    for relative in immutable_paths:
        target = destination / relative
        if target.exists() and file_hash(target) != file_hash(source / relative):
            raise ValueError(f'Destination already has different question data: {relative}; preserve it')
    for relative in immutable_paths + corpus_paths:
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
    result['source_project'] = str(source)
    result['copied_files'] = {str(p): file_hash(destination / p) for p in immutable_paths + corpus_paths if (destination / p).exists()}
    write_json(destination / 'results/stage2_data_import.json', result)
    validate_frozen(destination)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-project', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        result = validate_frozen(ROOT)
    elif args.from_project:
        result = import_frozen(args.from_project, ROOT)
    else:
        parser.error('Supply --from-project or --check-only')
    print(json.dumps(result, indent=2))
