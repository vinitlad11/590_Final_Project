"""Download a named public release and prepare a human-reviewed 50-question pilot.

No model calls. No automatic interpretation of quizbowl accept/reject directives.
Run from the project root. See docs/DATA_SETUP.md.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import argparse
import csv
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/qanta2021'
READY = ROOT / 'data/processed/qanta2021'
# The revision returned by the public catalog's 2021 development-file link.
DATA_REVISION = '12d2ce9c99e55643c0116f4623d3ac72656f7f7d'
DATA_REPO = 'https://huggingface.co/datasets/TasnimKabir12/qanta/resolve/'
RELEASE = '2021.12.20'
COUNTS = {'train': 30, 'calibration': 10, 'tune': 5, 'test': 5}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def raw_questions(path):
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = obj.get('questions') if isinstance(obj, dict) else obj
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f'{path}: expected a nonempty QANTA questions array, not an HTML download page')
    return rows


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ('train', 'dev'):
        name = f'qanta.{split}.{RELEASE}.json'
        path = RAW / name
        url = f'{DATA_REPO}{DATA_REVISION}/{name}?download=true'
        if not path.exists():
            tmp = path.with_suffix('.json.part')
            with urlopen(Request(url, headers={'User-Agent':'AdaptiveQA-StudentPilot/1.0'}), timeout=90) as response:
                with tmp.open('wb') as f:
                    shutil.copyfileobj(response, f)
            raw_questions(tmp)
            tmp.replace(path)
        rows = raw_questions(path)
        records.append({'file':name, 'expected_public_url':url, 'sha256':sha(path),
                        'bytes':path.stat().st_size, 'questions':len(rows)})
        print(f'{name}: {len(rows)} questions; SHA256 {sha(path)}', flush=True)
    save_json(RAW / 'download_manifest.json', {
        'release': RELEASE, 'mirror_revision':DATA_REVISION,
        'catalog':'https://qanta-org.github.io/data-and-code/',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(), 'files':records,
        'note':'Existing local files are validated and hashed, not assumed identical to the mirror.'})


def candidate_rows(rows, source, seen_ids, seen_texts):
    result, skipped = [], 0
    for row in rows:
        text = row.get('text', row.get('question', ''))
        qid = row.get('qanta_id', row.get('qid', row.get('id')))
        fold = str(row.get('fold') or source).lower()
        media = any(str(row.get(k, '')).lower() in ('true','1','yes') for k in ('has_image','has_audio'))
        if (not isinstance(text, str) or not text.strip() or qid is None or media
                or any(x in text.lower() for x in ('<img','[image','<audio','[audio'))
                or 'test' in fold or (source == 'train' and ('dev' in fold or 'val' in fold))):
            skipped += 1
            continue
        text = ' '.join(text.split())
        full_id = f'qanta2021:{qid}'
        norm = text.casefold()
        if full_id in seen_ids or norm in seen_texts:
            skipped += 1
            continue
        seen_ids.add(full_id); seen_texts.add(norm)
        answer = str(row.get('answer') or '')
        simple = bool(answer.strip()) and not re.search(r'accept|reject|prompt|\[|\{|<|\(|;|\bor\b', answer, re.I)
        result.append({'question_id':full_id, 'source_id':str(qid), 'source_fold':fold,
                       'source_file':source, 'question_text':text, 'source_answerline':answer,
                       'wikipedia_page':str(row.get('page') or ''),
                       'suggested_answers': [answer.strip()] if simple else []})
    return sorted(result, key=lambda r: hashlib.sha256(('42:' + r['question_id']).encode()).hexdigest()), skipped


def select():
    selection = READY / 'selection.json'
    review = READY / 'answer_review.csv'
    if selection.exists() or review.exists():
        raise FileExistsError('Selection/review already exists; preserve your review. Use it or start a separately named project copy.')
    seen_ids, seen_texts, pool, audits = set(), set(), {}, {}
    for source in ('train', 'dev'):
        path = RAW / f'qanta.{source}.{RELEASE}.json'
        rows = raw_questions(path)
        pool[source], skipped = candidate_rows(rows, source, seen_ids, seen_texts)
        audits[source] = {'file':path.name, 'sha256':sha(path), 'rows':len(rows), 'eligible':len(pool[source]), 'skipped':skipped}
    if len(pool['train']) < 30 or len(pool['dev']) < 20:
        raise ValueError('Need at least 30 eligible training and 20 eligible development questions')
    chosen = []
    offsets = {'train':0, 'dev':0}
    for split, count in COUNTS.items():
        source = 'train' if split == 'train' else 'dev'
        start = offsets[source]
        chosen.extend({**row, 'split':split} for row in pool[source][start:start + count])
        offsets[source] += count
    save_json(selection, {'dataset':'QANTA_2021_12_20_PILOT', 'seed':42, 'counts':COUNTS,
                         'test_role':'pilot diagnostic only; these 50 qids cannot be a later untouched holdout',
                         'source_audit':audits, 'questions':chosen})
    fields = ['question_id','source_fold','split','question_text','source_answerline','wikipedia_page',
              'accepted_answers_json','approved']
    with review.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow({**{k:row[k] for k in fields if k in row},
                'accepted_answers_json':json.dumps(row['suggested_answers'], ensure_ascii=False), 'approved':''})
    print(f'Review all 50 rows in {review}. Edit accepted_answers_json and approved only. Then run freeze.')


def freeze():
    selected = json.loads((READY / 'selection.json').read_text(encoding='utf-8'))
    original = {r['question_id']:r for r in selected['questions']}
    for source, info in selected['source_audit'].items():
        if sha(RAW / info['file']) != info['sha256']:
            raise ValueError('Raw source changed after selection')
    with (READY / 'answer_review.csv').open(encoding='utf-8-sig', newline='') as f:
        review = list(csv.DictReader(f))
    if len(review) != 50 or len({r['question_id'] for r in review}) != 50 or set(original) != {r['question_id'] for r in review}:
        raise ValueError('Review must contain the original 50 distinct qids exactly once')
    questions = []
    for row in review:
        qid = row['question_id']; source = original[qid]
        for key in ('source_fold','split','question_text','source_answerline','wikipedia_page'):
            if row[key] != source[key]:
                raise ValueError(f'{qid}: immutable {key} was edited; keep selection and review aligned')
        if row['approved'].strip().lower() != 'yes':
            raise ValueError(f'{qid}: review accepted aliases, then mark approved=yes')
        answers = json.loads(row['accepted_answers_json'])
        if (not isinstance(answers,list) or not answers or any(not isinstance(a,str) or not a.strip() for a in answers)
                or any(re.search(r'\b(accept|reject|prompt on)\b|[\[\]{}]',a,re.I) for a in answers)):
            raise ValueError(f'{qid}: enter a JSON list of accepted answer strings only')
        questions.append({'question_id':qid, 'question_text':source['question_text'],
                          'answers':list(dict.fromkeys(a.strip() for a in answers)),
                          'fold':source['source_fold'], 'has_image':False, 'has_audio':False})
    # A small subset of exactly the same reviewed pilot; qid/prefix seeds reuse cached calls.
    mini_counts = {'train':6,'calibration':2,'tune':1,'test':1}
    mini_ids = {r['question_id'] for split,n in mini_counts.items()
                for r in [r for r in selected['questions'] if r['split'] == split][:n]}
    outputs = {}
    for size, rows in ((50,questions),(10,[q for q in questions if q['question_id'] in mini_ids])):
        path = READY / f'pilot{size}.jsonl'
        text = ''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in rows)
        mapping = {q['question_id']:original[q['question_id']]['split'] for q in rows}
        split_path = READY / f'pilot{size}_splits.json'
        if path.exists() and path.read_text(encoding='utf-8') != text:
            raise ValueError('Frozen aliases changed. Preserve the old run and use a separately named dataset/config.')
        path.write_text(text, encoding='utf-8')
        save_json(split_path, mapping)
        outputs[path.name] = sha(path); outputs[split_path.name] = sha(split_path)
    save_json(READY / 'frozen_manifest.json', {'dataset':selected['dataset'], 'files':outputs,
        'review_sha256':sha(READY/'answer_review.csv'), 'selection_sha256':sha(READY/'selection.json'),
        'test_role':selected['test_role'], 'scoring':'reviewed alias exact match; answer F1 secondary'})
    print('Frozen pilot10 and pilot50 with question-level splits. Do not alter aliases after seeing model predictions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download','select','freeze'])
    args = parser.parse_args()
    {'download':download, 'select':select, 'freeze':freeze}[args.command]()
