"""Freeze up to 500 Wikipedia articles selected from QANTA training data only.

No questions, answers, or held-out page lists are indexed. Actual article text is
retrieved through the MediaWiki API, cached, cleaned and split into bounded chunks.
"""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import time
from prepare_qanta_public import RAW, ROOT, RELEASE, raw_questions, save_json, sha


def build(contact, limit=500):
    import mwparserfromhell
    if not 1 <= limit <= 500:
        raise ValueError('Use 1 to 500 articles for this pilot')
    if '@' not in contact and not contact.startswith('https://'):
        raise ValueError('Supply a real contact email or project URL for the Wikipedia User-Agent')
    source = RAW / f'qanta.train.{RELEASE}.json'
    # No dev file is opened: topic selection cannot consult held-out answer pages.
    rows = raw_questions(source)
    titles = {r['page'].replace('_',' ') for r in rows if isinstance(r.get('page'),str) and r['page'].strip()
              and 'train' in str(r.get('fold','train')).lower()}
    titles = sorted(titles, key=lambda t: hashlib.sha256(('42:'+t).encode()).hexdigest())[:limit]
    cache = ROOT / 'data/raw/wikipedia_pilot'
    cache.mkdir(parents=True, exist_ok=True)
    selection_path = cache / 'title_selection.json'
    selection = {'source_train_sha256':sha(source), 'seed':42, 'article_cap':limit, 'titles':titles}
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Corpus selection changed. Use a new corpus directory/experiment instead of overwriting it.')
    save_json(selection_path, selection)
    passages, records = [], []
    unique_revisions = set()
    for index, title in enumerate(titles, 1):
        key = hashlib.sha256(title.encode()).hexdigest()
        path = cache / (key + '.json')
        if not path.exists():
            params = {'action':'query','format':'json','formatversion':2,'redirects':1,
                      'prop':'revisions','rvprop':'ids|timestamp|content','rvslots':'main',
                      'titles':title, 'maxlag':5}
            request = Request('https://en.wikipedia.org/w/api.php?' + urlencode(params),
                headers={'User-Agent':f'AdaptiveQAStudentPilot/1.0 ({contact})'})
            with urlopen(request, timeout=60) as response:
                raw = json.load(response)
            if 'error' in raw:
                raise RuntimeError(f"Wikipedia returned {raw['error']}; wait and rerun, preserving the cache")
            save_json(path, raw)
            time.sleep(0.2)
        raw = json.loads(path.read_text(encoding='utf-8'))
        page = raw['query']['pages'][0]
        if 'missing' in page or not page.get('revisions'):
            records.append({'requested_title':title, 'status':'missing', 'raw_sha256':sha(path)})
            continue
        rev = page['revisions'][0]
        revision_id = rev['revid']
        if revision_id in unique_revisions:
            continue
        unique_revisions.add(revision_id)
        wikitext = rev['slots']['main']['content']
        code = mwparserfromhell.parse(wikitext)
        # Citation text, tables, and templates are not evidence paragraphs.
        for tag in list(code.filter_tags(recursive=True)):
            if str(tag.tag).lower() in {'ref','table','gallery','math','score'}:
                try: code.remove(tag)
                except ValueError: pass
        text = code.strip_code(normalize=True, collapse=True)
        words = text.split()[:600]
        url = f'https://en.wikipedia.org/w/index.php?oldid={revision_id}'
        for start in range(0, len(words), 150):
            chunk = words[start:start+150]
            if len(chunk) < 25:
                continue
            passages.append({'document_id':f'enwiki:{revision_id}:{start//150}',
                             'text':' '.join(chunk), 'title':page['title'], 'source':url,
                             'revision_id':revision_id, 'revision_timestamp':rev['timestamp'],
                             'license':'CC BY-SA 4.0; see source page history and notices',
                             'license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
                             'processing':'wikitext stripped; first 600 whitespace words; 150-word chunks'})
        records.append({'requested_title':title,'resolved_title':page['title'], 'revision_id':revision_id,
                        'source':url,'status':'included' if words else 'empty','raw_sha256':sha(path)})
        if index % 25 == 0 or index == len(titles):
            print(f'{index}/{len(titles)} article requests; {len(passages)} passages', flush=True)
    if not passages:
        raise ValueError('No usable passages; inspect API responses before proceeding')
    out = ROOT / 'data/raw/evidence_corpus_qanta2021.jsonl'
    content = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in passages)
    if out.exists() and out.read_text(encoding='utf-8') != content:
        raise ValueError('Frozen evidence corpus differs; preserve it and use a separately named experiment')
    out.write_text(content, encoding='utf-8')
    save_json(cache/'corpus_manifest.json', {'source_train_sha256':sha(source), 'corpus_sha256':sha(out),
        'article_cap':limit, 'passages':len(passages), 'max_passages':limit*4, 'max_words_per_passage':150,
        'selection_rule':'seeded hash of unique Wikipedia page titles in the QANTA training file only',
        'limitations':'training-topic selection can favor train topics and limit holdout retrieval coverage; current Wikipedia revisions, not a 2021 snapshot',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(), 'articles':records})
    print(f'Frozen {len(passages)} independent Wikipedia passages at {out}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contact', default=os.environ.get('WIKI_CONTACT',''))
    p.add_argument('--limit', type=int, default=500)
    a=p.parse_args()
    build(a.contact, a.limit)
