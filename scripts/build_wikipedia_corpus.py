"""Freeze up to 500 Wikipedia articles selected strictly from QANTA training data only.

No pilot questions, development/test questions, or answer-key files (such as selection.json,
pilot10.jsonl, pilot50.jsonl, or qanta.dev) are ever opened or consulted.
"""
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import time
from prepare_qanta_public import RAW, ROOT, RELEASE, raw_questions, save_json, sha


def canonical_title(raw_title):
    """Normalize QANTA page string formatting (underscores and whitespace only)."""
    return " ".join(str(raw_title or "").replace("_", " ").split())


def clean_wikitext(wikitext, page_title):
    import mwparserfromhell
    code = mwparserfromhell.parse(wikitext)
    # Preserve inline display text inside {{lang|...}} or {{transl|...}} before stripping templates.
    for tpl in list(code.filter_templates(recursive=True)):
        name = str(tpl.name).strip().lower()
        if name in {"lang", "transl", "nihongo", "lit"} and tpl.params:
            positional = [str(p.value).strip() for p in tpl.params if not p.showkey and str(p.value).strip()]
            replacement = positional[-1] if positional else ""
            try:
                code.replace(tpl, replacement)
            except ValueError:
                pass
    for tag in list(code.filter_tags(recursive=True)):
        if str(tag.tag).lower() in {"ref", "table", "gallery", "math", "score"}:
            try:
                code.remove(tag)
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"\b(?:thumb|frame|frameless|left|right|center|upright(?:=[\d.]+)?|\d+x?\d*px)\|", " ", text)
    text = " ".join(text.split())
    if page_title and not text.casefold().startswith(page_title.casefold()):
        text = f"{page_title}. {text}"
    return text


def select_titles_from_train_only(limit=500, train_path=None):
    """Select top Wikipedia article titles using ONLY qanta.train.2021.12.20.json.

    Procedure:
    1. Read eligible training-fold rows from qanta.train.2021.12.20.json (fold == 'train').
    2. Count question frequency per canonical Wikipedia page title in the training split.
    3. Rank titles by descending training-question frequency, tie-broken deterministically
       by sha256('42:' + title), and return the top `limit` titles.
    """
    source = Path(train_path) if train_path else RAW / f"qanta.train.{RELEASE}.json"
    if not source.exists():
        raise FileNotFoundError(f"Training file not found: {source}")
    rows = raw_questions(source)
    counts = Counter()
    display_by_key = {}
    for r in rows:
        fold = str(r.get("fold") or "train").lower()
        if "train" not in fold or "dev" in fold or "val" in fold or "test" in fold:
            continue
        raw_page = r.get("page")
        if not isinstance(raw_page, str) or not raw_page.strip():
            continue
        title = canonical_title(raw_page)
        if not title:
            continue
        key = title.casefold()
        counts[key] += 1
        if key not in display_by_key:
            display_by_key[key] = title

    ranked_keys = sorted(
        counts.keys(),
        key=lambda k: (-counts[k], hashlib.sha256(("42:" + display_by_key[k]).encode()).hexdigest()),
    )[:limit]
    titles = [display_by_key[k] for k in ranked_keys]
    freq_map = {display_by_key[k]: counts[k] for k in ranked_keys}
    return titles, sha(source), freq_map


def fetch_batch(titles_batch, cache, contact, extra_cache_dirs=(), max_retries=4):
    """Fetch a batch of titles from MediaWiki API (up to 25 at once) with per-title caching."""
    results = {}
    missing_to_fetch = []
    for t in titles_batch:
        key = hashlib.sha256(t.encode()).hexdigest()
        path = cache / (key + ".json")
        if path.exists():
            results[t] = (json.loads(path.read_text(encoding="utf-8")), path)
            continue
        found_extra = False
        for extra in extra_cache_dirs:
            cand = Path(extra) / (key + ".json")
            if cand.exists():
                raw = json.loads(cand.read_text(encoding="utf-8"))
                save_json(path, raw)
                results[t] = (raw, path)
                found_extra = True
                break
        if not found_extra:
            missing_to_fetch.append(t)

    if not missing_to_fetch:
        return results

    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main",
        "titles": "|".join(missing_to_fetch), "maxlag": 5,
    }
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    last_err = None
    batch_raw = None
    for attempt in range(max_retries):
        try:
            request = Request(url, headers={"User-Agent": f"AdaptiveQAStudentPilot/1.0 ({contact})"})
            with urlopen(request, timeout=60) as response:
                batch_raw = json.load(response)
            if "error" not in batch_raw:
                time.sleep(0.15)
                break
            last_err = f"Wikipedia returned {batch_raw['error']}"
            batch_raw = None
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.5 * (2 ** attempt))

    if batch_raw is None:
        raise RuntimeError(f"{last_err}; wait and rerun, preserving the cache")

    query_obj = batch_raw.get("query", {})
    norm_map = {n["from"]: n["to"] for n in query_obj.get("normalized", [])}
    redir_map = {r["from"]: r["to"] for r in query_obj.get("redirects", [])}
    pages_by_title = {p["title"]: p for p in query_obj.get("pages", []) if "title" in p}

    for t in missing_to_fetch:
        cur = norm_map.get(t, t)
        visited = set()
        while cur in redir_map and cur not in visited:
            visited.add(cur)
            cur = redir_map[cur]
        page_obj = pages_by_title.get(cur) or pages_by_title.get(t) or {"title": t, "missing": True}
        single_raw = {"batchcomplete": True, "query": {"pages": [page_obj]}}
        key = hashlib.sha256(t.encode()).hexdigest()
        path = cache / (key + ".json")
        save_json(path, single_raw)
        results[t] = (single_raw, path)

    return results


def build(contact, limit=500, overwrite_corpus=False, out_corpus=None, cache_subdir="wikipedia_train_top500"):
    if not 1 <= limit <= 500:
        raise ValueError("Use 1 to 500 articles for this pilot")
    if "@" not in contact and not contact.startswith("https://"):
        raise ValueError("Supply a real contact email or project URL for the Wikipedia User-Agent")

    cache = ROOT / "data/raw" / cache_subdir
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(out_corpus) if out_corpus else ROOT / "data/raw/evidence_corpus_qanta2021_train_top500.jsonl"

    extra_caches = [
        p for p in (
            ROOT / "data/raw/wikipedia_pilot",
            ROOT.parent / "adaptive-confidence-agent/data/raw/wikipedia_pilot",
            ROOT.parent / "adaptive-qa-stage03-v3/data/raw/wikipedia_pilot",
        )
        if p.is_dir() and p != cache
    ]

    titles, source_sha, freq_map = select_titles_from_train_only(limit=limit)
    selection_path = cache / "title_selection.json"
    selection = {
        "source_train_sha256": source_sha,
        "seed": 42,
        "article_cap": limit,
        "selection_procedure": "training_only_frequency_ranked_qanta_train_2021_12_20",
        "pilot_or_dev_files_consulted": False,
        "titles": titles,
    }
    if selection_path.exists() and not overwrite_corpus and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Corpus selection changed. Pass --overwrite-corpus or use a new corpus directory.")
    save_json(selection_path, selection)

    passages, records = [], []
    unique_revisions = set()
    batch_size = 20
    for b_start in range(0, len(titles), batch_size):
        batch_titles = titles[b_start:b_start + batch_size]
        fetched = fetch_batch(batch_titles, cache, contact, extra_cache_dirs=extra_caches)
        for title in batch_titles:
            raw, path = fetched[title]
            page = raw["query"]["pages"][0]
            if "missing" in page or not page.get("revisions"):
                records.append({
                    "requested_title": title,
                    "train_question_count": freq_map.get(title, 0),
                    "status": "missing",
                    "raw_sha256": sha(path),
                })
                continue
            rev = page["revisions"][0]
            revision_id = rev["revid"]
            if revision_id in unique_revisions:
                continue
            unique_revisions.add(revision_id)
            wikitext = rev["slots"]["main"]["content"]
            text = clean_wikitext(wikitext, page["title"])
            words = text.split()[:600]
            url = f"https://en.wikipedia.org/w/index.php?oldid={revision_id}"
            for start in range(0, len(words), 150):
                chunk = words[start:start + 150]
                if len(chunk) < 25:
                    continue
                chunk_text = " ".join(chunk)
                if page["title"] and not chunk_text.casefold().startswith(page["title"].casefold()):
                    chunk_text = " ".join(f"{page['title']}. {chunk_text}".split()[:150])
                passages.append({
                    "document_id": f"enwiki:{revision_id}:{start // 150}",
                    "text": chunk_text,
                    "title": page["title"],
                    "source": url,
                    "revision_id": revision_id,
                    "revision_timestamp": rev["timestamp"],
                    "license": "CC BY-SA 4.0; see source page history and notices",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "processing": "wikitext stripped with lang/title preservation; first 600 words; 150-word chunks",
                })
            records.append({
                "requested_title": title,
                "resolved_title": page["title"],
                "train_question_count": freq_map.get(title, 0),
                "revision_id": revision_id,
                "source": url,
                "status": "included" if words else "empty",
                "raw_sha256": sha(path),
            })
        done = min(b_start + batch_size, len(titles))
        print(f"{done}/{len(titles)} training-only article requests; {len(passages)} passages", flush=True)

    if not passages:
        raise ValueError("No usable passages; inspect API responses before proceeding")
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in passages)
    if out.exists() and not overwrite_corpus and out.read_text(encoding="utf-8") != content:
        raise ValueError("Frozen evidence corpus differs; pass --overwrite-corpus or use a separately named experiment")
    out.write_text(content, encoding="utf-8")
    save_json(cache / "corpus_manifest.json", {
        "source_train_sha256": source_sha,
        "corpus_sha256": sha(out),
        "article_cap": limit,
        "passages": len(passages),
        "max_passages": limit * 4,
        "max_words_per_passage": 150,
        "selection_rule": "Top-500 QANTA 2021 training-set topics by training question frequency (strictly training-only; zero pilot/dev/test access)",
        "pilot_or_dev_files_consulted": False,
        "limitations": "current Wikipedia revisions, not a 2021 snapshot",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "articles": records,
    })
    print(f"Frozen {len(passages)} training-only Wikipedia passages at {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", required=True, help="Contact email or project URL for Wikipedia API User-Agent")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--overwrite-corpus", action="store_true", help="Update corpus and manifest in place")
    parser.add_argument("--out-corpus", default=None, help="Optional custom output JSONL path")
    parser.add_argument("--cache-subdir", default="wikipedia_train_top500", help="Cache subdirectory under data/raw/")
    args = parser.parse_args()
    build(args.contact, args.limit, overwrite_corpus=args.overwrite_corpus, out_corpus=args.out_corpus, cache_subdir=args.cache_subdir)
