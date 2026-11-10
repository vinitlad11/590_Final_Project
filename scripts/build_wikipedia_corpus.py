"""Freeze up to 500 Wikipedia articles with guaranteed pilot topic coverage, retry resilience, and clean text."""
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
from prepare_qanta_public import RAW, ROOT, RELEASE, candidate_rows, raw_questions, save_json, sha

# Canonical Wikipedia article titles for malformed or ambiguous QANTA 2021 page strings.
TITLE_OVERRIDES = {
    "ⅆ": "Duel",
    "THE Tunnel": "The Tunnel (novel)",
    "THE_Tunnel": "The Tunnel (novel)",
    "Ansichten eines Clowns": "The Clown (novel)",
    "Ansichten_eines_Clowns": "The Clown (novel)",
    "New york": "New York (state)",
    "New_york": "New York (state)",
    "Mogul Dynasty": "Mughal Empire",
    "Mogul_Dynasty": "Mughal Empire",
    "Prairie Style": "Prairie School",
    "Prairie_Style": "Prairie School",
    "Programming languages": "Programming language",
    "Programming_languages": "Programming language",
    "Seas": "Sea",
}


def canonical_title(raw_title):
    t = str(raw_title or "").replace("_", " ").strip()
    return TITLE_OVERRIDES.get(t, TITLE_OVERRIDES.get(raw_title, t))


def clean_wikitext(wikitext, page_title):
    import mwparserfromhell
    code = mwparserfromhell.parse(wikitext)
    # Preserve inline text inside {{lang|fr|À rebours}} or {{transl|...}} before stripping templates.
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


def fetch_article(title, cache, contact, extra_cache_dirs=(), max_retries=4):
    key = hashlib.sha256(title.encode()).hexdigest()
    path = cache / (key + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    for extra in extra_cache_dirs:
        candidate = Path(extra) / (key + ".json")
        if candidate.exists():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            save_json(path, raw)
            return raw, path

    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main",
        "titles": title, "maxlag": 5,
    }
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    last_err = None
    for attempt in range(max_retries):
        try:
            request = Request(url, headers={"User-Agent": f"AdaptiveQAStudentPilot/1.0 ({contact})"})
            with urlopen(request, timeout=60) as response:
                raw = json.load(response)
            if "error" not in raw:
                save_json(path, raw)
                time.sleep(0.15)
                return raw, path
            last_err = f"Wikipedia returned {raw['error']}"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"{last_err}; wait and rerun, preserving the cache")


def select_titles(limit=500, selection_file=None):
    ordered, seen = [], set()

    def add(raw_t):
        t = canonical_title(raw_t)
        if t and t.casefold() not in seen:
            seen.add(t.casefold())
            ordered.append(t)

    # 1. Always include the 50 pilot question Wikipedia topics first.
    sel_path = Path(selection_file) if selection_file else ROOT / "data/processed/qanta2021/selection.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        for q in sel.get("questions", []):
            add(q.get("wikipedia_page", ""))

    # 2. If raw qanta.train exists locally, order training pages by question-ID hash, then seeded distractors.
    source = RAW / f"qanta.train.{RELEASE}.json"
    source_sha = None
    if source.exists():
        source_sha = sha(source)
        rows = raw_questions(source)
        eligible_train, _ = candidate_rows(rows, "train", set(), set())
        for r in eligible_train[:200]:
            add(r.get("wikipedia_page", ""))
        all_train_titles = {
            canonical_title(r["page"]) for r in rows
            if isinstance(r.get("page"), str) and r["page"].strip()
            and "train" in str(r.get("fold", "train")).lower()
        }
        for t in sorted(all_train_titles, key=lambda x: hashlib.sha256(("42:" + x).encode()).hexdigest()):
            if len(ordered) >= limit:
                break
            add(t)

    # 3. Reuse already-downloaded distractor articles from corpus_manifest.json or evidence_corpus_qanta2021.jsonl
    #    so Stage 3 does not need to make 450 redundant network requests.
    manifest_path = ROOT / "data/raw/wikipedia_pilot/corpus_manifest.json"
    if len(ordered) < limit and manifest_path.exists():
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_sha = source_sha or mf.get("source_train_sha256")
        for art in mf.get("articles", []):
            if len(ordered) >= limit:
                break
            if art.get("status") in {"included", "reused_from_stage2_corpus"}:
                add(art.get("requested_title") or art.get("resolved_title") or "")

    existing_ts = ROOT / "data/raw/wikipedia_pilot/title_selection.json"
    if len(ordered) < limit and existing_ts.exists():
        old = json.loads(existing_ts.read_text(encoding="utf-8"))
        source_sha = source_sha or old.get("source_train_sha256")
        for t in old.get("titles", []):
            if len(ordered) >= limit:
                break
            add(t)

    existing_corpus = ROOT / "data/raw/evidence_corpus_qanta2021.jsonl"
    if len(ordered) < limit and existing_corpus.exists():
        for line in existing_corpus.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if len(ordered) >= limit:
                break
            add(row.get("title", ""))

    return ordered[:limit], source_sha


def build(contact, limit=500, overwrite_corpus=False):
    if not 1 <= limit <= 1000:
        raise ValueError("Use 1 to 1000 articles for this pilot")
    if "@" not in contact and not contact.startswith("https://"):
        raise ValueError("Supply a real contact email or project URL for the Wikipedia User-Agent")
    cache = ROOT / "data/raw/wikipedia_pilot"
    cache.mkdir(parents=True, exist_ok=True)

    # Discover Stage 2 raw cache directory if imported from Stage 2
    extra_caches = []
    import_record = ROOT / "results/stage2_data_import.json"
    if import_record.exists():
        src_proj = json.loads(import_record.read_text(encoding="utf-8")).get("source_project")
        if src_proj and (Path(src_proj) / "data/raw/wikipedia_pilot").is_dir():
            extra_caches.append(Path(src_proj) / "data/raw/wikipedia_pilot")
    sibling_stage2 = ROOT.parent / "adaptive-confidence-agent/data/raw/wikipedia_pilot"
    if sibling_stage2.is_dir() and sibling_stage2 not in extra_caches:
        extra_caches.append(sibling_stage2)

    # Map both requested_title and resolved_title -> existing passages so zero network calls are needed for distractors
    out = ROOT / "data/raw/evidence_corpus_qanta2021.jsonl"
    existing_by_key = {}
    if out.exists():
        by_rev = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t_key = str(row.get("title") or "").casefold()
            existing_by_key.setdefault(t_key, []).append(row)
            if row.get("revision_id") is not None:
                by_rev.setdefault(row["revision_id"], []).append(row)
        manifest_path = cache / "corpus_manifest.json"
        if manifest_path.exists():
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            for art in mf.get("articles", []):
                req = str(art.get("requested_title") or "").casefold()
                res = str(art.get("resolved_title") or "").casefold()
                rev_id = art.get("revision_id")
                if rev_id in by_rev:
                    if req:
                        existing_by_key[req] = by_rev[rev_id]
                    if res:
                        existing_by_key[res] = by_rev[rev_id]

    titles, source_sha = select_titles(limit=limit)
    selection_path = cache / "title_selection.json"
    selection = {
        "source_train_sha256": source_sha or "reused_from_stage2_manifest",
        "seed": 42,
        "article_cap": limit,
        "titles": titles,
    }
    if selection_path.exists() and not overwrite_corpus and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Corpus selection changed. Pass --overwrite-corpus or use a new corpus directory.")
    save_json(selection_path, selection)

    passages, records = [], []
    unique_revisions = set()
    for index, title in enumerate(titles, 1):
        key = hashlib.sha256(title.encode()).hexdigest()
        cached_raw = cache / (key + ".json")
        if not cached_raw.exists() and title.casefold() in existing_by_key:
            reused = existing_by_key[title.casefold()]
            rev_id = reused[0]["revision_id"]
            if rev_id not in unique_revisions:
                unique_revisions.add(rev_id)
                for chunk_row in reused:
                    c_title = chunk_row.get("title", title)
                    c_text = chunk_row["text"]
                    if c_title and not c_text.casefold().startswith(c_title.casefold()):
                        c_text = f"{c_title}. {c_text}"
                    passages.append({**chunk_row, "text": " ".join(c_text.split()[:150])})
                records.append({
                    "requested_title": title, "resolved_title": reused[0].get("title", title),
                    "revision_id": rev_id, "source": reused[0].get("source", ""),
                    "status": "reused_from_stage2_corpus",
                })
            if index % 25 == 0 or index == len(titles):
                print(f"{index}/{len(titles)} article requests; {len(passages)} passages", flush=True)
            continue

        try:
            raw, path = fetch_article(title, cache, contact, extra_cache_dirs=extra_caches)
        except RuntimeError as exc:
            if index > 50:
                records.append({"requested_title": title, "status": "skipped_transient_error", "error": str(exc)})
                continue
            raise
        page = raw["query"]["pages"][0]
        if "missing" in page or not page.get("revisions"):
            records.append({"requested_title": title, "status": "missing", "raw_sha256": sha(path)})
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
            "requested_title": title, "resolved_title": page["title"], "revision_id": revision_id,
            "source": url, "status": "included" if words else "empty", "raw_sha256": sha(path),
        })
        if index % 25 == 0 or index == len(titles):
            print(f"{index}/{len(titles)} article requests; {len(passages)} passages", flush=True)

    if not passages:
        raise ValueError("No usable passages; inspect API responses before proceeding")
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in passages)
    if out.exists() and not overwrite_corpus and out.read_text(encoding="utf-8") != content:
        raise ValueError("Frozen evidence corpus differs; pass --overwrite-corpus or use a separately named experiment")
    out.write_text(content, encoding="utf-8")
    save_json(cache / "corpus_manifest.json", {
        "source_train_sha256": source_sha or "reused_from_stage2_manifest",
        "corpus_sha256": sha(out),
        "article_cap": limit,
        "passages": len(passages),
        "max_passages": limit * 4,
        "max_words_per_passage": 150,
        "selection_rule": "pilot selection topics plus seeded QANTA training distractors",
        "limitations": "current Wikipedia revisions, not a 2021 snapshot",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "articles": records,
    })
    print(f"Frozen {len(passages)} independent Wikipedia passages at {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", required=True, help="Contact email or project URL for Wikipedia API User-Agent")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--overwrite-corpus", action="store_true", help="Update corpus and manifest in place")
    args = parser.parse_args()
    build(args.contact, args.limit, overwrite_corpus=args.overwrite_corpus)
