"""Fixed-corpus BM25 with inverted postings, title weighting, stopword filtering, and durable caching."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
import time
from adaptive_qa.io import digest, file_hash, read_jsonl, write_json
from adaptive_qa.llm.model import Usage

STOPWORDS = frozenset({
    "a", "about", "after", "all", "also", "an", "and", "another", "any", "are", "around", "as", "at",
    "be", "because", "been", "before", "being", "between", "both", "but", "by", "called", "can", "could",
    "did", "do", "does", "during", "each", "either", "first", "following", "for", "from", "ftp", "give",
    "had", "has", "have", "he", "her", "here", "him", "his", "how", "identify", "if", "in", "include",
    "includes", "including", "into", "is", "it", "its", "later", "known", "made", "make", "many", "may",
    "more", "most", "much", "must", "my", "name", "named", "near", "no", "not", "now", "of", "on", "one",
    "only", "or", "other", "others", "our", "out", "over", "own", "part", "people", "person", "place",
    "point", "points", "same", "see", "several", "she", "should", "since", "so", "some", "such", "take",
    "takes", "ten", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "three", "through", "time", "times", "title", "titled", "to", "two", "under", "up", "upon",
    "use", "used", "uses", "using", "very", "was", "way", "we", "well", "were", "what", "when", "where",
    "which", "while", "who", "whom", "whose", "will", "with", "within", "without", "work", "works",
    "would", "wrote", "written", "year", "years", "10"
})


def tokenize(text):
    return re.findall(r"\w+", str(text).casefold())


def query_terms(query):
    """Filter quizbowl boilerplate and stopwords when informative clue tokens exist."""
    tokens = tokenize(query)
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return filtered if filtered else tokens


class BM25:
    VERSION = "bm25-okapi-positive-idf-v1"

    def __init__(self, corpus_path, cache_dir, max_doc_words=180, k1=1.5, b=0.75):
        if not 0 <= b <= 1 or k1 <= 0:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_hash = file_hash(corpus_path)
        self.docs = read_jsonl(corpus_path)
        self.postings, self.lengths, seen = defaultdict(list), [], set()
        for index, doc in enumerate(self.docs):
            if not isinstance(doc.get("document_id"), str) or doc["document_id"] in seen or not isinstance(doc.get("text"), str):
                raise ValueError("Corpus requires unique string document_id and text")
            if set(doc) & {"ground_truth", "clean_answers", "question_id", "answers"}:
                raise ValueError("Corpus must be independent evidence, not benchmark answer records")
            seen.add(doc["document_id"])
            doc["text"] = " ".join(doc["text"].split()[:max_doc_words])
            title_text = str(doc.get("title") or "").strip()
            # Index title tokens with a 2x weight boost so chunks :1, :2, :3 remain strongly linked to their subject.
            indexed_tokens = tokenize(doc["text"]) + (tokenize(title_text) * 2 if title_text else [])
            counts = Counter(indexed_tokens)
            if not counts:
                raise ValueError("Empty corpus document")
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((index, count))
        if not self.docs:
            raise ValueError("Need a nonempty frozen evidence corpus")
        self.avg_length = sum(self.lengths) / len(self.docs)
        self.identity = {"version": self.VERSION, "corpus_sha256": self.corpus_hash,
                         "k1": k1, "b": b, "max_doc_words": max_doc_words}
        self.new_calls, self.cache_hits = 0, 0

    def retrieve(self, query, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        key = digest({**self.identity, "query": query, "top_k": top_k})
        cache_path = self.cache_dir / (key + ".json")
        if cache_path.exists():
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return obj["documents"], Usage(retrieval_calls=1, latency_seconds=obj["latency_seconds"])
        start, scores = time.perf_counter(), defaultdict(float)
        q_counts = Counter(query_terms(query))
        for term, q_tf in q_counts.items():
            matches = self.postings.get(term, [])
            if not matches:
                continue
            idf = math.log(1 + (len(self.docs) - len(matches) + 0.5) / (len(matches) + 0.5))
            q_weight = 1.0 + math.log(q_tf)
            for index, count in matches:
                denom = count + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg_length)
                scores[index] += q_weight * idf * count * (self.k1 + 1) / denom
        best = sorted(scores, key=lambda i: (-scores[i], self.docs[i]["document_id"]))[:top_k]
        documents = []
        for rank, i in enumerate(best):
            item = {
                "document_id": self.docs[i]["document_id"],
                "text": self.docs[i]["text"],
                "retrieval_score": scores[i],
                "rank": rank + 1,
            }
            if self.docs[i].get("title"):
                item["title"] = self.docs[i]["title"]
            documents.append(item)
        latency = time.perf_counter() - start
        write_json(cache_path, {"documents": documents, "latency_seconds": latency})
        self.new_calls += 1
        return documents, Usage(retrieval_calls=1, latency_seconds=latency)
