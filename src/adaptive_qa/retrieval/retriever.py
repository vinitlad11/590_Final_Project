"""Fixed-corpus BM25 with inverted postings and corpus-aware durable caching."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
import time
from adaptive_qa.io import digest, file_hash, read_jsonl, write_json
from adaptive_qa.llm.model import Usage


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


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
            counts = Counter(tokenize(doc["text"]))
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
        for term in set(tokenize(query)):
            matches = self.postings.get(term, [])
            idf = math.log(1 + (len(self.docs) - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, count in matches:
                denom = count + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg_length)
                scores[index] += idf * count * (self.k1 + 1) / denom
        best = sorted(scores, key=lambda i: (-scores[i], self.docs[i]["document_id"]))[:top_k]
        documents = [{"document_id": self.docs[i]["document_id"], "text": self.docs[i]["text"],
                      "retrieval_score": scores[i], "rank": rank + 1} for rank, i in enumerate(best)]
        latency = time.perf_counter() - start
        write_json(cache_path, {"documents": documents, "latency_seconds": latency})
        self.new_calls += 1
        return documents, Usage(retrieval_calls=1, latency_seconds=latency)
