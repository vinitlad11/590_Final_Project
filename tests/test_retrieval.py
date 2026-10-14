import json
import tempfile
import unittest
from pathlib import Path
from adaptive_qa.retrieval.retriever import BM25
from adaptive_qa.retrieval.evidence import evidence_features


class RetrievalTests(unittest.TestCase):
    def test_rank_empty_query_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "corpus.jsonl"
            p.write_text('{"document_id":"a","text":"red planet mars"}\n'
                         '{"document_id":"b","text":"blue ocean earth"}\n')
            r = BM25(p, Path(d) / "cache")
            docs, usage = r.retrieve("mars")
            self.assertEqual(docs[0]["document_id"], "a")
            self.assertEqual(r.retrieve("mars")[1], usage)
            self.assertEqual(r.cache_hits, 1)
            self.assertEqual(r.retrieve("absentword")[0], [])
            p.write_text('{"document_id":"c","text":"mars new document"}\n')
            other = BM25(p, Path(d) / "cache")
            self.assertEqual(other.retrieve("mars")[0][0]["document_id"], "c")
            self.assertNotEqual(other.corpus_hash, r.corpus_hash)

    def test_no_future_evidence_at_root(self):
        features = evidence_features("Mars", [])
        self.assertTrue(all(value == 0 for value in features.values()))
        docs = [{"text":"The answer is Mars.", "retrieval_score":3}]
        self.assertEqual(evidence_features("Mars", docs)["evidence_support"], 1)
        self.assertEqual(evidence_features("Mar", docs)["evidence_support"], 0)
