from dataclasses import asdict
import tempfile
import unittest
from adaptive_qa.llm.answer import parse_answer
from adaptive_qa.llm.model import CachedLLM, FixtureLLM
from adaptive_qa.llm.prompts import build_prompt


class LLMTests(unittest.TestCase):
    def test_strict_parser(self):
        self.assertTrue(parse_answer('```json\n{"answer":"x","confidence":0.6}\n```').valid)
        for text in ['{"answer":"x","confidence":99}', '{"answer":"x","confidence":true}',
                     '{"answer":"x","confidence":NaN}', 'not JSON', '{"confidence":0.5}',
                     '{"answer":"x","confidence":"0.9"}']:
            self.assertFalse(parse_answer(text).valid)
        self.assertIsNone(parse_answer('{"answer":"x"}', confidence=False).confidence)

    def test_cache_is_persistent_and_seed_and_prompt_sensitive(self):
        with tempfile.TemporaryDirectory() as d:
            c = CachedLLM(FixtureLLM(), {"name": "test"}, d, 3)
            a = c.answer_with_confidence("base is 2 and increment is 3", seed=1)
            b = c.answer_with_confidence("base is 2 and increment is 3", seed=1)
            self.assertEqual(a.raw, b.raw)
            self.assertTrue(b.cache_hit)
            self.assertEqual(a.usage, b.usage)  # Warm cache must not make logical policy costs free.
            c.answer_with_confidence("base is 2 and increment is 3", seed=2)
            c.answer_with_confidence("base is 3 and increment is 3", seed=2)
            self.assertEqual(c.new_calls, 3)
            with self.assertRaises(RuntimeError):
                c.answer_with_confidence("another", seed=2)
            fresh = CachedLLM(FixtureLLM(), {"name": "test"}, d)
            self.assertTrue(fresh.answer_with_confidence("base is 2 and increment is 3", seed=1).cache_hit)
            self.assertEqual(fresh.new_calls, 0)

    def test_deliberation_has_no_retrieved_evidence(self):
        prompt = build_prompt("Only the prefix", "DELIBERATE", "old guess")
        self.assertIn("old guess", prompt)
        self.assertNotIn('"evidence":', prompt)


if __name__ == "__main__":
    unittest.main()
