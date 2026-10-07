import tempfile
import unittest
from pathlib import Path
from adaptive_qa.config import load_config
from adaptive_qa.data.load_qanta import Question, load_questions
from adaptive_qa.data.preprocess import make_states, split_questions
from adaptive_qa.evaluation.answers import normalize, exact_match, answer_f1


class Foundations(unittest.TestCase):
    def test_aliases_and_empty_answers(self):
        self.assertTrue(exact_match("The ALBERT Einstein!", ["Albert Einstein"]))
        self.assertTrue(exact_match("USA", ["United States", "USA"]))
        self.assertFalse(exact_match("", [""]))
        self.assertFalse(exact_match("3", ["-3"]))
        self.assertNotEqual(normalize("C++"), normalize("C"))
        self.assertAlmostEqual(answer_f1("Albert Einstein", ["Einstein"]), 2 / 3)

    def test_prefixes_do_not_expose_future_text(self):
        q = Question("1", "one two three four five six seven eight", ("secret",))
        states = make_states(q)
        self.assertEqual([s.question_text for s in states],
                         ["one two", "one two three four", "one two three four five six", q.question_text])
        self.assertFalse(hasattr(states[0], "answers"))
        self.assertEqual(len(make_states(Question("2", "short", ("x",)))), 1)

    def test_question_groups_and_reproducibility(self):
        qs = [Question(str(i), f"Question {i}", (str(i),)) for i in range(50)]
        a = split_questions(qs)
        self.assertEqual(a, split_questions(list(reversed(qs))))
        self.assertEqual([sum(v == s for v in a.values()) for s in
                          ("train", "calibration", "tune", "test")], [30, 5, 5, 10])
        with self.assertRaises(ValueError):
            split_questions([Question(q.question_id, q.question_text, q.answers, official_split="test") for q in qs])

    def test_csv_aliases_and_multimodal_filter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tossups.csv"
            p.write_text('qid,question,clean_answers,has_image,has_audio\n1,A question,"[""yes"", ""y""]",False,False\n2,Picture,"[""x""]",True,False\n')
            qs, report = load_questions(p)
            self.assertEqual(qs[0].answers, ("yes", "y"))
            self.assertEqual(report["skipped_multimodal"], 1)

    def test_duplicate_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text('{"qid":"1","question":"same","answer":"a"}\n'
                         '{"qid":"2","question":"Same","answer":"a"}\n')
            with self.assertRaises(ValueError):
                load_questions(p)

    def test_config_rejects_typo_and_fake_benchmark(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.yaml"
            p.write_text('seed: 4\n')
            self.assertEqual(load_config(p)["seed"], 4)
            p.write_text('modle: {}\n')
            with self.assertRaises(ValueError):
                load_config(p)
            p.write_text('dataset:\n  name: QANTA2026\n')
            with self.assertRaises(ValueError):
                load_config(p)

    def test_explicit_manifest_cannot_move_official_test_to_training(self):
        import json
        qs = [Question(str(i), f"Question {i}", (str(i),), official_split="test" if i == 0 else "train") for i in range(10)]
        mapping = {str(i):("train" if i < 6 else "calibration" if i == 6 else "tune" if i == 7 else "test") for i in range(10)}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "splits.json"
            p.write_text(json.dumps(mapping))
            with self.assertRaises(ValueError):
                split_questions(qs, manifest_path=p)


if __name__ == "__main__":
    unittest.main()
