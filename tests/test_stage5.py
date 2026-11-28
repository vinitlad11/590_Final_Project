from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


class Stage5Tests(unittest.TestCase):
    def test_stage5_holdout_excludes_pilot50_ids_and_texts(self):
        spec = importlib.util.spec_from_file_location("prepare_stage5_holdout", ROOT / "scripts/prepare_stage5_holdout.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pilot_file = d / "pilot50.jsonl"
            pilot_file.write_text(
                json.dumps({"question_id": "qanta2021:1", "question_text": "Overlap ID question", "answers": ["A"]}) + "\n"
                + json.dumps({"question_id": "qanta2021:2", "question_text": "Duplicate text question", "answers": ["B"]}) + "\n",
                encoding="utf-8",
            )
            dev_rows = [
                {"qanta_id": 1, "fold": "dev", "text": "Overlap ID question", "answer": "A", "page": "A"},
                {"qanta_id": 999, "fold": "dev", "text": "Duplicate text question", "answer": "B", "page": "B"},
            ]
            for i in range(10, 30):
                dev_rows.append({
                    "qanta_id": i,
                    "fold": "dev",
                    "text": f"Unique untouched holdout question number {i}",
                    "answer": f"Entity {i} [accept Alias {i}]",
                    "page": f"Entity_{i}_(novel)",
                })
            dev_file = d / "qanta.dev.2021.12.20.json"
            dev_file.write_text(json.dumps({"questions": dev_rows}), encoding="utf-8")

            out_jsonl, out_splits, out_manifest = mod.build_holdout(
                count=10,
                dev_path=dev_file,
                pilot50_path=pilot_file,
                out_dir=d / "out",
            )
            selected = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(selected), 10)
            qids = {q["question_id"] for q in selected}
            self.assertNotIn("qanta2021:1", qids)
            self.assertNotIn("qanta2021:999", qids)
            for q in selected:
                self.assertTrue(len(q["answers"]) >= 1)
                self.assertFalse(any("accept" in a.lower() for a in q["answers"]))
            mf = json.loads(out_manifest.read_text(encoding="utf-8"))
            self.assertEqual(mf["overlap_with_stage3_pilot50"], 0)


if __name__ == "__main__":
    unittest.main()
