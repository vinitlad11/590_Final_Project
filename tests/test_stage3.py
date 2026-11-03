from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from adaptive_qa.config import DEFAULTS, validate
from adaptive_qa.baseline import run_baseline
from adaptive_qa.io import read_jsonl
from adaptive_qa.llm.model import CachedLLM, FixtureLLM
from adaptive_qa.llm.prompts import build_prompt
from adaptive_qa.outcomes import collect_outcomes
from adaptive_qa.pilot import pilot_report
from adaptive_qa.reproducibility import manifest, check_manifest
from adaptive_qa.sessions import record_invocation

ROOT = Path(__file__).resolve().parents[1]


def config(root, size):
    c = deepcopy(DEFAULTS)
    c['_root'] = str(root)
    c['dataset']['path'] = str(ROOT / 'data/fixtures/questions.jsonl')
    c['dataset']['limit_questions'] = size
    c['retrieval']['corpus'] = str(ROOT / 'data/fixtures/corpus.jsonl')
    c['model'].update(prompt_version='qa-answer-v2', samples=2)
    c['controller'].update(actions=['RETRIEVE'], max_steps=1)
    c['evaluation']['include_wait'] = False
    c['fractions'] = [0.5, 1.0]
    c['experiment_id'] = 'fixture_stage3_' + str(size)
    return c


class Stage3Tests(unittest.TestCase):
    def test_answer_only_prompt_has_no_example_answer_and_rejects_calibration(self):
        prompt = build_prompt('Only revealed clues', confidence=False, version='qa-answer-v2')
        self.assertNotIn('short answer', prompt)
        self.assertEqual(json.loads(prompt.split('DATA_JSON:\n')[1]), {'revealed_question': 'Only revealed clues'})
        with self.assertRaises(ValueError):
            build_prompt('x', version='qa-answer-v2')
        c = config(ROOT, 10)
        validate(c)
        c['calibration']['enabled'] = True
        with self.assertRaises(ValueError):
            validate(c)

    def test_failed_attempt_is_retained_and_does_not_invent_token_counts(self):
        class Failing(FixtureLLM):
            def generate(self, *args, **kwargs):
                raise RuntimeError('simulated interruption')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            llm = CachedLLM(Failing(), {}, p / 'cache', max_new_calls=1, ledger_path=p / 'usage.json')
            with self.assertRaises(RuntimeError):
                llm.generate('test', 0, 1)
            row = json.loads((p / 'usage.json').read_text())
            self.assertEqual((row['new_calls'], row['failed_calls'], row['pending_calls']), (1, 1, 0))
            self.assertEqual(row['incurred_usage']['input_tokens'], 0)
            with self.assertRaises(RuntimeError):
                llm.generate('test', 0, 1)
            self.assertEqual(llm.new_calls, 1)

    def test_followup_to_pilot_reuses_calls_and_groups_requested_prefixes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = config(root, 10)
            out = root / 'results' / c['experiment_id']; out.mkdir(parents=True)
            manifest(c, out)
            with record_invocation(c, out, 'baseline'):
                baseline = run_baseline(c, out)
            self.assertTrue(all(r['parse_valid'] and r['verbal_confidence'] is None for r in baseline))
            self.assertEqual(json.loads((out / 'baseline_collection_usage.json').read_text())['new_calls'], 40)
            with record_invocation(c, out, 'collect'):
                collect_outcomes(c, out)
            self.assertEqual(json.loads((out / 'collection_usage.json').read_text())['new_calls'], 60)
            report = pilot_report(c, out)
            self.assertEqual([r['states'] for r in report['by_prefix']], [10, 10])
            self.assertEqual(report['samples_reviewed'], 100)
            self.assertEqual(report['logical_collection_usage']['llm_calls'], 100)
            self.assertEqual(report['actual_invocations']['attempted_backend_calls'], 100)
            self.assertEqual(report['actual_invocations']['unfinished_invocations'], 0)
            c = config(root, 50)
            out = root / 'results' / c['experiment_id']; out.mkdir(parents=True)
            manifest(c, out)
            with record_invocation(c, out, 'collect'):
                collect_outcomes(c, out)
            usage = json.loads((out / 'collection_usage.json').read_text())
            self.assertEqual((usage['new_calls'], usage['cache_hits']), (400, 100))
            report = pilot_report(c, out)
            self.assertEqual((report['questions'], report['root_states'], report['observed_nodes']), (50, 100, 200))
            self.assertEqual([r['states'] for r in report['by_prefix']], [50, 50])
            self.assertEqual(report['logical_collection_usage']['llm_calls'], 500)
            self.assertEqual(report['logical_collection_usage']['retrieval_calls'], 100)
            self.assertEqual(report['samples_reviewed'], 500)
            spec = importlib.util.spec_from_file_location('verify_collection', ROOT / 'scripts/verify_collection.py')
            verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
            self.assertEqual(verifier.verify(c, out)['status'], 'PASSED')
            # A truncated run must not be mistaken for a successful collection.
            path = out / 'action_nodes.jsonl'
            lines = path.read_text().splitlines(); path.write_text('\n'.join(lines[:-1]) + '\n')
            with self.assertRaises(ValueError):
                verifier.verify(c, out)

    def test_reporting_refuses_source_changed_after_collection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'src').mkdir()
            source = root / 'src/example.py'; source.write_text('x = 1\n')
            c = config(root, 10)
            out = root / 'run'; out.mkdir()
            manifest(c, out)
            source.write_text('x = 2\n')
            with self.assertRaises(ValueError):
                check_manifest(c, out)


if __name__ == '__main__':
    unittest.main()
