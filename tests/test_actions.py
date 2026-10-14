from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from adaptive_qa.config import DEFAULTS
from adaptive_qa.controller.utility import CostModel
from adaptive_qa.controller.oracle import oracle_values, immediate_action_values, best_action
from adaptive_qa.data.preprocess import Observation
from adaptive_qa.llm.model import Usage, CachedLLM, FixtureLLM
from adaptive_qa.outcomes import collect_tree
from adaptive_qa.retrieval.retriever import BM25


class ActionTests(unittest.TestCase):
    def test_usage_rejects_invalid_measured_counts(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                Usage(input_tokens=value)
        with self.assertRaises(ValueError):
            Usage(latency_seconds=float("nan"))

    def test_cost_and_utility_against_hand_calculation(self):
        cm = CostModel(DEFAULTS["cost"], DEFAULTS["reward"])
        u = Usage(100, 50, 1)
        self.assertAlmostEqual(cm.cost(u), 0.15)
        self.assertAlmostEqual(cm.utility(True, True, u, 0.25), 0.9625)
        self.assertAlmostEqual(cm.utility(False, False, u, 0.25), -0.0375)  # Abstention doesn't erase sunk costs.
        self.assertIsNone(cm.dollars(u))

    def test_actual_tree_depth_and_usage_and_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            c = deepcopy(DEFAULTS)
            c["controller"]["max_steps"] = 2
            p = Path(d) / "corpus.jsonl"
            p.write_text('{"document_id":"d","text":"base is 5 and increment is 2"}\n')
            r = BM25(p, Path(d) / "retrieval")
            llm = CachedLLM(FixtureLLM(), c["model"], Path(d) / "llm")
            state = Observation("q", "q:1", "base is 5 and increment is 2", 1, 1, 8)
            rows = collect_tree(llm, r, c, state)
            self.assertEqual(len(rows), 7)
            self.assertEqual(max(n["depth"] for n in rows), 2)
            self.assertEqual(rows[0]["evidence_available"], 0)
            for node in rows:
                node["correct"] = int(node["prediction"] == "7")
                self.assertEqual(node["total_usage"]["llm_calls"], (node["depth"] + 1) * 3)
                if node["history"].endswith("D"):
                    self.assertEqual(node["retrieval_usage"]["retrieval_calls"], 0)
                if node["remaining_steps"] == 0:
                    self.assertEqual(node["children"], {})
            nodes = {n["node_id"]: n for n in rows}
            cm = CostModel(c["cost"], c["reward"])
            values = oracle_values(nodes, rows[0]["node_id"], cm, 0.2)
            self.assertGreaterEqual(max(values.values()), values["ABSTAIN"])

    def test_oracle_simple_known_counterfactual(self):
        nodes = {"r":{"correct":0,"parse_valid":True,"children":{"RETRIEVE":"c"}},
                 "c":{"correct":1,"parse_valid":True,"children":{}, "incremental_usage":Usage(100).__dict__}}
        cm = CostModel(DEFAULTS["cost"], DEFAULTS["reward"])
        v = oracle_values(nodes, "r", cm, 0.5)
        self.assertEqual(best_action(v), "RETRIEVE")
        self.assertAlmostEqual(v["RETRIEVE"], 0.95)

    def test_failed_action_training_reward_is_not_clipped_to_abstention(self):
        nodes = {"r":{"correct":0,"parse_valid":True,"children":{"RETRIEVE":"c"}},
                 "c":{"correct":0,"parse_valid":True,"children":{},"incremental_usage":Usage(100).__dict__}}
        cm = CostModel(DEFAULTS["cost"], DEFAULTS["reward"])
        self.assertAlmostEqual(immediate_action_values(nodes, "r", cm, 0.5)["RETRIEVE"], -1.05)
        self.assertAlmostEqual(oracle_values(nodes, "r", cm, 0.5)["RETRIEVE"], -0.05)
