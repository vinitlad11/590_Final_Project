from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch
from adaptive_qa.config import DEFAULTS
from adaptive_qa.llm.model import OllamaLLM


class AdapterTests(unittest.TestCase):
    def test_ollama_transport_contract_and_usage(self):
        config = deepcopy(DEFAULTS["model"])
        config.update(backend="ollama", name="test:tiny", revision="auto")
        received = []

        def fake(request, **kwargs):
            received.append(request)
            if request.full_url.endswith("/api/tags"):
                return io.BytesIO(json.dumps({"models":[{"name":"test:tiny","digest":"known-digest"}]}).encode())
            return io.BytesIO(json.dumps({"done":True,"model":"test:tiny","response":'{"answer":"x","confidence":0.7}',
                                         "prompt_eval_count":37,"eval_count":12}).encode())

        with patch("urllib.request.urlopen", side_effect=fake):
            model = OllamaLLM(config)
            result = model.answer_with_confidence("revealed prefix", seed=17)
        self.assertEqual(result.usage.input_tokens, 37)
        self.assertEqual(result.usage.output_tokens, 12)
        self.assertEqual(result.usage.llm_calls, 1)
        self.assertIn("known-digest", result.model)
        body = json.loads(received[-1].data)
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["seed"], 17)

    def test_no_invented_api_token_counts(self):
        config = deepcopy(DEFAULTS["model"])
        config.update(backend="ollama", name="test:tiny", revision="auto")
        def fake(request, **kwargs):
            obj = ({"models":[{"name":"test:tiny","digest":"abc"}]} if request.full_url.endswith("/api/tags")
                   else {"done":True,"response":"x"})
            return io.BytesIO(json.dumps(obj).encode())
        with patch("urllib.request.urlopen", side_effect=fake):
            model = OllamaLLM(config)
            with self.assertRaises(RuntimeError):
                model.answer("question")
