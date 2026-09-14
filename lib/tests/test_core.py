from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

LIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB))

from awe_scoring.config import load_config, package_root
from awe_scoring.estimators import ScoreVotingClassifier, build_estimator
from awe_scoring.evaluation import evaluate
from awe_scoring.model_adapters import (
    BedrockAdapter,
    OllamaAdapter,
    build_adapter,
    build_converse_request,
    extract_text,
)
from awe_scoring.pipeline import process_rows
from awe_scoring.profiles import parse_profile, profile_features
from awe_scoring.routing import hard_gate, route_key, score_to_range


class RoutingTests(unittest.TestCase):
    def test_four_quadrants(self) -> None:
        self.assertEqual(route_key("English", "td"), "english-td")
        self.assertEqual(route_key("en", "CV"), "english-cv")
        self.assertEqual(route_key("French", "TD"), "french-td")
        self.assertEqual(route_key("fr", "cv"), "french-cv")

    def test_score_first_projection(self) -> None:
        self.assertEqual([score_to_range(value, "TD") for value in range(7)], ["Low"] * 3 + ["Medium"] * 2 + ["High"] * 2)
        self.assertEqual([score_to_range(value, "CV") for value in range(5)], ["Low"] * 2 + ["Medium"] * 2 + ["High"])

    def test_hard_gates(self) -> None:
        self.assertEqual(hard_gate({"skill": "TD", "response_text": ""}, 0), (0, "blank-or-non-linguistic"))
        self.assertEqual(hard_gate({"skill": "CV", "response_text": "short"}, 1), (1, "cv-under-30-authored-words"))


class ProfileTests(unittest.TestCase):
    def test_compact_parsing_and_frozen_expansion_counts(self) -> None:
        td = parse_profile("T2|P2|B2|S2|L2|E2|O2|A2|F2|C2", "TD")
        cv = parse_profile("I2|R2|G2|S2|U2|P2|O2|A2|C2", "CV", derived_d=1)
        self.assertEqual(len(profile_features("TD", td)), 53)
        self.assertEqual(len(profile_features("CV", cv)), 54)

    def test_response_shapes(self) -> None:
        self.assertEqual(extract_text([{"generated_text": "I2"}]), "I2")
        self.assertEqual(extract_text({"choices": [{"message": {"content": "I2"}}]}), "I2")
        self.assertEqual(extract_text({"message": {"role": "assistant", "content": "I2"}}), "I2")
        converse = {"output": {"message": {"role": "assistant", "content": [{"text": "I2"}]}}}
        self.assertEqual(extract_text(converse), "I2")

    def test_explicit_ollama_model_wins_over_environment(self) -> None:
        config = load_config()["model"]
        config["ollama"]["model"] = "gemma4:e4b"
        config["ollama"]["_explicit_model_override"] = True
        with patch.dict("os.environ", {"AWE_OLLAMA_MODEL": "qwen2.5:3b"}):
            self.assertEqual(OllamaAdapter(config).model_name, "gemma4:e4b")


class BedrockAdapterTests(unittest.TestCase):
    def test_system_message_is_hoisted_out_of_turns(self) -> None:
        model = load_config()["model"]
        messages = [{"role": "system", "content": "rubric"}, {"role": "user", "content": "response"}]
        request = build_converse_request(messages, model)
        self.assertEqual(request["system"], [{"text": "rubric"}])
        self.assertEqual(request["messages"], [{"role": "user", "content": [{"text": "response"}]}])
        self.assertEqual(request["inferenceConfig"], {"temperature": 0.0, "maxTokens": 80})

    def test_system_key_omitted_when_absent(self) -> None:
        request = build_converse_request([{"role": "user", "content": "response"}], {})
        self.assertNotIn("system", request)

    def test_model_id_from_environment_and_required(self) -> None:
        model = load_config()["model"]
        with patch("boto3.client") as client:
            with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "us.anthropic.claude-test"}):
                self.assertEqual(BedrockAdapter(model).model_id, "us.anthropic.claude-test")
                self.assertIsInstance(build_adapter(model), BedrockAdapter)
            self.assertTrue(client.called)
        with patch.dict("os.environ", {"BEDROCK_MODEL_ID": ""}):
            with self.assertRaises(ValueError):
                BedrockAdapter(model)

    def test_invoke_returns_converse_text(self) -> None:
        model = load_config()["model"]
        stub = Mock()
        stub.converse.return_value = {"output": {"message": {"role": "assistant", "content": [{"text": "I2|R2"}]}}}
        with patch("boto3.client", return_value=stub):
            with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "us.anthropic.claude-test"}):
                adapter = BedrockAdapter(model)
        self.assertEqual(adapter.invoke([{"role": "user", "content": "response"}]), "I2|R2")
        self.assertEqual(stub.converse.call_args.kwargs["modelId"], "us.anthropic.claude-test")


class ArchitectureTests(unittest.TestCase):
    def test_estimators_match_documented_families(self) -> None:
        forest = build_estimator("random_forest_depth4", 7)
        self.assertEqual(forest.max_depth, 4)
        voting = build_estimator("tree_ordinal_median", 7).steps[-1][1]
        self.assertIsInstance(voting, ScoreVotingClassifier)
        self.assertEqual(voting.decode, "median")

    def test_feature_contract_counts(self) -> None:
        expected = {"english-td": 132, "english-cv": 304, "french-td": 240, "french-cv": 261}
        root = package_root() / "context_materials/feature_contracts"
        for route, count in expected.items():
            contract = json.loads((root / f"{route}.json").read_text(encoding="utf-8"))
            self.assertEqual(contract["feature_count"], count)
            self.assertEqual(len(contract["feature_names"]), count)

    def test_provider_neutral_mock_run(self) -> None:
        config = load_config()
        text = " ".join(["This response remains readable and controlled."] * 8)
        rows = [{"id": "cv-1", "language": "en", "skill": "CV", "response_text": text}]
        output = process_rows(rows, config, adapter_override="mock")
        self.assertEqual(output[0]["route"], "english-cv")
        self.assertEqual(output[0]["model_attempts"], 1)
        self.assertEqual(output[0]["llm_profile"]["D"], 3)
        self.assertIsNone(output[0]["stage2_score"])

    def test_evaluation_and_bundle_smoke(self) -> None:
        config = load_config()
        config["routes"]["english-td"]["repeats"] = 1
        config["routes"]["english-td"]["folds"] = 2
        rows = []
        for score in range(7):
            for copy in range(2):
                rows.append({
                    "id": f"td-{score}-{copy}",
                    "route": "english-td",
                    "skill": "TD",
                    "expert_score": score,
                    "features": {"det_signal": float(score), "det_copy": float(copy)},
                })
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "models.joblib"
            report, predictions = evaluate(rows, config, bundle)
            self.assertEqual(report["routes"]["english-td"]["n"], 14)
            self.assertEqual(len(predictions), 14)
            self.assertTrue(bundle.exists())


if __name__ == "__main__":
    unittest.main()
