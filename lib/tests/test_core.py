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
import joblib
import numpy as np

from awe_scoring.estimators import MedianCumulativeOrdinalClassifier, ScoreVotingClassifier, build_estimator
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
from awe_scoring.routing import compile_spec, hard_gate, resolve_route

CAEC_CONFIG = package_root() / "config/caec.yaml"


def _gate(spec, row: dict, word_count: int) -> tuple:
    return hard_gate(spec, resolve_route(spec, row), row, {}, word_count)


def _config(**overrides) -> dict:
    config = load_config()
    config.update(overrides)
    return config


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = compile_spec(load_config())

    def route_name(self, language: str, skill: str | None) -> str:
        return resolve_route(self.spec, {"language": language, "skill": skill}).name

    def test_four_quadrants(self) -> None:
        self.assertEqual(self.route_name("English", "td"), "english-td")
        self.assertEqual(self.route_name("en", "CV"), "english-cv")
        self.assertEqual(self.route_name("French", "TD"), "french-td")
        self.assertEqual(self.route_name("fr", "cv"), "french-cv")
        with self.assertRaises(ValueError):
            self.route_name("en", "OR")
        with self.assertRaises(ValueError):
            self.route_name("en", None)  # two English routes: skill is required

    def test_score_first_projection(self) -> None:
        td, cv = self.spec.skills["TD"], self.spec.skills["CV"]
        self.assertEqual([td.to_range(value) for value in range(7)], ["Low"] * 3 + ["Medium"] * 2 + ["High"] * 2)
        self.assertEqual([cv.to_range(value) for value in range(5)], ["Low"] * 2 + ["Medium"] * 2 + ["High"])
        with self.assertRaises(ValueError):
            cv.to_range(5)

    def test_hard_gates(self) -> None:
        spec = self.spec
        self.assertEqual(_gate(spec, {"language": "en", "skill": "TD", "response_text": ""}, 0), (0, "blank-or-non-linguistic"))
        self.assertEqual(_gate(spec, {"language": "en", "skill": "CV", "response_text": "short"}, 1), (1, "cv-under-30-authored-words"))
        self.assertEqual(_gate(spec, {"language": "en", "skill": "TD", "response_text": "short"}, 1), (None, None))
        non_linguistic = {"language": "fr", "skill": "TD", "response_text": "zzzz", "is_linguistic": False}
        self.assertEqual(_gate(spec, non_linguistic, 40), (0, "blank-or-non-linguistic"))
        self.assertEqual(_gate(spec, {**non_linguistic, "is_linguistic": 0}, 40), (None, None))


class ConfigurableSkillTests(unittest.TestCase):
    def test_custom_labels_thresholds_and_score_lists(self) -> None:
        spec = compile_spec(_config(skills={
            "TD": {"score_values": list(range(7)), "ranges": [
                {"label": "Emerging", "scores": [0, 1]},
                {"label": "Developing", "max": 3},
                {"label": "Proficient", "max": 5},
                {"label": "Extending", "scores": [6]},
            ]},
            "CV": {"score_values": list(range(5))},
        }))
        td = spec.skills["TD"]
        self.assertEqual(td.labels, ["Emerging", "Developing", "Proficient", "Extending"])
        self.assertEqual(
            [td.to_range(value) for value in range(7)],
            ["Emerging"] * 2 + ["Developing"] * 2 + ["Proficient"] * 2 + ["Extending"],
        )
        # Without ranges each score is its own category.
        self.assertEqual(spec.skills["CV"].labels, ["0", "1", "2", "3", "4"])

    def test_invalid_skills_and_gates_are_rejected(self) -> None:
        cv = {"score_values": list(range(5))}
        blank = {"field": "response_text", "blank": True}
        invalid = {
            "overlapping ranges": {"skills": {"TD": {"score_values": [0, 1, 2], "ranges": [
                {"label": "A", "scores": [0, 1]}, {"label": "B", "scores": [1, 2]}]}, "CV": cv}},
            "uncovered score": {"skills": {"TD": {"score_values": [0, 1, 2], "ranges": [{"label": "A", "max": 1}]}, "CV": cv}},
            "duplicate labels": {"skills": {"TD": {"score_values": [0, 1], "ranges": [
                {"label": "A", "max": 0}, {"label": "A", "max": 1}]}, "CV": cv}},
            "gate score outside scale": {"hard_gates": [{"name": "x", "score": 7, "when": blank}]},
            "unknown gate operator": {"hard_gates": [{"name": "x", "score": 0, "when": {"field": "response_text", "like": "a"}}]},
        }
        for name, overrides in invalid.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    compile_spec(_config(**overrides))

    def test_invalid_routes_are_rejected(self) -> None:
        mutations = {
            "unknown learner": {"learner": "gradient_boosting"},
            "undefined skill": {"skill": "OR"},
            "feature view mismatch": {"feature_view": "combined"},
            "duplicate selector": {"skill": "CV"},
        }
        for name, change in mutations.items():
            with self.subTest(name):
                config = load_config()
                config["routes"]["english-td"].update(change)
                with self.assertRaises(ValueError):
                    compile_spec(config)
        config = load_config()
        config["skills"]["OR"] = {"score_values": [0, 1, 2]}
        config["routes"]["english-or"] = {"language": "en", "skill": "OR", "uses_llm": True, "learner": "random_forest_depth4"}
        with self.assertRaisesRegex(ValueError, "profile_schema"):
            compile_spec(config)
        config["routes"]["english-or"]["profile_schema"] = "CV"
        self.assertEqual(compile_spec(config).routes["english-or"].profile_schema, "CV")

    def test_learner_is_switchable_per_route(self) -> None:
        config = load_config()
        config["routes"]["english-td"].update({"learner": "cumulative_ordinal_median_c003", "repeats": 1, "folds": 2})
        rows = [{
            "id": f"td-{score}-{copy}", "route": "english-td", "expert_score": score,
            "features": {"det_signal": float(score), "det_copy": float(copy)},
        } for score in range(7) for copy in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "models.joblib"
            report, _ = evaluate(rows, config, bundle)
            fitted = joblib.load(bundle)["routes"]["english-td"]
        self.assertEqual(report["routes"]["english-td"]["learner"], "cumulative_ordinal_median_c003")
        self.assertIsInstance(fitted["estimator"].steps[-1][1], MedianCumulativeOrdinalClassifier)


class _ZeroEstimator:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=int)


class CaecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CAEC_CONFIG)
        self.spec = compile_spec(self.config)

    def test_rows_without_skill_route_by_language(self) -> None:
        self.assertEqual(resolve_route(self.spec, {"language": "en"}).name, "english-holistic")
        self.assertEqual(resolve_route(self.spec, {"language": "French", "skill": "holistic"}).name, "french-holistic")
        with self.assertRaises(ValueError):
            resolve_route(self.spec, {"language": "en", "skill": "TD"})

    def test_only_blank_responses_score_zero(self) -> None:
        self.assertEqual(_gate(self.spec, {"language": "en", "response_text": "  \n"}, 0), (0, "blank-response"))
        self.assertEqual(_gate(self.spec, {"language": "en", "response_text": "Hi"}, 1), (None, None))
        self.assertEqual(_gate(self.spec, {"language": "en", "response_text": "zz", "is_linguistic": False}, 1), (None, None))

    def test_pass_fail_projection(self) -> None:
        skill = self.spec.skills["HOLISTIC"]
        self.assertEqual([skill.to_range(value) for value in range(10)], ["Fail"] * 5 + ["Pass"] * 5)

    def test_ungated_predictions_never_fall_below_one(self) -> None:
        bundle = {"routes": {"english-holistic": {
            "estimator": _ZeroEstimator(), "feature_names": ["det_portable_word_count"], "score_min": 0,
        }}}
        rows = [
            {"id": "blank", "language": "en", "response_text": ""},
            {"id": "word", "language": "en", "response_text": "Hello"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.joblib"
            joblib.dump(bundle, path)
            output = {row["id"]: row for row in process_rows(rows, self.config, bundle_path=path)}
        self.assertEqual((output["blank"]["stage2_score"], output["blank"]["scoring_source"]), (0, "hard-gate"))
        self.assertEqual((output["word"]["stage2_score"], output["word"]["hml"]), (1, "Fail"))
        self.assertEqual(output["word"]["skill"], "HOLISTIC")

    def test_evaluation_applies_gates_and_reports_pass_fail(self) -> None:
        self.config["routes"]["english-holistic"].update({"repeats": 1, "folds": 2})
        rows = [{
            "id": f"h-{score}-{copy}", "route": "english-holistic", "expert_score": score, "hard_score": None,
            "features": {"det_signal": float(score), "det_copy": float(copy)},
        } for score in range(1, 10) for copy in range(2)]
        rows += [{
            "id": f"blank-{copy}", "route": "english-holistic", "expert_score": 0, "hard_score": 0,
            "features": {"det_signal": 0.0, "det_copy": float(copy)},
        } for copy in range(3)]
        report, predictions = evaluate(rows, self.config)
        route = report["routes"]["english-holistic"]
        self.assertEqual(route["range_labels"], ["Fail", "Pass"])
        self.assertEqual((route["n"], route["n_hard_gated"]), (21, 3))
        for row in predictions:
            if row["id"].startswith("blank"):
                self.assertEqual(row["predicted_score"], 0)
            else:
                self.assertGreaterEqual(row["predicted_score"], 1)


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
