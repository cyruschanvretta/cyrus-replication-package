from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import deterministic_features, error_dispersion
from .model_adapters import ModelAdapter, build_adapter
from .profiles import build_messages, parse_profile, profile_features
from .routing import hard_gate, normalize_language, normalize_skill, route_key, score_to_range


def _word_count(features: dict[str, float], text: str) -> int:
    for name in ("det_builtin_word_count", "det_portable_word_count"):
        if name in features:
            return int(features[name])
    return len(text.split())


def _align(feature_map: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(feature_map.get(name, 0.0)) for name in feature_names]], dtype=float)


def _validate_strict_contract(route_key_value: str, features: dict[str, float], llm_expected: bool) -> None:
    path = Path(__file__).resolve().parents[2] / "context_materials/feature_contracts" / f"{route_key_value}.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = set(contract["feature_names"])
    actual = set(features)
    unexpected = sorted(actual - expected)
    required = expected if llm_expected else {name for name in expected if name.startswith("det_")}
    missing = sorted(required - actual)
    if unexpected or missing:
        raise ValueError(
            f"{route_key_value} strict feature contract mismatch: "
            f"missing={missing[:8]} ({len(missing)}), unexpected={unexpected[:8]} ({len(unexpected)})"
        )


def _profile_once(
    row: dict[str, Any],
    deterministic: dict[str, float],
    route: dict[str, Any],
    adapter: ModelAdapter,
    retries: int,
) -> tuple[dict[str, int], int, str]:
    messages = build_messages(row, deterministic, route)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = adapter.invoke(messages)
            profile = parse_profile(raw, row["skill"], error_dispersion(deterministic))
            return profile, attempt, raw.strip().splitlines()[0]
        except Exception as error:
            last_error = error
            messages = [
                messages[0],
                {
                    "role": "user",
                    "content": messages[1]["content"]
                    + "\n\nPROTOCOL REPAIR: Return only the requested compact field line. No explanation.",
                },
            ]
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def process_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    adapter_override: str | None = None,
    bundle_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Every input row requires a unique non-empty id")
    routes = config["routes"]
    needs_llm = any(routes[route_key(row["language"], row["skill"])]["uses_llm"] for row in rows)
    adapter = build_adapter(config["model"], adapter_override) if needs_llm else None
    bundle = joblib.load(bundle_path) if bundle_path else None

    def work(source: dict[str, Any]) -> dict[str, Any]:
        row = dict(source)
        row["language"] = normalize_language(row["language"])
        row["skill"] = normalize_skill(row["skill"])
        key = route_key(row["language"], row["skill"])
        route = routes[key]
        deterministic = deterministic_features(row, config["deterministic"])
        hard_score, hard_reason = hard_gate(row, _word_count(deterministic, str(row.get("response_text", ""))))
        profile = None
        accepted_output = None
        attempts = 0
        feature_map = dict(deterministic)
        if route["uses_llm"] and hard_score is None:
            assert adapter is not None
            profile, attempts, accepted_output = _profile_once(
                row, deterministic, route, adapter, int(config["model"].get("retries", 4))
            )
            feature_map.update(profile_features(row["skill"], profile))
        if config["deterministic"].get("mode") == "provided":
            _validate_strict_contract(key, feature_map, bool(route["uses_llm"] and hard_score is None))
        predicted_score = hard_score
        scoring_source = "hard-gate" if hard_score is not None else "features-only"
        if predicted_score is None and bundle:
            fitted = bundle.get("routes", {}).get(key)
            if not fitted:
                raise ValueError(f"Model bundle has no route {key}")
            shifted = int(fitted["estimator"].predict(_align(feature_map, fitted["feature_names"]))[0])
            predicted_score = shifted + int(fitted["score_min"])
            scoring_source = "fitted-score-first-model"
        return {
            "id": row["id"],
            "language": row["language"],
            "skill": row["skill"],
            "route": key,
            "expert_score": row.get("expert_score"),
            "hard_score": hard_score,
            "hard_gate_reason": hard_reason,
            "features": feature_map,
            "llm_profile": profile,
            "accepted_model_output": accepted_output,
            "model_attempts": attempts,
            "stage2_score": predicted_score,
            "hml": score_to_range(predicted_score, row["skill"]) if predicted_score is not None else None,
            "scoring_source": scoring_source,
        }

    concurrency = max(1, int(config["model"].get("concurrency", 1)))
    output: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(work, row): index for index, row in enumerate(rows)}
        for future in as_completed(futures):
            output[futures[future]] = future.result()
    return [row for row in output if row is not None]


def context_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "context_materials").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
