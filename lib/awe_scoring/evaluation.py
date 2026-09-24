from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .estimators import build_estimator
from .routing import compile_spec


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in rows], dtype=float)


def _metrics(actual: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(actual, predicted, labels=labels)
    indices = {label: index for index, label in enumerate(labels)}
    a = [indices[value] for value in actual]
    p = [indices[value] for value in predicted]
    return {
        "n": len(actual),
        "exact_rate": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=labels, average="macro"),
        "quadratic_weighted_kappa": cohen_kappa_score(a, p, weights="quadratic"),
        "confusion_actual_rows_predicted_columns": matrix.tolist(),
    }


def evaluate(records: list[dict[str, Any]], config: dict[str, Any], save_bundle: str | Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("expert_score") is None:
            raise ValueError(f"Record {row.get('id')} is missing expert_score")
        grouped.setdefault(row["route"], []).append(row)
    report: dict[str, Any] = {"schema_version": 1, "routes": {}}
    predictions: list[dict[str, Any]] = []
    fitted_routes: dict[str, Any] = {}
    seed = int(config.get("seed", 20260913))
    spec = compile_spec(config)
    apply_hard_gates = bool((config.get("evaluation") or {}).get("apply_hard_gates", False))
    for route_key, rows in sorted(grouped.items()):
        compiled = spec.routes[route_key]
        route = compiled.settings
        skill = compiled.skill
        scores = np.asarray([int(row["expert_score"]) for row in rows], dtype=int)
        invalid = sorted({int(score) for score in scores} - set(skill.score_values))
        if invalid:
            raise ValueError(f"{route_key} expert scores {invalid} are not {skill.name} score values {list(skill.score_values)}")
        # With apply_hard_gates, gated rows take their gate score and are held out of fitting.
        gated = np.asarray([apply_hard_gates and row.get("hard_score") is not None for row in rows], dtype=bool)
        model_index = np.flatnonzero(~gated)
        model_rows = [rows[index] for index in model_index]
        if not model_rows:
            raise ValueError(f"{route_key} has no ungated rows to fit")
        feature_names = sorted(set().union(*(row["features"].keys() for row in model_rows)))
        expected = route.get("expected_frozen_feature_count")
        if config["deterministic"].get("mode") == "provided" and expected is not None and len(feature_names) != int(expected):
            raise ValueError(
                f"{route_key} strict evaluation expected {expected} features, found {len(feature_names)}"
            )
        x = _matrix(model_rows, feature_names)
        model_scores = scores[model_index]
        score_min = int(model_scores.min())
        y = model_scores - score_min
        floored = np.asarray([row.get("hard_score") is None for row in model_rows], dtype=bool)
        observed = np.unique(y)
        if not np.array_equal(observed, np.arange(len(observed))):
            raise ValueError(f"{route_key} scores must be contiguous; got {(observed + score_min).tolist()}")
        repeats = int(route["repeats"])
        folds = int(route["folds"])
        repeat_predictions: list[np.ndarray] = []
        per_repeat: list[float] = []
        for repeat in range(repeats):
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
            model_oof = np.full(len(model_rows), -1, dtype=int)
            for train, test in splitter.split(x, y):
                estimator = build_estimator(route["learner"], seed + repeat)
                estimator.fit(x[train], y[train])
                model_oof[test] = estimator.predict(x[test]).astype(int) + score_min
            if skill.ungated_min_score is not None:
                model_oof = np.where(floored, np.maximum(model_oof, skill.ungated_min_score), model_oof)
            oof = np.asarray([int(row.get("hard_score") or 0) for row in rows], dtype=int)
            oof[model_index] = model_oof
            repeat_predictions.append(oof)
            per_repeat.append(float(np.mean(oof == scores)))
        stacked = np.vstack(repeat_predictions)
        consensus_scores = np.asarray([_mode(stacked[:, index].tolist()) for index in range(len(rows))])
        actual_ranges = [skill.to_range(score) for score in scores]
        consensus_ranges = [skill.to_range(score) for score in consensus_scores]
        range_repeat = [
            [skill.to_range(value) for value in repeat]
            for repeat in repeat_predictions
        ]
        consistency = np.mean([
            max(Counter(repeat[index] for repeat in range_repeat).values()) / repeats
            for index in range(len(rows))
        ])
        route_report = {
            "n": len(rows),
            "learner": route["learner"],
            "skill": skill.name,
            "range_labels": skill.labels,
            "feature_count": len(feature_names),
            "expected_frozen_feature_count": None if expected is None else int(expected),
            "strict_feature_count_match": None if expected is None else len(feature_names) == int(expected),
            "repeats": repeats,
            "folds": folds,
            "stage2_per_repeat_exact_rates": per_repeat,
            "stage2_consensus_exact_rate": float(np.mean(consensus_scores == scores)),
            "hml_consensus": _metrics(actual_ranges, consensus_ranges, skill.labels),
            "mean_hml_prediction_consistency": float(consistency),
        }
        if apply_hard_gates:
            route_report["n_hard_gated"] = int(gated.sum())
        report["routes"][route_key] = route_report
        predictions.extend({
            "id": row["id"],
            "route": route_key,
            "expert_score": int(scores[index]),
            "predicted_score": int(consensus_scores[index]),
            "expert_hml": actual_ranges[index],
            "predicted_hml": consensus_ranges[index],
            "repeat_scores": [int(repeat[index]) for repeat in repeat_predictions],
        } for index, row in enumerate(rows))
        fitted = build_estimator(route["learner"], seed)
        fitted.fit(x, y)
        fitted_routes[route_key] = {
            "learner": route["learner"],
            "feature_names": feature_names,
            "score_min": score_min,
            "estimator": fitted,
        }
    if save_bundle:
        target = Path(save_bundle)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": 1, "seed": seed, "routes": fitted_routes}, target)
    return report, predictions
