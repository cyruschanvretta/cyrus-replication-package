from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .estimators import build_estimator
from .routing import RANGE_LABELS, score_to_range


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in rows], dtype=float)


def _metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = list(RANGE_LABELS)
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
    for route_key, rows in sorted(grouped.items()):
        route = config["routes"][route_key]
        feature_names = sorted(set().union(*(row["features"].keys() for row in rows)))
        expected = int(route.get("expected_frozen_feature_count", 0))
        if config["deterministic"].get("mode") == "provided" and len(feature_names) != expected:
            raise ValueError(
                f"{route_key} strict evaluation expected {expected} features, found {len(feature_names)}"
            )
        x = _matrix(rows, feature_names)
        scores = np.asarray([int(row["expert_score"]) for row in rows], dtype=int)
        score_min = int(scores.min())
        y = scores - score_min
        observed = np.unique(y)
        if not np.array_equal(observed, np.arange(len(observed))):
            raise ValueError(f"{route_key} scores must be contiguous; got {(observed + score_min).tolist()}")
        repeats = int(route["repeats"])
        folds = int(route["folds"])
        repeat_predictions: list[np.ndarray] = []
        per_repeat: list[float] = []
        for repeat in range(repeats):
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
            oof = np.full(len(rows), -1, dtype=int)
            for train, test in splitter.split(x, y):
                estimator = build_estimator(route["learner"], seed + repeat)
                estimator.fit(x[train], y[train])
                oof[test] = estimator.predict(x[test]).astype(int) + score_min
            repeat_predictions.append(oof)
            per_repeat.append(float(np.mean(oof == scores)))
        stacked = np.vstack(repeat_predictions)
        consensus_scores = np.asarray([_mode(stacked[:, index].tolist()) for index in range(len(rows))])
        actual_ranges = [score_to_range(score, rows[index]["skill"]) for index, score in enumerate(scores)]
        consensus_ranges = [score_to_range(score, rows[index]["skill"]) for index, score in enumerate(consensus_scores)]
        range_repeat = [
            [score_to_range(value, rows[index]["skill"]) for index, value in enumerate(repeat)]
            for repeat in repeat_predictions
        ]
        consistency = np.mean([
            max(Counter(repeat[index] for repeat in range_repeat).values()) / repeats
            for index in range(len(rows))
        ])
        route_report = {
            "n": len(rows),
            "learner": route["learner"],
            "feature_count": len(feature_names),
            "expected_frozen_feature_count": expected,
            "strict_feature_count_match": len(feature_names) == expected,
            "repeats": repeats,
            "folds": folds,
            "stage2_per_repeat_exact_rates": per_repeat,
            "stage2_consensus_exact_rate": float(np.mean(consensus_scores == scores)),
            "hml_consensus": _metrics(actual_ranges, consensus_ranges),
            "mean_hml_prediction_consistency": float(consistency),
        }
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
