from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Learner names accepted by `routes.<name>.learner`; see build_estimator.
LEARNERS = (
    "random_forest_depth4",
    "cumulative_ordinal_median_c003",
    "tree_ordinal_median",
    "tree_ordinal_expected",
)


class CumulativeOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, c: float = 1.0, class_weight: str | None = None, random_state: int = 0):
        self.c = c
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CumulativeOrdinalClassifier":
        observed = np.unique(y).astype(int)
        if not np.array_equal(observed, np.arange(len(observed))):
            raise ValueError(f"Ordinal targets must be contiguous from zero; got {observed.tolist()}")
        self.boundaries_ = []
        for threshold in range(1, len(observed)):
            model = LogisticRegression(
                C=self.c,
                class_weight=self.class_weight,
                max_iter=3000,
                random_state=self.random_state,
            )
            model.fit(x, (y >= threshold).astype(int))
            self.boundaries_.append(model)
        self.classes_ = observed
        return self

    def predict_cumulative_proba(self, x: np.ndarray) -> np.ndarray:
        cumulative = np.column_stack([model.predict_proba(x)[:, 1] for model in self.boundaries_])
        return np.minimum.accumulate(cumulative, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cumulative = self.predict_cumulative_proba(x)
        columns = [1.0 - cumulative[:, 0]]
        columns.extend(cumulative[:, index - 1] - cumulative[:, index] for index in range(1, cumulative.shape[1]))
        columns.append(cumulative[:, -1])
        return np.clip(np.column_stack(columns), 0.0, 1.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(int)


class MedianCumulativeOrdinalClassifier(CumulativeOrdinalClassifier):
    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_cumulative_proba(x) >= 0.5).sum(axis=1).astype(int)


class ScoreVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, members: str = "tree-ordinal", decode: str = "median", depth: int = 7, random_state: int = 0):
        self.members = members
        self.decode = decode
        self.depth = depth
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScoreVotingClassifier":
        observed = np.unique(y).astype(int)
        if not np.array_equal(observed, np.arange(len(observed))):
            raise ValueError(f"Score targets must be contiguous from zero; got {observed.tolist()}")
        factories = {
            "tree": lambda: ExtraTreesClassifier(
                n_estimators=500,
                max_depth=self.depth,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,
            ),
            "ordinal": lambda: CumulativeOrdinalClassifier(c=0.02, class_weight=None, random_state=self.random_state),
        }
        names = self.members.split("-")
        if not names or any(name not in factories for name in names):
            raise ValueError(f"Unsupported voting members: {self.members}")
        self.models_ = [factories[name]() for name in names]
        for model in self.models_:
            model.fit(x, y)
        self.classes_ = observed
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = np.mean([model.predict_proba(x) for model in self.models_], axis=0)
        totals = probabilities.sum(axis=1, keepdims=True)
        return probabilities / np.where(totals == 0, 1.0, totals)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        if self.decode == "expected":
            expected = probabilities @ self.classes_.astype(float)
            return np.clip(np.rint(expected), self.classes_[0], self.classes_[-1]).astype(int)
        if self.decode == "median":
            return (np.cumsum(probabilities, axis=1) < 0.5).sum(axis=1).astype(int)
        return np.argmax(probabilities, axis=1).astype(int)


def build_estimator(learner: str, seed: int) -> Any:
    if learner == "random_forest_depth4":
        return RandomForestClassifier(
            n_estimators=350,
            max_depth=4,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
    if learner == "cumulative_ordinal_median_c003":
        return make_pipeline(
            VarianceThreshold(),
            SelectKBest(score_func=f_classif, k="all"),
            StandardScaler(),
            MedianCumulativeOrdinalClassifier(c=0.03, class_weight=None, random_state=seed),
        )
    if learner in {"tree_ordinal_median", "tree_ordinal_expected"}:
        decode = "median" if learner.endswith("median") else "expected"
        return make_pipeline(
            StandardScaler(),
            ScoreVotingClassifier(members="tree-ordinal", decode=decode, depth=7, random_state=seed),
        )
    raise ValueError(f"Unknown learner: {learner}")
