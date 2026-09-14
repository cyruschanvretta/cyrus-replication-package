# English × Topic Development

## Frozen approach

This route does not use an LLM. It predicts the 0–6 Stage-2 TD score with a
random forest over 132 question-invariant deterministic linguistic features,
then maps 0–2 to Low, 3–4 to Medium, and 5–6 to High.

Estimator:

- `RandomForestClassifier`
- 350 trees
- maximum depth 4
- minimum leaf size 1
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- fixed seed; one CPU thread per fit

Features describe length, sentence shape, lexical diversity, readability,
connectives, repetition, grammar/spelling flags, syntax, and related linguistic
structure. They do not include item id, task text, response text, or labels.

## Evidence

Completed run
`qwen-stage1-range-20260913-v1/english-td/score-first-stability-v1/g001-forest-deterministic-d4-leaf1`
(n=419) achieved:

- H/M/L consensus 96.42%; minimum repeat 96.18%; mean repeat 96.37%.
- Balanced accuracy 96.45%; macro-F1 96.49%; QWK 0.9712.
- Leave-one-item-out H/M/L 96.18%.
- Exact Stage-2 score 91.17%.
- Confusion matrix `[[119,5,0],[3,151,4],[0,3,134]]`; no non-adjacent misses.

These are response-set tuning metrics, not holdout or production metrics.

## Replication notes

Use route `english-td` and do not initialize a model adapter when the input
contains only this route. Strict numerical replication requires the exact 132
features listed in `context_materials/feature_contracts/english-td.json`.
Surface correctness must never be used as a proxy for idea development.
