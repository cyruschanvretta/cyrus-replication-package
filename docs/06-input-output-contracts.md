# Input and output contracts

## Response JSONL

Each line must contain:

```json
{"id":"r-001","language":"en","skill":"CV","response_text":"..."}
```

`skill` may be omitted when the configuration has a single route for the
row's language (for example CAEC, where every response has one holistic
score).

Optional fields:

- `task_prompt`: required for TD profile routes; intentionally omitted from CV
  model requests.
- `expert_score`: one of the skill's configured `score_values` (0–6 for TD,
  0–4 for CV, 0–9 for CAEC HOLISTIC); required by `evaluate.py`.
- `is_linguistic`: set false only when upstream software has authoritatively
  identified a non-linguistic response.
- `deterministic_features`: exact numeric feature mapping. Supply this for
  evidence-equivalent replication.

Language accepts `en`, `english`, `fr`, or `french`. Skill accepts any skill
defined under `skills` in the active config (case-insensitive): `TD` or `CV`
in `config/default.yaml`, `HOLISTIC` in `config/caec.yaml`.

## Feature JSONL from `run.py`

The runner writes no response text. Each output row contains the response id,
route, hard-gate result, numeric feature map, optional compact LLM profile,
optional expert score, and optional prediction from a supplied model bundle.
`hml` is the predicted score's range label from the skill's configured ranges.

## Model output protocol

The model must return one line and no prose.

- TD: `T#|P#|B#|S#|L#|E#|O#|A#|F#|C#`
- CV: `I#|R#|G#|S#|U#|P#|O#|A#|C#`

Every `#` is replaced by one digit from 0 through 3. CV error dispersion `D`
is calculated by deterministic software and is not emitted by the model.

## Evaluation output

`evaluate.py` writes `summary.json`, `predictions.jsonl`, and optionally a
joblib model bundle. Each route reports its `skill` and `range_labels`.
Metrics include every repeat, consensus range accuracy,
balanced accuracy, macro-F1, quadratic weighted kappa, exact Stage-2 score,
prediction stability, confusion matrix, and row counts.
