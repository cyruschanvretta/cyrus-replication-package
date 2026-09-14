# Input and output contracts

## Response JSONL

Each line must contain:

```json
{"id":"r-001","language":"en","skill":"CV","response_text":"..."}
```

Optional fields:

- `task_prompt`: required for TD profile routes; intentionally omitted from CV
  model requests.
- `expert_score`: integer 0–6 for TD or 0–4 for CV; required by `evaluate.py`.
- `is_linguistic`: set false only when upstream software has authoritatively
  identified a non-linguistic response.
- `deterministic_features`: exact numeric feature mapping. Supply this for
  evidence-equivalent replication.

Language accepts `en`, `english`, `fr`, or `french`. Skill accepts `TD` or `CV`.

## Feature JSONL from `run.py`

The runner writes no response text. Each output row contains the response id,
route, hard-gate result, numeric feature map, optional compact LLM profile,
optional expert score, and optional prediction from a supplied model bundle.

## Model output protocol

The model must return one line and no prose.

- TD: `T#|P#|B#|S#|L#|E#|O#|A#|F#|C#`
- CV: `I#|R#|G#|S#|U#|P#|O#|A#|C#`

Every `#` is replaced by one digit from 0 through 3. CV error dispersion `D`
is calculated by deterministic software and is not emitted by the model.

## Evaluation output

`evaluate.py` writes `summary.json`, `predictions.jsonl`, and optionally a
joblib model bundle. Metrics include every repeat, consensus H/M/L accuracy,
balanced accuracy, macro-F1, quadratic weighted kappa, exact Stage-2 score,
prediction stability, confusion matrix, and row counts.
