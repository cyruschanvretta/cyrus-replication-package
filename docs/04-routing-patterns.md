# Routing patterns

## Router order

1. Normalize language to English or French and skill to upper case.
2. Resolve the route whose `language` and `skill` match the row (see
   "Configuring skills, gates, and routes" below).
3. Extract deterministic measurements once.
4. Apply the configured hard gates.
5. Invoke the supplied model only when that route requires a descriptive
   profile and the response was not gated.
6. Expand the profile to numeric ordinal and threshold features.
7. Predict the complete Stage-2 score with the route's learner.
8. Project the score to the skill's configured ranges (H/M/L for OSSLT/TPCL).

No routing decision may use an item id, known score, expert label, demographic
attribute, response-set membership, or similarity to a remembered answer.

## Hard-gate route

The default (OSSLT/TPCL) configuration defines these gates:

- Blank or authoritatively non-linguistic: Stage-2 score 0.
- CV with fewer than 30 student-authored words: Stage-2 score 1.
- Otherwise continue to the quadrant model.

The caller is responsible for excluding copied prompt words from the authored
word count when necessary.

## Evidence routes

- `english-td`: deterministic → depth-4 forest.
- `english-cv`: deterministic + compact CV profile → tree/ordinal median vote.
- `french-td`: deterministic + compact TD profile → tree/ordinal expected vote.
- `french-cv`: deterministic → cumulative ordinal median.

## Model transport routes

`bedrock` is the default. It calls the Amazon Bedrock Converse API, which
normalizes one request and response shape across model vendors, so no payload
mode has to be chosen. Model id, region, and credentials come from environment
variables; there is no endpoint to provision.

`sagemaker` invokes a named SageMaker runtime endpoint and supports either a
chat-message payload or a TGI-style prompt payload. Endpoint name and region
come from environment variables.

`http` sends the same provider-neutral request to a configured HTTPS endpoint.
An optional bearer token is read from an environment variable.

`ollama` sends a non-streaming chat request to a local Ollama server. The
runtime URL and local model tag are independently configurable, so the default
Gemma 3 12B model can be mapped to the local `gemma3:12b` tag without changing
the scoring prompts.

`callable` imports `module:function`, allowing an in-process model server,
local inference wrapper, test harness, or organization-specific SDK.

`mock` is for contract tests only and must never be used for scoring evidence.

## Failure and retry route

Retry only transport failures or invalid compact output. Keep the first valid
profile and stop; retries are not voting replicates. If all retries fail, mark
the run inconclusive and do not silently route to a different model. Store a
manifest, but never store credentials or student text in context materials.

## Score projection

The default (OSSLT/TPCL) configuration projects:

- TD: 0–2 Low, 3–4 Medium, 5–6 High.
- CV: 0–1 Low, 2–3 Medium, 4 High.

Direct H/M/L fitting is diagnostic only. Operational replication is always
score first.

## Configuring skills, gates, and routes

Assessment rules live in the config file, not in code. `config/default.yaml`
holds OSSLT/TPCL; `config/caec.yaml` holds CAEC. Select one with `--config`.

### `skills`

Each skill (the score scale being reported) declares its score values and
reporting ranges, listed low to high. A range gives either `max` (highest
score, inclusive) or an explicit `scores` list. Labels are free text. If
`ranges` is omitted, each score is its own category and metrics are computed
on exact scores.

```yaml
skills:
  HOLISTIC:
    score_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ungated_min_score: 1        # optional floor for model predictions on ungated rows
    ranges:
      - {label: Fail, max: 4}
      - {label: Pass, max: 9}
```

Any skill defined here is supported. At load time the package checks that
every score value falls in exactly one range and that labels are unique.

### `hard_gates`

An ordered list; the first matching gate fixes the score and its `name` is
recorded as `hard_gate_reason`. `skills` or `routes` optionally restrict a
gate. A condition reads one source — `field` (input row), `feature`
(deterministic feature map), or `measure: word_count` — and applies one
operator: `equals`, `not_equals`, `in`, `lt`, `le`, `gt`, `ge`, or `blank`.
Combine conditions with `all:` or `any:`. A missing value never matches.

```yaml
hard_gates:
  - name: cv-under-30-authored-words
    score: 1
    skills: [CV]
    when: {measure: word_count, lt: 30}
```

### `routes`

The key is the route name. `language` and `skill` select the rows. A row with
no `skill` goes to the only route for its language; if the language has more
than one route, the row must name its skill. Per route you can switch:

- `learner`: `random_forest_depth4`, `cumulative_ordinal_median_c003`,
  `tree_ordinal_median`, or `tree_ordinal_expected`.
- `uses_llm` with a matching `feature_view` (`deterministic` for false,
  `combined` for true).
- `profile_schema` (`TD` or `CV`): the compact LLM profile an LLM route uses.
  It defaults to the route's skill, so it is required only for other skills.

`expected_frozen_feature_count` is optional and is checked only when set.

### `evaluation.apply_hard_gates`

When true, gated rows take their gate score in evaluation and are held out of
model fitting. OSSLT/TPCL leave it off, so evaluation fits every row as in
the reported experiments; CAEC turns it on.
