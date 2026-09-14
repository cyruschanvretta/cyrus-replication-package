# Routing patterns

## Router order

1. Normalize language to English or French and trait to TD or CV.
2. Apply deterministic hard gates.
3. Resolve the quadrant route from `config/default.yaml`.
4. Extract deterministic measurements once.
5. Invoke the supplied model only when that route requires a descriptive
   profile and the response was not gated.
6. Expand the profile to numeric ordinal and threshold features.
7. Predict the complete Stage-2 score with the quadrant learner.
8. Project the score to H/M/L using fixed cuts.

No routing decision may use an item id, known score, expert label, demographic
attribute, response-set membership, or similarity to a remembered answer.

## Hard-gate route

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

`sagemaker` is the deployment-oriented default. It invokes a named SageMaker
runtime endpoint and supports either a chat-message payload or a TGI-style
prompt payload. Endpoint name and region come from environment variables.

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

- TD: 0–2 Low, 3–4 Medium, 5–6 High.
- CV: 0–1 Low, 2–3 Medium, 4 High.

Direct H/M/L fitting is diagnostic only. Operational replication is always
score first.
