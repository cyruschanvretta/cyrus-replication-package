# Replication protocol

1. Record Python, dependency, model-endpoint, deterministic-extractor, context,
   and configuration hashes before running.
2. Verify that the evaluation cohort is disjoint from every holdout and
   protected cohort programmatically.
3. Put source responses only in `responses/`; do not copy them into prompts,
   documentation, logs, or context materials.
4. Run `run.py` once. Preserve its manifest and feature JSONL. Do not repeat
   successful model calls to manufacture agreement.
5. Run `evaluate.py`. Use the route's fixed repeat/fold design. All transforms
   must be fitted inside each fold.
6. Quote the run id and n with every metric. Compare systems row-by-row and use
   a paired bootstrap interval before calling a change an improvement.
7. Keep failed, interrupted, and timed-out attempts marked `inconclusive`.
8. Do not inspect or tune on a holdout. Score it once only after the candidate,
   feature registry, prompts, learner, cuts, and seed are frozen.

## Strict versus functional replication

Strict replication requires the original feature names in
`context_materials/feature_contracts/`, matching deterministic extractor
versions, matching labelled cohorts, and the route's documented learner.

Functional replication uses the built-in portable feature extractor. It tests
the model adapter, routing, score-first design, and artifact flow, but it cannot
be cited as a reproduction of the original accuracy.
