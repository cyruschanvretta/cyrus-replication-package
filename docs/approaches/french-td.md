# French × Topic Development

## Frozen approach

This route combines an LLM-generated ten-field TD profile with deterministic
French linguistic features, producing 240 frozen predictors. The original
completed evidence used a local Qwen profile. This handoff recommends Gemma 3
12B as the default supplied model, but that substitution has not yet been
validated for French TD and must be treated as a replication question.

The Stage-2 0–6 score uses the same ExtraTrees and cumulative-ordinal members as
English CV, but decodes the averaged class distribution to its rounded expected
score. TD scores map to Low (0–2), Medium (3–4), and High (5–6).

The ten model fields are task engagement, position stability, support breadth,
specificity, support linkage, explanation depth, organization, idea
advancement, functional completeness, and internal consistency. The task
prompt is ephemeral per response because task engagement cannot be assessed
without it.

## Existing architecture evidence

Completed Qwen-based run
`qwen-stage1-range-20260913-v1/french-td/score-first-stability-voting-v1/g001-score-vote-combined-tree-ordinal-expected`
(n=137) achieved:

- H/M/L consensus 91.24%; minimum repeat 91.24%; mean repeat 91.53%.
- Balanced accuracy 91.46%; macro-F1 91.52%; QWK 0.9267.
- Leave-one-item-out H/M/L 83.94%.
- Exact Stage-2 score 74.45%.
- Confusion matrix `[[40,5,0],[2,49,3],[0,2,36]]`; no non-adjacent misses.

These numbers belong to the Qwen-based completed run. They must not be quoted
as Gemma 12B performance.

## Replication gate

Run Gemma 12B on a French-labelled tuning cohort, compare paired predictions to
the frozen Qwen profiles, and inspect per-field scale use. Do not promote it in
this quadrant unless repeated H/M/L, exact score, and unseen-item stress remain
acceptable.
