# French × Conventions

## Frozen approach

This route is deterministic and makes no LLM call. It uses all 261 frozen
French linguistic predictors after variance filtering and score-independent
`SelectKBest(k="all")`, standardizes them inside each fold, and fits one
logistic model for each cumulative Stage-2 score threshold.

Each cumulative boundary uses `C=0.03` and no class weighting. Independently
fitted cumulative probabilities are projected to a monotonic sequence and
decoded by the ordinal median. Stage-2 scores map to Low (0–1), Medium (2–3),
and High (4).

## Evidence

Completed run
`qwen-stage1-range-20260913-v1/french-cv/score-first-stability-selected-v1/g001-select-cumulative-deterministic-kall-c0.03-median`
(n=126) achieved:

- H/M/L consensus 92.06%; minimum repeat 92.06%; mean repeat 92.86%.
- Balanced accuracy 91.80%; macro-F1 92.18%; QWK 0.9252.
- Leave-one-item-out H/M/L 93.65%.
- Exact Stage-2 score 87.30%.
- Confusion matrix `[[33,3,0],[2,54,2],[0,3,29]]`; no non-adjacent misses.

The 94.44% identical-across-all-repeats rate narrowly missed a stricter
research-only 95% stability gate, so this was described as a provisional tuning
freeze. The user-facing 90% repeated-performance target was met.

## Replication notes

Keep French normalization separate from English. Disable item/topic anchors and
do not silently substitute English grammar, dictionary, or readability models.
Strict numerical replication requires the 261-feature contract.
