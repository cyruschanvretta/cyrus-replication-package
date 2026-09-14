# English × Conventions

## Recommended approach

This route makes one Gemma 3 12B Instruct call for each response that survives
the deterministic gates. The model emits a compact descriptive profile; it
does not emit a score or rationale. Fifty-four deterministic expansions of the
profile are combined with 250 deterministic linguistic measurements, yielding
the frozen 304-predictor matrix.

Profile fields are intelligibility (`I`), repair scope (`R`), grammar (`G`),
sentence/syntax control (`S`), usage (`U`), punctuation (`P`), orthography
(`O`), syntactic ambition (`A`), and consistency (`C`). Error dispersion (`D`)
is supplied by deterministic software. Each field is ordinal 0–3.

The Stage-2 1–4 score is predicted by an equal-probability vote between:

- ExtraTrees: 500 trees, depth 7, leaf 1, square-root feature sampling,
  balanced class weights.
- Cumulative ordinal logistic models: one `Y >= threshold` model per score
  boundary, `C=0.02`, with monotonic cumulative probabilities.

The averaged score distribution is decoded by its ordinal median. The score is
then mapped to Low (0–1), Medium (2–3), or High (4).

## Evidence

Completed run
`gemma3-12b-en-cv-20260914-v1/english-cv/matched-feature-v1/g001-score-vote-combined-tree-ordinal-median`
(n=395) achieved:

- H/M/L consensus 84.05%; minimum repeat 82.78%; mean repeat 83.71%.
- Balanced accuracy 82.61%; macro-F1 83.75%; QWK 0.8281.
- Leave-one-item-out H/M/L 81.77%.
- Exact Stage-2 score 74.68%.
- Confusion matrix `[[74,18,0],[5,180,20],[0,20,78]]`; no non-adjacent misses.

Versus completed Gemma 4B run
`gemma3-4b-en-cv-20260914-v1/english-cv/matched-feature-v1/g001-score-vote-combined-tree-ordinal-median`
(n=395), H/M/L improved 3.29 points, paired-bootstrap 95% CI [+0.25,
+6.33]. Versus completed Qwen incumbent
`qwen-stage1-range-20260913-v1/english-cv/score-first-voting-v1/g009-score-vote-combined-tree-ordinal-median`
(n=395), H/M/L improved 1.52 points but remained unresolved, 95% CI [-1.52,
+4.81].

## Known limitation

Gemma 12B assigned profile value 3 only once across 3,429 model-generated field
judgements in the completed profile run. The downstream learner extracted a
useful signal, but upper-scale compression remains a model/prompt weakness.
Do not recalibrate against a holdout. Test any prompt correction first on a
separate tuning cohort and preserve the current prompt as the comparator.
