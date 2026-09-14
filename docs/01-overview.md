# Overview

## Objective

Predict the full Stage-2 writing score first, then deterministically project it
to Stage-1 Low/Medium/High. Language and trait select one of four fixed routes;
item identity, response-set membership, expert labels, and response text are
never model predictors.

## Current routing matrix

| Route | Evidence | Stage-2 learner | LLM |
|---|---|---|---|
| English TD | 132 deterministic linguistic features | depth-4 random forest | none |
| English CV | Gemma profile plus deterministic linguistic features; 304 frozen predictors | ExtraTrees + cumulative ordinal vote, median decode | Gemma 3 12B Instruct |
| French TD | descriptive profile plus deterministic linguistic features; 240 predictors | ExtraTrees + cumulative ordinal vote, expected-score decode | Gemma 3 12B default; French validation pending |
| French CV | 261 deterministic linguistic features | cumulative ordinal logistic, median decode | none |

The reported run evidence is response-set tuning evidence. It is not a claim
about the untouched dev-60 holdout, protected level-2 responses, unseen items,
or production performance.

## Why Gemma 3 12B is the default

Completed English-CV run
`gemma3-12b-en-cv-20260914-v1/english-cv/matched-feature-v1/g001-score-vote-combined-tree-ordinal-median`
(n=395) reached 84.05% H/M/L and 74.68% exact Stage-2 score. Relative to the
completed Gemma 4B matched run (n=395), H/M/L improved 3.29 percentage points,
paired-bootstrap 95% CI [+0.25, +6.33]. Relative to the completed local-Qwen
incumbent (n=395), the +1.52-point H/M/L difference remains unresolved, 95% CI
[-1.52, +4.81]. Gemma 12B is therefore the recommended operational default,
not a proven universal replacement across all quadrants.

## Invariants

- One accepted LLM profile per eligible response; retries only recover failed
  transport or invalid protocol output.
- Blank/non-linguistic and CV-under-30-word gates are deterministic.
- The LLM emits descriptive observations, never an expert score.
- All normalization, feature selection, and fitting occur inside training folds.
- TD and CV use different score ranges and cuts.
- English and French retain separate routes and normalization.
- A failed or interrupted run is inconclusive, never a regression.
