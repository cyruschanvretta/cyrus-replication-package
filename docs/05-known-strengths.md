# Known strengths and limitations

## Strengths

- The architecture is trait- and language-specific instead of forcing one
  generalized prompt or learner across four different tasks.
- English TD is highly accurate without model inference: completed run
  `qwen-stage1-range-20260913-v1/english-td/score-first-stability-v1/g001-forest-deterministic-d4-leaf1`
  (n=419) reached 96.42% H/M/L and 91.17% exact score.
- French CV is also model-free: completed run
  `qwen-stage1-range-20260913-v1/french-cv/score-first-stability-selected-v1/g001-select-cumulative-deterministic-kall-c0.03-median`
  (n=126) reached 92.06% H/M/L and 87.30% exact score.
- French TD's combined architecture captures complementary content and
  linguistic evidence: completed Qwen-based run
  `qwen-stage1-range-20260913-v1/french-td/score-first-stability-voting-v1/g001-score-vote-combined-tree-ordinal-expected`
  (n=137) reached 91.24% H/M/L with no non-adjacent misses.
- Gemma 12B improved the English-CV Gemma family: completed run
  `gemma3-12b-en-cv-20260914-v1/english-cv/matched-feature-v1/g001-score-vote-combined-tree-ordinal-median`
  (n=395) reached 84.05% H/M/L, 3.29 points above completed Gemma 4B run
  `gemma3-4b-en-cv-20260914-v1/english-cv/matched-feature-v1/g001-score-vote-combined-tree-ordinal-median`
  (n=395), paired-bootstrap 95% CI [+0.25, +6.33].
- Every selected architecture predicts Stage-2 first and produced no
  non-adjacent H/M/L consensus errors in its cited run.
- Model calls are descriptive, compact, label-blind, cacheable, and made once.

## Limitations

- All cited results are tuning evidence. dev-60 and protected level-2 were not
  used, so the package must not label them production accuracy.
- English CV remains below 90%; its strongest unresolved component is the
  Medium/High boundary.
- Gemma 12B's +1.52-point English-CV H/M/L result over Qwen is unresolved, 95%
  CI [-1.52, +4.81].
- Gemma 12B strongly compressed the upper end of the profile scale.
- French TD leave-one-item-out H/M/L was 83.94%, below the 90% target.
- The Gemma 12B substitution is untested in French TD.
- Portable deterministic features are not equivalent to the frozen production
  feature extractor. Strict replication needs matching feature contracts and
  versions.
- Automated linguistic flags can be wrong and are never score proxies.
