# Heuristics

## Governing rule

Heuristics are observable questions and boundary checks, not direct score
lookups. Numeric measurements may direct attention, but no raw count maps to a
band. The full source inventory—77 YAML files including active indexes,
alternatives, and retired research rules—is preserved under
`context_materials/heuristics/source/`.

## Topic Development

Ask these questions in order:

1. What is the student's controlling opinion? Is it explicit and maintained?
2. What distinct supporting reasons are present?
3. For each reason, what concrete detail or example is supplied?
4. What does each detail establish, and how is it linked to the opinion?
5. Does the writer explain how, why, consequence, or significance?
6. Do later ideas advance the reasoning or merely repeat/parallel earlier ones?
7. Does organization clarify relationships? Formal paragraphs and polished
   conventions are not prerequisites.
8. Which complete adjacent score description is met? Test the one below and
   the one above before deciding.

Load-bearing boundaries:

- 1/2: labels or noun lists need at least one explicit link or analytic move.
- 2/3: generic assertions without usable explanation remain limited.
- 3/4: a clear sustained opinion needs sufficient support; some specificity is
  enough and mechanical organization is permitted.
- 4/5: sufficient support must be consistently specific and logically linked.
- 5/6: support must be thoughtfully chosen and ideas must progress; mechanical
  signposting alone cannot block 6.

The production Stage-1 TD index currently names:
`h-boundary-1-vs-2`, `h-strict-1-when-responses-thin`,
`h-grade-2-requires-analytic-move`,
`h-tighten-2-vs-3-generic-assertion`,
`h-td-named-without-unpacking-floor-2`, `h-td-named-entity-limit`,
`h-td-ignore-mechanics-for-5-6`,
`h-td-prevent-mechanical-penalty-for-6`,
`h-td-int003-mid-range-floor`, `h-td-int003-ceiling-lift-5`, and
`h-td-int003-ceiling-lift-6`. The broader 36-rule TD catalog is retained for
traceability but should not be loaded wholesale because it contains duplicate
constructs.

## Conventions

Ask these questions in order:

1. Can the student's intended meaning be recovered on an ordinary read?
2. Which sentences require rereading or reconstruction, and how broad is the
   repair?
3. Which grammar, sentence-boundary, usage, punctuation, and spelling patterns
   are credible after checking automated flags in context?
4. Are problems isolated, clustered, or distributed through the response?
5. Does the response also contain independently controlled sentences and
   sections?
6. Is the reader's rhythm constantly interrupted, merely distracted, not
   distracted, or consistently controlled?

Active CV rules cover meaning-first judgement, clean-control evidence,
fragment limits, first-draft tolerance, the explicit under-30-word gate,
structural collapse, systemic error burden, repeated spelling, and the rule
that no single error category decides the score. The current index names all
15 active files in `context_materials/heuristics/source/_index-cv.yaml`.

## Measurement safeguards

- Collapse repeats of the same misspelling conceptually.
- Verify LanguageTool-style flags in context; they are fallible evidence.
- Keep positive control evidence as well as error burden.
- Do not use conventions to score TD or idea quality to score CV.
- Keep English and French feature distributions separate.
- Never fit a threshold on a holdout or protected cohort.
