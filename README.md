# AWE four-quadrant scoring replication package

This handoff packages the current score-first architectures for English/French
Topic Development (TD) and Conventions (CV). It contains no student responses.
Place Cyrus's response files in `responses/` and keep generated artifacts outside
that folder.

The recommended default model for routes that require an LLM is **Gemma 3 12B
Instruct**. The runtime is provider-neutral: the model may be supplied through
AWS Bedrock, an AWS SageMaker endpoint, a generic HTTP chat endpoint, local
Ollama, or a Python callable. The two deterministic routes do not invoke the
model.

`MANIFEST.json` records the package file inventory and SHA-256 hashes. Rebuild
it after an intentional handoff edit with
`python lib/dependencies/build_manifest.py`.

## Quick start

Use Python 3.11 (the package was verified with Python 3.11.5). Then initialize
the pinned virtual environment:

```powershell
python lib/dependencies/initialize.py
```

1. Copy `.env.example` to `.env` and configure the selected adapter. The default
   adapter is Bedrock: set `BEDROCK_MODEL_ID`, `AWS_REGION`, and either
   `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or an instance role or profile.
   For SageMaker instead, set `model.adapter: sagemaker`,
   `AWE_SAGEMAKER_ENDPOINT`, and `AWS_REGION`.
2. Add labelled JSONL to `responses/replication.jsonl`; see
   `docs/06-input-output-contracts.md`.
3. Extract deterministic and LLM features once:

   ```powershell
   .venv/Scripts/python lib/scripts/run.py --input responses/replication.jsonl --output artifacts/replication-features.jsonl
   ```

4. Evaluate and save fitted route models:

   ```powershell
   .venv/Scripts/python lib/scripts/evaluate.py --input artifacts/replication-features.jsonl --output artifacts/evaluation --save-bundle artifacts/models.joblib
   ```

5. Score new responses with the saved bundle:

   ```powershell
   .venv/Scripts/python lib/scripts/run.py --input responses/new.jsonl --output artifacts/new-scores.jsonl --model-bundle artifacts/models.joblib
   ```

For a fully offline run with no AWS account, add `--adapter ollama`. Use
`--model-name gemma3:12b` to override the local tag and `--concurrency 1` on a
memory-constrained workstation.

On macOS/Linux, use `.venv/bin/python` instead. Run `python -m unittest discover
-s lib/tests -v` for the local contract tests.

## Important evidence boundary

The package reproduces the routing and learner families. Reproducing the
reported metrics also requires the frozen deterministic feature definitions and
the same labelled cohorts. The built-in `portable` extractor is intended for
integration tests and new training; it is not a claim of numerical parity with
the 132/240/261/304-feature experiments. For strict replication, provide each
row's exact numeric `deterministic_features` and use
`deterministic.mode: provided`.

Start with `docs/01-overview.md` and `docs/07-replication-protocol.md`.
