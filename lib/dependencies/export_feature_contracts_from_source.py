#!/usr/bin/env python3
"""Mechanically snapshot frozen feature-name contracts from completed summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    package = Path(__file__).resolve().parents[2]
    repository = package.parent
    sources = {
        "english-td": "data-container/prod/runs/stage1-range-evolution/qwen-stage1-range-20260913-v1/evolution/english-td/score-first-stability-v1/g001-forest-deterministic-d4-leaf1/summary.json",
        "english-cv": "data-container/prod/runs/stage1-range-evolution/gemma3-12b-en-cv-20260914-v1/english-cv/matched-feature-v1/g001-score-vote-combined-tree-ordinal-median/summary.json",
        "french-td": "data-container/prod/runs/stage1-range-evolution/qwen-stage1-range-20260913-v1/evolution/french-td/score-first-stability-voting-v1/g001-score-vote-combined-tree-ordinal-expected/summary.json",
        "french-cv": "data-container/prod/runs/stage1-range-evolution/qwen-stage1-range-20260913-v1/evolution/french-cv/score-first-stability-selected-v1/g001-select-cumulative-deterministic-kall-c0.03-median/summary.json",
    }
    output = package / "context_materials/feature_contracts"
    output.mkdir(parents=True, exist_ok=True)
    for route, relative in sources.items():
        path = repository / relative
        summary = json.loads(path.read_text(encoding="utf-8"))
        names = summary["metrics"]["feature_names"]
        contract = {
            "schema_version": 1,
            "route": route,
            "source_run_id": summary["run_id"],
            "source_summary_sha256": sha256(path),
            "feature_count": len(names),
            "feature_names": names,
        }
        (output / f"{route}.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        print(f"{route}: {len(names)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
