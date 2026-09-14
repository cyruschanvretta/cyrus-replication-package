#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB))

from awe_scoring.config import load_config
from awe_scoring.evaluation import evaluate
from awe_scoring.io import read_jsonl, sha256_file, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeated out-of-fold evaluation for the four score-first routes")
    parser.add_argument("--input", required=True, help="Feature JSONL created by run.py")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config")
    parser.add_argument("--save-bundle")
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_jsonl(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report, predictions = evaluate(records, config, args.save_bundle)
    report.update({
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(args.input),
        "config_sha256": sha256_file(config["_path"]),
    })
    write_json(output / "summary.json", report)
    write_jsonl(output / "predictions.jsonl", predictions)
    print(json.dumps({"status": "complete", "n": len(records), "output": str(output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
