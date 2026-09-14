#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB))

from awe_scoring.config import load_config, package_root
from awe_scoring.io import read_jsonl, sha256_file, write_json, write_jsonl
from awe_scoring.pipeline import context_digest, process_rows


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract four-quadrant features and optionally score with a fitted bundle")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--adapter", choices=["sagemaker", "http", "ollama", "callable", "mock"])
    parser.add_argument("--model-name", help="Deployment or local-runtime model name override")
    parser.add_argument("--concurrency", type=int, help="Concurrent response workers")
    parser.add_argument("--model-bundle")
    args = parser.parse_args()

    root = package_root()
    load_env(root / ".env")
    config = load_config(args.config)
    if args.model_name:
        config["model"]["recommended_family"] = args.model_name
        config["model"].setdefault("ollama", {})["model"] = args.model_name
        config["model"]["ollama"]["_explicit_model_override"] = True
    if args.concurrency is not None:
        if args.concurrency < 1:
            parser.error("--concurrency must be at least 1")
        config["model"]["concurrency"] = args.concurrency
    rows = read_jsonl(args.input)
    started = datetime.now(timezone.utc)
    status = "inconclusive"
    try:
        output = process_rows(rows, config, args.adapter, args.model_bundle)
        status = "complete"
        write_jsonl(args.output, output)
    except Exception:
        raise
    finally:
        manifest = {
            "schema_version": 1,
            "status": status,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "input": str(Path(args.input).resolve()),
            "input_sha256": sha256_file(args.input),
            "config": config["_path"],
            "config_sha256": sha256_file(config["_path"]),
            "context_sha256": context_digest(root),
            "n": len(rows),
            "model_family": config["model"]["recommended_family"],
            "adapter": args.adapter or config["model"]["adapter"],
            "concurrency": config["model"]["concurrency"],
            "model_bundle": str(Path(args.model_bundle).resolve()) if args.model_bundle else None,
            "output": str(Path(args.output).resolve()),
        }
        write_json(str(args.output) + ".manifest.json", manifest)
    print(json.dumps({"status": status, "n": len(rows), "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
