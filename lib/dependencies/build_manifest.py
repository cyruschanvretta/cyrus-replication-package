#!/usr/bin/env python3
"""Build a deterministic integrity manifest for the Cyrus handoff package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PACKAGE_ROOT / "MANIFEST.json"
# VCS internals plus the paths .gitignore excludes: machine-local and generated
# files never ship with the handoff, so hashing them would make the manifest
# non-deterministic across workstations.
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "artifacts"}
EXCLUDED_NAMES = {OUTPUT_PATH.name, ".env"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if path.name in EXCLUDED_NAMES or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "package": "cyrus-replication-package",
        "recommended_llm": "Gemma 3 12B Instruct",
        "provider_neutral": True,
        "responses_directory_expected_empty": True,
        "file_count": len(files),
        "files": files,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(files)} file records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
