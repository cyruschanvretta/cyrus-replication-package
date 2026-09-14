from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve() if path else package_root() / "config/default.yaml"
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {source}")
    config["_path"] = str(source)
    return config
