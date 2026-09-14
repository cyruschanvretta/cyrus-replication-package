#!/usr/bin/env python3
"""Create the package virtual environment and install pinned dependencies."""
from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default=".venv", help="Environment path relative to package root")
    parser.add_argument("--upgrade-pip", action="store_true")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[2]
    environment = (package_root / args.venv).resolve()
    requirements = Path(__file__).with_name("requirements.txt")
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if args.upgrade_pip:
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
    subprocess.run([str(python), "-m", "unittest", "discover", "-s", str(package_root / "lib/tests"), "-v"], check=True)
    print(f"Ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
