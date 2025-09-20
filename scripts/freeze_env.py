#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook_output" / "requirements-freeze.txt"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        OUT.write_text(res.stdout)
        print(f"Wrote freeze → {OUT}")
    except subprocess.CalledProcessError as e:
        OUT.write_text(f"freeze_failed: {e}\n{e.stdout}\n{e.stderr}\n")
        print("pip freeze failed; wrote diagnostic", file=sys.stderr)


if __name__ == "__main__":
    main()

