#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TPL = (
    "# Example environment\n"
    "# Copy to .env and fill in locally. Never commit real secrets.\n\n"
    "# Provider\nPOE_API_KEY=your_poe_key_here\nOPENAI_API_KEY=your_openai_key_here\nOPENAI_BASE_URL=https://api.poe.com/v1\n\n"
    "# Local models\n# OPENAI_BASE_URL=http://localhost:1234/v1\n\n"
    "# Optional\nHF_TOKEN=\nDATABASE_URL=\n"
)

def main() -> None:
    out = ROOT / ".env.example"
    if out.exists():
        print(f"Exists: {out}")
        return
    out.write_text(TPL)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

