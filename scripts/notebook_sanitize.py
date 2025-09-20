#!/usr/bin/env python3
"""
Notebook sanitizer for ALAIN-Kit polyglot content.

Goals
- Keep notebooks executable under a Python kernel by converting non-Python
  snippets (TS/TSX/JS/JSON/spec blocks) into Markdown fenced code blocks.
- Normalize common smart punctuation to ASCII in code cells.
- Optionally pin language_info.version if missing.

Usage
  python scripts/notebook_sanitize.py path/to/notebook.ipynb [--write]

With --write, modifies the notebook in place and creates a .bak backup.
Without --write, prints a summary only.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
from typing import Tuple

import nbformat


SMART_MAP = {
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote/apostrophe
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",  # ellipsis
    "\u00A0": " ",  # NBSP
    "\u200B": "",  # zero-width space
    "\u2022": " ",  # bullet
    "\u2192": "->",  # right arrow
    "\u00B7": " ",  # middle dot
    "\u202F": " ",  # narrow NBSP
}


def _normalize_ascii(s: str) -> Tuple[str, int]:
    changed = 0
    for k, v in SMART_MAP.items():
        if k in s:
            s = s.replace(k, v)
            changed += 1
    return s, changed


def _guess_lang(src: str) -> str:
    # Heuristics for TS/TSX/JS/JSON
    first_line = next((ln for ln in src.splitlines() if ln.strip()), "")
    m = re.match(r"\s*//\s*.*\.(tsx|ts|js|jsx)\b", first_line)
    if m:
        return {"tsx": "tsx", "ts": "ts", "js": "js", "jsx": "jsx"}[m.group(1)]
    if first_line.strip().startswith("{") and ":" in src and '"' in src:
        return "json"
    if re.search(r"\bexport\b|\bimport\b.*from\b|as const|<\w+.*>", src):
        # crude TS/TSX detection
        return "ts"
    return "text"


def sanitize_notebook(path: Path, write: bool) -> dict:
    nb = nbformat.read(path, as_version=4)
    changed_cells = 0
    converted = 0
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source") or "")
        # ASCII normalization
        norm, delta = _normalize_ascii(src)
        if delta:
            c["source"] = norm
            src = norm
            changed_cells += 1
        # Skip classification if Python parses
        try:
            ast.parse(src)
            continue
        except SyntaxError:
            pass
        # Convert to Markdown fenced code
        lang = _guess_lang(src)
        fenced = f"```{lang}\n{src}\n```\n"
        c["cell_type"] = "markdown"
        c["source"] = fenced
        c.pop("outputs", None)
        c.pop("execution_count", None)
        converted += 1

    # Ensure language_info has a version (non-fatal)
    meta = nb.get("metadata", {}).get("language_info", {})
    if isinstance(meta, dict) and not meta.get("version"):
        nb["metadata"].setdefault("language_info", {})["version"] = "3.11"

    summary = {
        "path": str(path),
        "ascii_normalized_cells": changed_cells,
        "converted_non_python_to_markdown": converted,
    }

    if write:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        nbformat.write(nb, path)
        summary["backup"] = str(backup)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", help="Path to .ipynb")
    ap.add_argument("--write", action="store_true", help="Modify in place and create .bak backup")
    args = ap.parse_args()

    summary = sanitize_notebook(Path(args.notebook), write=args.write)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

