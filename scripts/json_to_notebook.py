#!/usr/bin/env python3
"""
Convert a compact ALAIN‑Kit Lite lesson JSON into a runnable Jupyter notebook (.ipynb).

Expectations
- The lesson dict contains: title, description, provider, model, learning_objectives, steps[].
- Each step has title, content, and either a concrete "code" string (preferred) or a
  "code_template" string that will be included as commented guidance.

CLI
  python scripts/json_to_notebook.py --in path/to/lesson.json --out path/to/notebook.ipynb

The notebook is built to execute top‑to‑bottom under a Python kernel. Non‑Python blocks
are rendered as Markdown by upstream sanitizers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import textwrap


def _mk_title_cells(lesson: dict[str, Any]) -> list:
    title = lesson.get("title", "ALAIN‑Kit Lite Notebook")
    desc = lesson.get("description", "")
    provider = lesson.get("provider", "")
    model = lesson.get("model", "")
    base_url = lesson.get("base_url", "")

    lines = [f"# {title}", "", desc]
    if provider or model or base_url:
        details = [
            f"- Provider: {provider}" if provider else None,
            f"- Model: {model}" if model else None,
            f"- Base URL: {base_url}" if base_url else None,
        ]
        lines += ["", "Details:"] + [d for d in details if d]

    return [new_markdown_cell("\n".join(lines))]


def _mk_objectives_cell(objectives: list[str]) -> list:
    if not objectives:
        return []
    bullets = "\n".join(f"- {o}" for o in objectives)
    return [new_markdown_cell(f"## Learning Objectives\n\n{bullets}")]


def _mk_step_cells(steps: list[dict]) -> list:
    cells = []
    for i, step in enumerate(steps, start=1):
        title = step.get("title", f"Step {i}")
        content = step.get("content", "")
        cells.append(new_markdown_cell(f"## Step {i}: {title}\n\n{content}"))

        if "code" in step and isinstance(step["code"], str) and step["code"].strip():
            code = textwrap.dedent(step["code"]).lstrip("\n")
            cells.append(new_code_cell(code))
        else:
            # Fall back to a safe scaffold that will always execute.
            template = (step.get("code_template") or "").strip()
            scaffold = f"""
# Guidance (template):\n# {template or 'No template provided.'}\n\nprint('Template ready. Set RUN_LIVE=True and add client calls above to execute against your provider.')
""".strip()
            cells.append(new_code_cell(scaffold))

    return cells


def build_notebook(lesson: dict[str, Any]) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb["cells"] = []

    # Title/description
    nb["cells"] += _mk_title_cells(lesson)

    # Objectives
    nb["cells"] += _mk_objectives_cell(lesson.get("learning_objectives", []))

    # Steps
    nb["cells"] += _mk_step_cells(lesson.get("steps", []))

    # Footer note
    nb["cells"].append(
        new_markdown_cell(
            "Note: Live API calls require OPENAI_API_KEY and OPENAI_BASE_URL to be set. "
            "The setup cell helps map keys for Poe or gateways."
        )
    )

    # Minimal metadata for Python 3.11 kernel
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.11",
    }

    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to lesson JSON")
    ap.add_argument("--out", dest="out_path", required=True, help="Path to write .ipynb")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    lesson = json.loads(in_path.read_text(encoding="utf-8"))

    nb = build_notebook(lesson)
    nbformat.write(nb, out_path)


if __name__ == "__main__":
    main()
