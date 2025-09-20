from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

import nbformat

from alain_kit_lite_generate import generate_notebook


def execute_notebook_offline(nb_path) -> SimpleNamespace:
    nb = nbformat.read(nb_path, as_version=4)
    # Shared namespace to simulate a Jupyter kernel state across cells.
    g = {
        "__name__": "__main__",
    }
    buf = io.StringIO()
    import textwrap
    with redirect_stdout(buf):
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            src = textwrap.dedent(cell.get("source") or "").lstrip("\n")
            try:
                code = compile(src, filename=str(nb_path), mode="exec")
                exec(code, g, g)
            except Exception as exc:  # pragma: no cover - report which cell failed
                raise AssertionError(f"Notebook execution failed: {exc}\nCell:\n{src}")
    return SimpleNamespace(stdout=buf.getvalue())


def test_generate_and_execute_notebook_smoke(tmp_path, monkeypatch):
    # Ensure offline path: disable live calls
    monkeypatch.setenv("RUN_LIVE", "0")
    # Also clear keys to ensure gating logic is respected
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POE_API_KEY", raising=False)

    res = generate_notebook(
        hf_model="openai/gpt-oss-20b",
        brief="Test prompt for smoke",
        level="beginner",
        provider="poe",
        out_dir=tmp_path,
        sanitize=False,
    )

    assert res.json_path.exists()
    assert res.notebook_path.exists()

    out = execute_notebook_offline(res.notebook_path)
    assert "Base URL" in out.stdout
    assert "RUN_LIVE=False" in out.stdout
    assert "[SMOKE] Echo" in out.stdout
