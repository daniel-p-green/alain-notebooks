from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat

from alain_kit_lite_generate import level_steps, make_lesson

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "json_to_notebook", REPO_ROOT / "scripts" / "json_to_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_beginner_level_steps_include_complete_code_block():
    steps = level_steps("beginner", "Test prompt", "gpt-oss-20b")

    assert all("code" in step for step in steps), "Each beginner step should provide executable code."

    setup_code = steps[0]["code"]
    assert "_ensure_package(\"openai\")" in setup_code
    assert "RuntimeError(\"Set POE_API_KEY" in setup_code
    assert "pass" not in setup_code

    first_call_code = steps[1]["code"]
    assert "client.chat.completions.create" in first_call_code
    assert "Test prompt" in first_call_code


def test_build_notebook_uses_code_field(tmp_path):
    lesson = make_lesson(
        provider="poe",
        chat_model="gpt-oss-20b",
        hf_model="openai/gpt-oss-20b",
        brief="Test prompt",
        level="beginner",
    )

    builder = load_builder_module()
    notebook = builder.build_notebook(lesson)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 3

    first_cell = code_cells[0].source
    assert "_ensure_package(\"openai\")" in first_cell
    assert "pass" not in first_cell
    assert "Test prompt" in code_cells[1].source

    nb_path = tmp_path / "notebook.ipynb"
    nbformat.write(notebook, nb_path)
    assert nb_path.exists()
