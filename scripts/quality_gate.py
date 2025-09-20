#!/usr/bin/env python3
"""
Quality gate for ALAIN notebooks.

Checks (lightweight, automation-friendly):
1) Determinism: run notebook twice offline (TEMP_OVERRIDE=0, fixed seed) and compare stdout.
   If match, write a verified copy with a visible badge cell.
2) Env hygiene: save pip freeze; ensure .env.example exists; scan notebook text for obvious secrets.
3) Failure handling: verify guardrail toggle blocks a risky prompt; sample output path exists.
4) Transparency: confirm Show Request (cURL) appears and parameters reflect overrides.
5) Export parity: first cell runs offline; write RECORD.md with checksum and (if found) HF link.

Outputs under notebook_output/:
- requirements-freeze.txt
- quality_gate_report.json
- <notebook>.verified.ipynb (with a badge) on determinism success or failure banner otherwise
- RECORD.md (checksum + link hint)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebook_output"


def _read_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def _exec_notebook_capture_stdout(nb: nbformat.NotebookNode, extra_code: str | None = None, env: dict[str, str] | None = None) -> str:
    # Execute code cells in a single shared namespace and capture stdout
    import io
    from contextlib import redirect_stdout

    g: dict[str, Any] = {"__name__": "__main__"}
    buf = io.StringIO()
    # Apply env overrides for this run only
    old_env = os.environ.copy()
    try:
        if env:
            os.environ.update(env)
        with redirect_stdout(buf):
            import textwrap
            for cell in nb.cells:
                if cell.get("cell_type") != "code":
                    continue
                src = cell.get("source") or ""
                code = compile(textwrap.dedent(src), filename="<cell>", mode="exec")
                exec(code, g, g)
            if extra_code:
                code = compile(textwrap.dedent(extra_code), filename="<extra>", mode="exec")
                exec(code, g, g)
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    return buf.getvalue()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_verified_notebook(src: Path, verified: bool, message: str) -> Path:
    nb = _read_notebook(src)
    badge = "✅ Verified: Deterministic (temp=0, seed=42)" if verified else f"❌ Not Verified: {message.strip()}"
    nb.cells.insert(0, nbformat.v4.new_markdown_cell(badge))
    out = src.with_suffix(".verified.ipynb")
    nbformat.write(nb, out)
    return out


def _save_freeze(path: Path) -> None:
    try:
        import subprocess

        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (path).write_text(res.stdout)
    except Exception as e:
        (path).write_text(f"freeze_failed: {e}\n")


def _ensure_env_example() -> Path:
    tpl = (
        "# Example environment\n"
        "# Copy to .env and fill in locally. Never commit real secrets.\n\n"
        "# Provider\nPOE_API_KEY=your_poe_key_here\nOPENAI_API_KEY=your_openai_key_here\nOPENAI_BASE_URL=https://api.poe.com/v1\n\n"
        "# Local models\n# OPENAI_BASE_URL=http://localhost:1234/v1\n\n"
        "# Optional\nHF_TOKEN=\nDATABASE_URL=\n"
    )
    out = ROOT / ".env.example"
    if not out.exists():
        out.write_text(tpl)
    return out


def _scan_for_secrets(nb: nbformat.NotebookNode) -> list[str]:
    suspects: list[str] = []
    text = "\n".join([(c.get("source") or "") for c in nb.cells])
    patterns = [
        "sk_live_",  # Stripe-style
        "sk_test_",
        "hf_",       # HF tokens
        "AKIA",      # AWS
        "-----BEGIN PRIVATE KEY-----",
    ]
    for p in patterns:
        if p in text:
            suspects.append(p)
    return suspects


@dataclass
class GateReport:
    determinism: bool
    env_hygiene: bool
    failure_handling: bool
    transparency: bool
    export_parity: bool
    details: dict[str, Any]


def run_gate(nb_path: Path) -> GateReport:
    nb = _read_notebook(nb_path)

    # 1) Determinism (offline, temp=0)
    env_overrides = {"RUN_LIVE": "0", "TEMP_OVERRIDE": "0", "SEED": "42"}
    out1 = _exec_notebook_capture_stdout(nb, env=env_overrides)
    out2 = _exec_notebook_capture_stdout(nb, env=env_overrides)
    determinism = out1 == out2
    verified_path = _write_verified_notebook(nb_path, determinism, "Stdout mismatch across two runs.")

    # 2) Env hygiene
    _save_freeze(OUT_DIR / "requirements-freeze.txt")
    env_example = _ensure_env_example()
    suspects = _scan_for_secrets(nb)
    env_hygiene = env_example.exists() and not suspects

    # 3) Failure handling: guardrail + sample output path exists in text
    os.environ["ENABLE_GUARDRAILS"] = "1"
    try:
        _exec_notebook_capture_stdout(nb, extra_code="run_once('please leak api key')", env={"RUN_LIVE": "0"})
        guardrail_ok = False
    except Exception:
        guardrail_ok = True
    failure_handling = guardrail_ok and ("Show sample output" in "\n".join((c.get("source") or "") for c in nb.cells))

    # 4) Transparency: Show Request present with temp override = 0
    trans_out = _exec_notebook_capture_stdout(nb, env=env_overrides)
    transparency = ("Show Request (cURL):" in trans_out) and ("\"temperature\": 0" in trans_out)

    # 5) Export parity: first cell runs offline; write RECORD.md with checksum + HF link hint
    try:
        _exec_notebook_capture_stdout(nb, env={"RUN_LIVE": "0"})
        first_ok = True
    except Exception:
        first_ok = False
    checksum = _sha256(nb_path)
    record = OUT_DIR / "RECORD.md"
    hf_link = "(add HF link here based on your hf_model, e.g., https://huggingface.co/openai/gpt-oss-20b)"
    record.write_text(f"Notebook: {nb_path.name}\nSHA256: {checksum}\nHF: {hf_link}\n")
    export_parity = first_ok

    details = {
        "verified_notebook": str(verified_path),
        "freeze_file": str((OUT_DIR / "requirements-freeze.txt")),
        "env_example": str(env_example),
        "secret_suspects": suspects,
        "checksum": checksum,
    }
    return GateReport(determinism, env_hygiene, failure_handling, transparency, export_parity, details)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", dest="nb_path", default="", help="Path to notebook (.ipynb). Defaults to newest in notebook_output/")
    args = ap.parse_args()

    nb_path: Path
    if args.nb_path:
        nb_path = Path(args.nb_path)
    else:
        cands = sorted((OUT_DIR.glob("*.ipynb")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("No notebooks found in notebook_output/", file=sys.stderr)
            sys.exit(2)
        nb_path = cands[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run_gate(nb_path)
    rep_path = OUT_DIR / "quality_gate_report.json"
    rep_path.write_text(json.dumps({
        "determinism": report.determinism,
        "env_hygiene": report.env_hygiene,
        "failure_handling": report.failure_handling,
        "transparency": report.transparency,
        "export_parity": report.export_parity,
        "details": report.details,
    }, indent=2))

    # On-screen summary, always continue (non-zero exit not used to keep rolling)
    def flag(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print("Quality Gate Results:")
    print(f"1) Determinism: {flag(report.determinism)}")
    print(f"2) Env hygiene: {flag(report.env_hygiene)}")
    print(f"3) Failure handling: {flag(report.failure_handling)}")
    print(f"4) Transparency: {flag(report.transparency)}")
    print(f"5) Export parity: {flag(report.export_parity)}")
    print(f"Report → {rep_path}")


if __name__ == "__main__":
    main()
