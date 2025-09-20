#!/usr/bin/env python3
"""
ALAIN‑Kit Lite generator

Create a compact lesson JSON and render a runnable .ipynb using the
existing scripts/json_to_notebook.py. Keeps token usage low by using
templated sections tailored by level and provider.

Usage examples:

  python alain_kit_lite_generate.py \
    --hf-model openai/gpt-oss-20b \
    --brief "Minimal streaming chat example" \
    --level beginner \
    --provider poe

  python alain_kit_lite_generate.py \
    --hf-model meta-llama/Llama-3.1-8B-Instruct \
    --brief "Compare temperature 0.2 vs 0.9" \
    --level advanced \
    --provider local \
    --base-url http://localhost:1234/v1 \
    --chat-model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import inspect
import textwrap
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "notebook_output"
BUILDER = ROOT / "scripts" / "json_to_notebook.py"
SANITIZER = ROOT / "scripts" / "notebook_sanitize.py"


@dataclass(frozen=True)
class GenerationResult:
    """Paths and metadata returned after a successful notebook build."""

    json_path: Path
    notebook_path: Path
    lesson: dict[str, Any]


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower()).strip("-")[:60]


def level_steps(level: str, brief: str, chat_model: str) -> list[dict]:
    # Short, token‑light content and templates per level
    if level == "beginner":
        safe_brief = repr(brief)
        base_setup = textwrap.dedent(inspect.cleandoc(
            """
            # Step 1: Environment & Client Setup
            # Offline-friendly: gate live API calls behind RUN_LIVE.

            from __future__ import annotations

            import os
            import random
            import subprocess
            import sys
            from importlib import import_module
            from pathlib import Path


            RUN_LIVE = os.getenv("RUN_LIVE", "0").strip() in ("1", "true", "True")
            # Determinism knobs (used by quality gate and examples)
            SEED = int(os.getenv("SEED", "42"))
            random.seed(SEED)
            try:
                import numpy as _np  # type: ignore
            except Exception as _e:  # pragma: no cover
                _np = None
            if _np is not None:
                _np.random.seed(SEED)

            def _ensure_package(module: str, pip_name: str | None = None) -> None:
                # Install a module on demand to keep the notebook self-contained.
                try:
                    import_module(module)
                except Exception:  # pragma: no cover
                    if not RUN_LIVE:
                        return
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or module])


            _ensure_package("dotenv", "python-dotenv")

            try:
                from dotenv import load_dotenv
            except ImportError:
                load_dotenv = None


            def _ingest_env_file(path: Path) -> None:
                # Load key=value pairs without overwriting existing values.
                if load_dotenv:
                    load_dotenv(path, override=False)
                    return
                if not path.exists():
                    return
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


            for candidate in (Path(".env"), Path(".env.local")):
                _ingest_env_file(candidate)

            if os.getenv("POE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = os.environ["POE_API_KEY"]

            BASE_URL = os.environ.setdefault("OPENAI_BASE_URL", "https://api.poe.com/v1")
            API_KEY = os.getenv("OPENAI_API_KEY")

            MODEL_ID = "{chat_model}"

            if RUN_LIVE:
                _ensure_package("openai")
                from openai import OpenAI
                client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
                if not API_KEY:
                    raise RuntimeError("Set POE_API_KEY or OPENAI_API_KEY before continuing.")
            else:
                # Deterministic stub client for smoke tests/offline execution.
                class _Msg:
                    def __init__(self, content: str):
                        self.content = content

                class _Choice:
                    def __init__(self, content: str):
                        self.message = _Msg(content)

                class _Reply:
                    def __init__(self, content: str):
                        self.choices = [_Choice(content)]
                        class _U:
                            'usage container'
                        self.usage = _U()
                        self.usage.prompt_tokens = 0
                        self.usage.completion_tokens = 0

                class _DummyCompletions:
                    def create(self, *, model: str, messages, **kwargs):  # type: ignore[no-untyped-def]
                        if isinstance(messages, (list, tuple)):
                            prompt = next(
                                (m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"),
                                "",
                            )
                        else:
                            prompt = ""
                        return _Reply("[SMOKE] Echo from " + str(model) + " → " + str(prompt))

                class _DummyChat:
                    def __init__(self):
                        self.completions = _DummyCompletions()

                class _DummyClient:
                    def __init__(self, *a, **k):
                        self.chat = _DummyChat()

                client = _DummyClient()

            print(f"Base URL → {{BASE_URL}}")
            print(f"Client ready for model → {{MODEL_ID}} (RUN_LIVE={{RUN_LIVE}})")
            print(f"Seed → {{SEED}}")
            """
        ).format(chat_model=chat_model))

        first_call = textwrap.dedent(inspect.cleandoc(
            """
            # Step 2: First Chat Completion
            # Draft a short helper that sends the user brief to the model once.

            def _guardrail_block(prompt: str) -> None:
                # Optional guardrail: block risky prompts when enabled (ENABLE_GUARDRAILS=1).
                enabled = os.getenv("ENABLE_GUARDRAILS", "0").strip() in ("1", "true", "True")
                if not enabled:
                    return
                text = (prompt or "").lower()
                risky = ("rm -rf" in text) or ("api key" in text) or ("password" in text)
                if risky:
                    raise ValueError("Blocked by guardrails: prompt contains risky terms.")

            def _show_request(params: dict) -> None:
                # Print a cURL-like view of the request for transparency.
                try:
                    import json as _json
                    payload = {{
                        "model": params["model"],
                        "messages": params["messages"],
                        "max_tokens": params.get("max_tokens", 256),
                        "temperature": params.get("temperature", 0),
                    }}
                    curl = (
                        "curl -sS -X POST "
                        + os.getenv("OPENAI_BASE_URL", "https://api.poe.com/v1") + "/chat/completions "
                        + "-H 'Content-Type: application/json' "
                        + "-H 'Authorization: Bearer $OPENAI_API_KEY' "
                        + "-d '" + _json.dumps(payload).replace("'", "\\'") + "'"
                    )
                    print("Show Request (cURL):")
                    print(curl)
                except Exception:
                    pass

            def _cost_hint(model: str) -> str:
                # Minimal, static hints where known; otherwise N/A.
                table = {{}}
                return table.get(model, "N/A")

            def run_once(prompt: str) -> str:
                # Call the Poe-hosted model with safe defaults and return the text.
                _guardrail_block(prompt)
                temperature = float(os.getenv("TEMP_OVERRIDE", os.getenv("TEMPERATURE", "0.6")))
                params = dict(
                    model=MODEL_ID,
                    messages=[
                        {{"role": "system", "content": "You are a patient teacher."}},
                        {{"role": "user", "content": prompt}},
                    ],
                    max_tokens=320,
                    temperature=temperature,
                )
                try:
                    response = client.chat.completions.create(**params)
                except Exception as e:
                    print(f"Live request failed: {{e}}")
                    print("Show sample output:")
                    return "[SAMPLE OUTPUT] This is a representative reply shown when the provider is unavailable."
                _show_request(params)
                choice = response.choices[0].message.content
                print(f"Prompt tokens → {{getattr(response.usage, 'prompt_tokens', 'N/A')}}, completion → {{getattr(response.usage, 'completion_tokens', 'N/A')}}")
                hint = _cost_hint(MODEL_ID)
                print(f"Cost hint → {{hint}}")
                return choice


            first_reply = run_once({safe_brief})
            print("\\nModel reply:\\n")
            print(first_reply)
            """
        ).format(safe_brief=safe_brief))

        parameter_tune = textwrap.dedent(inspect.cleandoc(
            """
            # Step 3: Adjust Parameters
            # Compare two temperatures to see how tone and verbosity change.

            def compare_temperatures(prompt: str, temperatures: tuple[float, float]) -> None:
                for temp in temperatures:
                    print(f"\\n--- Temperature {{temp}} ---")
                    reply = client.chat.completions.create(
                        model=MODEL_ID,
                        messages=[
                            {{"role": "system", "content": "You teach with concise, friendly explanations."}},
                            {{"role": "user", "content": prompt}},
                        ],
                        max_tokens=280,
                        temperature=temp,
                    )
                    print(reply.choices[0].message.content.strip())


            compare_temperatures({safe_brief}, (0.2, 0.8))
            """
        ).format(safe_brief=safe_brief))

        return [
            {
                "title": "Environment & Client Setup",
                "content": (
                    "Configure an OpenAI-compatible client pointed at Poe and confirm the "
                    "model you plan to use."
                ),
                "code": base_setup,
            },
            {
                "title": "First Chat Completion",
                "content": "Send your brief to the model with safe defaults and review the reply.",
                "code": first_call,
            },
            {
                "title": "Adjust Parameters",
                "content": "Contrast two temperatures to see how the narrative changes.",
                "model_params": {"temperature": 0.6},
                "code": parameter_tune,
            },
        ]
    if level == "intermediate":
        return [
            {
                "title": "Streaming Basics",
                "content": (
                    "Use streaming to improve perceived latency. Parse chunks safely."
                ),
                "code_template": "Explain streaming vs. non‑streaming and when to use each."
            },
            {
                "title": "Streaming Demo",
                "content": "Stream the model’s response for the brief.",
                "model_params": {"temperature": 0.5},
                "code_template": brief
            },
            {
                "title": "Telemetry",
                "content": "Capture latency and token usage from the response.",
                "model_params": {"temperature": 0.7},
                "code_template": "Summarize key points in 4 bullets and keep under 120 tokens."
            },
        ]
    # advanced
    return [
        {
            "title": "Provider Swap & Guardrails",
            "content": (
                "Note the base URL and key expectations for Poe vs. gateways."
            ),
            "code_template": "Print which provider is active and any detected model name."
        },
        {
            "title": "Pairwise Compare",
            "content": "Call the same prompt on two models and print side‑by‑side outputs.",
            "model_params": {"temperature": 0.3},
            "code_template": brief
        },
        {
            "title": "Mini Elo Update",
            "content": "Apply a single Elo update assuming model A won. Show new ratings.",
            "model_params": {"temperature": 0.7},
            "code_template": (
                "Given ratings rA=1500, rB=1500 and outcome A=win, compute new Elo with K=24."
            )
        },
    ]


def make_lesson(provider: str, chat_model: str, hf_model: str, brief: str, level: str) -> dict:
    title = f"Lite Notebook · {hf_model} · {level.capitalize()}"
    description = (
        "Token‑light tutorial: environment setup + runnable calls."
        " Uses OpenAI SDK against selected provider (Poe/OpenAI‑compatible/local)."
    )
    objectives = [
        "Configure provider and API key correctly",
        "Run a model call with safe defaults",
        "Tune basic parameters and/or streaming",
        "Record simple telemetry or ranking step"
    ]
    steps = []
    for i, s in enumerate(level_steps(level, brief, chat_model), start=1):
        step = {
            "step_order": i,
            "title": s["title"],
            "content": s.get("content", ""),
        }
        if "code" in s:
            step["code"] = s["code"]
        else:
            step["code_template"] = s.get("code_template", "")
        if "model_params" in s:
            step["model_params"] = s["model_params"]
        steps.append(step)

    return {
        "title": title,
        "description": description,
        "provider": provider,
        "model": chat_model,
        "learning_objectives": objectives,
        "steps": steps,
        "assessments": [
            {
                "question": "Which env var provides the Poe key?",
                "options": ["OPENAI_BASE_URL", "POE_API_KEY", "NEXT_RUNTIME", "HF_TOKEN"],
                "correct_index": 1,
                "explanation": "Poe auth uses POE_API_KEY; the code maps it to OPENAI_API_KEY at runtime."
            }
        ]
    }


def _resolve_chat_model(provider: str, chat_model: str, hf_model: str) -> str:
    if chat_model:
        return chat_model
    if provider == "poe":
        return "gpt-oss-20b"
    # For gateways/local, default to the HF id (works for LM Studio/vLLM if loaded)
    return hf_model


def generate_notebook(
    hf_model: str,
    brief: str,
    level: str = "beginner",
    provider: str = "poe",
    base_url: str = "",
    chat_model: str | None = None,
    out_dir: Path | str = OUT_DIR,
    *,
    builder: Path | None = None,
    timestamp: int | None = None,
    sanitize: bool = True,
) -> GenerationResult:
    """Create the lesson JSON and notebook using the existing builder script.

    Parameters mirror the CLI options so the generator can be reused programmatically.
    """

    if level not in {"beginner", "intermediate", "advanced"}:
        raise ValueError(f"Unsupported level: {level}")
    if provider not in {"poe", "openai-compatible", "local"}:
        raise ValueError(f"Unsupported provider: {provider}")

    effective_provider = provider if provider != "local" else "openai-compatible"
    resolved_chat_model = _resolve_chat_model(effective_provider, chat_model or "", hf_model)

    lesson = make_lesson(effective_provider, resolved_chat_model, hf_model, brief, level)
    # Surface base_url in the lesson JSON so the builder/notebook can reflect it.
    if base_url:
        lesson["base_url"] = base_url

    ts = timestamp if timestamp is not None else int(time.time())
    slug = slugify(f"{hf_model}-{level}")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"lite_{slug}_{ts}.json"
    ipynb_path = output_dir / f"lite_{slug}_{ts}.ipynb"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(lesson, f, indent=2)

    builder_script = builder or BUILDER
    cmd = [
        "python",
        str(builder_script),
        "--in",
        str(json_path),
        "--out",
        str(ipynb_path),
    ]
    subprocess.check_call(cmd, cwd=ROOT)

    if sanitize and SANITIZER.exists():
        subprocess.check_call(
            ["python", str(SANITIZER), str(ipynb_path), "--write"],
            cwd=ROOT,
        )

    return GenerationResult(json_path=json_path, notebook_path=ipynb_path, lesson=lesson)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True, help="Hugging Face model id (owner/model)")
    ap.add_argument("--brief", required=True, help="Short instruction to drive examples")
    ap.add_argument("--level", choices=["beginner", "intermediate", "advanced"], default="beginner")
    ap.add_argument("--provider", choices=["poe", "openai-compatible", "local"], default="poe")
    ap.add_argument("--base-url", default="", help="Override base URL for openai-compatible/local")
    ap.add_argument("--chat-model", default="", help="Override chat model id used in code")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for files")
    args = ap.parse_args()

    result = generate_notebook(
        hf_model=args.hf_model,
        brief=args.brief,
        level=args.level,
        provider=args.provider,
        base_url=args.base_url,
        chat_model=args.chat_model or None,
        out_dir=args.out_dir,
    )

    print("✓ ALAIN‑Kit Lite lesson JSON:", result.json_path)
    print("✓ Notebook:", result.notebook_path)
    print("Run it with your env configured. For Poe, set POE_API_KEY; the notebook maps it to OPENAI_API_KEY.")
    if args.base_url:
        print("Note: set OPENAI_BASE_URL=", args.base_url)


if __name__ == "__main__":
    main()
