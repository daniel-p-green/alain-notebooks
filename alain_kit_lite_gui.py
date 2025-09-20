#!/usr/bin/env python3
"""Simple Tkinter GUI for the ALAIN-Kit Lite generator.

Design choices follow publicly documented OpenAI brand guidance:
- Minimal white layouts with black typography for clarity, aligning with the
  official color palette (primary HEX #000000) [1].
- Plenty of whitespace and restrained accents without altered logos per the
  OpenAI brand usage guidelines [2].

[1] https://brandpalettes.com/openai-color-codes/
[2] https://openai.com/brand
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from tkinter import scrolledtext

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for env loading
    load_dotenv = None

from alain_kit_lite_generate import GenerationResult, generate_notebook

PRIMARY_BG = "#F7F7F6"  # alain-card
WINDOW_BG = "#FFFFFF"   # alain-bg
PRIMARY_FG = "#111827"  # alain-text
ACCENT_BG = "#0058A3"   # alain-blue
ACCENT_BG_HOVER = "#004580"  # alain-stroke
SECONDARY_ACCENT = "#1E3A8A"  # alain-navy for stronger contrast
FIELD_BG = "#FFFFFF"
FIELD_BORDER = "#004580"

LOADED_ENV_PATHS: list[str] = []


def _manual_load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        os.environ.setdefault(key, value)


def _load_env() -> None:
    """Load .env/.env.local near the script to surface provider keys."""

    if not load_dotenv:
        loader = None
    else:
        loader = load_dotenv

    script_root = Path(__file__).resolve().parent
    candidates = {script_root, script_root.parent, Path.cwd()}
    for root in candidates:
        for name in (".env", ".env.local"):
            env_path = root / name
            if env_path.exists():
                env_str = str(env_path)
                if env_str in LOADED_ENV_PATHS:
                    continue
                loaded = False
                if loader:
                    loaded = loader(env_path, override=False)
                if not loaded:
                    _manual_load_env_file(env_path)
                LOADED_ENV_PATHS.append(env_str)


class ALAINKitLiteGUI(tk.Tk):
    """Window that wraps the CLI generator in a form-centric workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ALAIN-Kit Lite Builder")
        self.configure(bg=WINDOW_BG)
        self.geometry("720x640")
        self.minsize(640, 560)

        _load_env()
        self._env_paths = LOADED_ENV_PATHS.copy()

        self._init_fonts()
        self._init_state()
        self._build_layout()

    def _init_fonts(self) -> None:
        self.header_font = tkfont.Font(family="Montserrat", size=20, weight="bold")
        self.body_font = tkfont.Font(family="Inter", size=12)
        self.small_font = tkfont.Font(family="Inter", size=10)

    def _init_state(self) -> None:
        self.hf_model_var = tk.StringVar(value="openai/gpt-oss-20b")
        self.level_var = tk.StringVar(value="beginner")
        self.provider_var = tk.StringVar(value="poe")
        default_base = os.getenv("OPENAI_BASE_URL", "https://api.poe.com/v1")
        self.base_url_var = tk.StringVar(value=default_base)
        self.chat_model_var = tk.StringVar()
        default_out_dir = Path("notebook_output").resolve()
        self.out_dir_var = tk.StringVar(value=str(default_out_dir))

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        container = tk.Frame(self, bg=PRIMARY_BG, padx=32, pady=32, bd=1, relief="solid", highlightbackground=FIELD_BORDER, highlightthickness=1)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(1, weight=1)

        header_frame = tk.Frame(container, bg=ACCENT_BG)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header_frame.grid_columnconfigure(0, weight=1)
        header = tk.Label(
            header_frame,
            text="ALAIN-Kit Lite Notebook Builder",
            font=self.header_font,
            bg=ACCENT_BG,
            fg="#FFFFFF",
            padx=12,
            pady=8,
        )
        header.grid(row=0, column=0, sticky="w")

        subheader = tk.Label(
            container,
            text=(
                "Generate lightweight lesson JSON + notebooks from templates."
                " Configure inputs below and click Generate."
            ),
            font=self.body_font,
            bg=PRIMARY_BG,
            fg=PRIMARY_FG,
            wraplength=560,
            justify="left",
        )
        subheader.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 24))

        self._add_label_and_entry(container, "Hugging Face model id", self.hf_model_var, row=2)

        brief_label = tk.Label(
            container,
            text="Brief (will seed the code templates)",
            font=self.body_font,
            bg=PRIMARY_BG,
            fg=PRIMARY_FG,
        )
        brief_label.grid(row=3, column=0, sticky="w")
        self.brief_text = tk.Text(
            container,
            height=4,
            font=self.body_font,
            bg=FIELD_BG,
            fg=PRIMARY_FG,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=FIELD_BORDER,
            wrap="word",
        )
        self.brief_text.insert("1.0", "Show me a minimal streaming chat example with tips.")
        self.brief_text.grid(row=3, column=1, sticky="ew", pady=(0, 16))

        self._add_dropdown(
            container,
            label="Skill level",
            variable=self.level_var,
            values=["beginner", "intermediate", "advanced"],
            row=4,
        )
        self._add_dropdown(
            container,
            label="Provider",
            variable=self.provider_var,
            values=["poe", "openai-compatible", "local"],
            row=5,
        )

        self._add_label_and_entry(container, "Base URL (optional)", self.base_url_var, row=6)
        self._add_label_and_entry(container, "Chat model override (optional)", self.chat_model_var, row=7)

        out_dir_label = tk.Label(
            container,
            text="Output directory",
            font=self.body_font,
            bg=PRIMARY_BG,
            fg=PRIMARY_FG,
        )
        out_dir_label.grid(row=8, column=0, sticky="w")
        out_dir_frame = tk.Frame(container, bg=PRIMARY_BG)
        out_dir_frame.grid(row=8, column=1, sticky="ew", pady=(0, 16))
        out_dir_frame.grid_columnconfigure(0, weight=1)
        out_dir_entry = tk.Entry(
            out_dir_frame,
            textvariable=self.out_dir_var,
            font=self.body_font,
            bg=FIELD_BG,
            fg=PRIMARY_FG,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=FIELD_BORDER,
            highlightcolor=FIELD_BORDER,
        )
        out_dir_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        browse_btn = tk.Button(
            out_dir_frame,
            text="Browse",
            command=self._choose_output_dir,
            font=self.body_font,
            bg=SECONDARY_ACCENT,
            fg="#FFFFFF",
            activebackground=ACCENT_BG,
            activeforeground="#FFFFFF",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
        )
        browse_btn.grid(row=0, column=1)

        self.generate_btn = tk.Button(
            container,
            text="Generate notebook",
            command=self._on_generate_clicked,
            font=self.body_font,
            bg=ACCENT_BG,
            fg="#FFFFFF",
            activebackground=ACCENT_BG_HOVER,
            activeforeground="#FFFFFF",
            bd=0,
            padx=16,
            pady=10,
            cursor="hand2",
        )
        self.generate_btn.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 20))

        status_label = tk.Label(
            container,
            text="Status",
            font=self.body_font,
            bg=PRIMARY_BG,
            fg=ACCENT_BG,
        )
        status_label.grid(row=10, column=0, sticky="w")
        log_frame = tk.Frame(container, bg=PRIMARY_BG, highlightbackground=FIELD_BORDER, highlightthickness=1)
        log_frame.grid(row=10, column=1, sticky="nsew")
        self.log = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            font=self.small_font,
            bg=FIELD_BG,
            fg=PRIMARY_FG,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=FIELD_BORDER,
            state="disabled",
            wrap="word",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        container.rowconfigure(10, weight=1)

    def _add_label_and_entry(self, parent: tk.Misc, label: str, variable: tk.StringVar, row: int) -> None:
        lbl = tk.Label(parent, text=label, font=self.body_font, bg=PRIMARY_BG, fg=PRIMARY_FG)
        lbl.grid(row=row, column=0, sticky="w")
        entry = tk.Entry(
            parent,
            textvariable=variable,
            font=self.body_font,
            bg=FIELD_BG,
            fg=PRIMARY_FG,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=FIELD_BORDER,
            highlightcolor=FIELD_BORDER,
        )
        entry.grid(row=row, column=1, sticky="ew", pady=(0, 16))

    def _add_dropdown(
        self,
        parent: tk.Misc,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        row: int,
    ) -> None:
        lbl = tk.Label(parent, text=label, font=self.body_font, bg=PRIMARY_BG, fg=PRIMARY_FG)
        lbl.grid(row=row, column=0, sticky="w")
        option = tk.OptionMenu(parent, variable, *values)
        option.configure(
            font=self.body_font,
            bg=FIELD_BG,
            fg=PRIMARY_FG,
            highlightthickness=1,
            highlightbackground=FIELD_BORDER,
            activebackground=FIELD_BG,
        )
        menu = option.nametowidget(option.menuname)
        menu.configure(font=self.body_font, bg=FIELD_BG, fg=PRIMARY_FG)
        option.grid(row=row, column=1, sticky="ew", pady=(0, 16))

    def _choose_output_dir(self) -> None:
        current = Path(self.out_dir_var.get()).expanduser()
        initial = current if current.exists() else Path.cwd()
        selected = filedialog.askdirectory(initialdir=initial)
        if selected:
            self.out_dir_var.set(selected)

    def _on_generate_clicked(self) -> None:
        brief = self.brief_text.get("1.0", "end").strip()
        if not brief:
            messagebox.showerror("Missing brief", "Please provide a short brief to seed the notebook.")
            return

        params = dict(
            hf_model=self.hf_model_var.get().strip(),
            brief=brief,
            level=self.level_var.get(),
            provider=self.provider_var.get(),
            base_url=self.base_url_var.get().strip(),
            chat_model=self.chat_model_var.get().strip() or None,
            out_dir=self.out_dir_var.get().strip(),
        )

        if not params["hf_model"]:
            messagebox.showerror("Missing model", "Please supply a Hugging Face model id.")
            return

        self._append_log("Starting generation...", reset=True)
        self._set_generate_enabled(False)
        worker = threading.Thread(target=self._run_generation, args=(params,), daemon=True)
        worker.start()

    def _run_generation(self, params: dict[str, str | None]) -> None:
        try:
            result = generate_notebook(**params)
        except Exception as exc:  # pylint: disable=broad-except
            self.after(0, self._handle_failure, exc)
            return

        self.after(0, self._handle_success, result, params)

    def _handle_success(self, result: GenerationResult, params: dict[str, str | None]) -> None:
        self._append_log(
            "\n".join(
                [
                    "✓ Generation complete",
                    f"Lesson JSON: {result.json_path}",
                    f"Notebook: {result.notebook_path}",
                ]
            )
        )
        if self._env_paths:
            self._append_log("Loaded env from: " + ", ".join(self._env_paths))
        provider = (params.get("provider") or "").lower()
        if provider == "poe":
            if os.getenv("POE_API_KEY"):
                masked = os.getenv("POE_API_KEY")
                if masked and len(masked) > 8:
                    masked = masked[:4] + "…" + masked[-4:]
                self._append_log(f"Detected POE_API_KEY in environment ({masked}).")
            else:
                self._append_log("POE_API_KEY missing. Set it in .env or your shell; the notebook maps it to OPENAI_API_KEY.")
        elif provider in {"openai-compatible", "local"}:
            if os.getenv("OPENAI_API_KEY"):
                self._append_log("Detected OPENAI_API_KEY; ensure OPENAI_BASE_URL points to your gateway.")
            else:
                self._append_log("Set OPENAI_API_KEY and OPENAI_BASE_URL for your gateway before running the notebook.")
        base_url = params.get("base_url") or ""
        if base_url and base_url != "https://api.poe.com/v1":
            self._append_log(f"Remember to set OPENAI_BASE_URL={base_url}")
        self._set_generate_enabled(True)

    def _handle_failure(self, exc: Exception) -> None:
        self._append_log(f"Generation failed: {exc}")
        messagebox.showerror("Generation failed", str(exc))
        self._set_generate_enabled(True)

    def _append_log(self, message: str, *, reset: bool = False) -> None:
        self.log.configure(state="normal")
        if reset:
            self.log.delete("1.0", "end")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_generate_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.generate_btn.configure(state=state)


def main() -> None:
    app = ALAINKitLiteGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
