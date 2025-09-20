ALAIN‑Kit Lite

Purpose
- Generate compact, high‑quality code notebooks with minimal tokens by assembling from templates instead of multi‑phase prompting.
- Inputs: Hugging Face model id, a brief prompt, a skill level (beginner|intermediate|advanced), and a provider (Poe, OpenAI‑compatible, or local LM Studio/Ollama/vLLM).

What it creates
- A small lesson JSON and a runnable .ipynb using the existing json_to_notebook.py builder.
- Setup/env cells that map correctly for the selected provider:
  - Poe → base URL https://api.poe.com/v1 and API key from POE_API_KEY (mapped to OPENAI_API_KEY)
  - OpenAI‑compatible → use provided --base-url or defaults (e.g., LM Studio http://localhost:1234/v1)
  - Local (Ollama/vLLM) → set OPENAI_BASE_URL to the local endpoint (e.g., http://localhost:11434/v1)

Quick start
1) Ensure Python deps (nbformat):
   - pip install nbformat

2) Create a .env.local (if you’ll run Poe or gateway providers):
   - POE_API_KEY=sk‑... (Poe)
   - or OPENAI_API_KEY=sk‑... + OPENAI_BASE_URL=https://your‑gateway/v1

3) Generate a beginner notebook (Poe):
   - python alain_kit_lite_generate.py \
       --hf-model openai/gpt-oss-20b \
       --brief "Show me a minimal streaming chat example with tips." \
       --level beginner \
       --provider poe

4) Generate an advanced local (LM Studio) notebook:
   - python alain_kit_lite_generate.py \
       --hf-model meta-llama/Llama-3.1-8B-Instruct \
       --brief "Compare temperature 0.2 vs 0.9 on the same prompt and report token usage." \
       --level advanced \
       --provider local \
       --base-url http://localhost:1234/v1 \
       --chat-model meta-llama/Llama-3.1-8B-Instruct

Outputs
- notebook_output/lite_{slug}_{level}_{timestamp}.json   (lesson json)
- notebook_output/lite_{slug}_{level}_{timestamp}.ipynb  (final notebook)

Notes
- Beginner focuses on one non‑streaming call and environment setup.
- Intermediate adds streaming, parameters, and basic telemetry.
- Advanced adds pairwise model compare and a tiny Elo demo (in‑notebook). 

Polyglot notebooks policy (important)
- ALAIN‑Kit content is often polyglot (Python + TypeScript/TSX/JSON). Jupyter notebooks have a single kernel, so only Python cells should execute.
- The builder must render non‑Python snippets as Markdown fenced code blocks (```ts, ```tsx, ```json), never as Python code cells.
- Normalize smart punctuation to ASCII in code blocks to avoid accidental syntax errors (quotes/dashes/ellipsis).
- Provide a top‑of‑notebook setup cell that maps provider env vars and seeds randomness deterministically.

Sanitizing generated notebooks
- Run the sanitizer to enforce the above before validation and commit:

```bash
python scripts/notebook_sanitize.py "Demo Notebooks/your_notebook.ipynb" --write
pytest --nbval --current-env
```

Brand/model naming
- Use OpenAI brand assets and names per https://openai.com/brand. Do not imply endorsement; avoid using model names as product titles. Treat "GPT‑5" as a placeholder unless the model is publicly available in your provider.
