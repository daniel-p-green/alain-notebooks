# AGENTS Guide for alain-notebooks

This repository collects example notebooks generated with ALAIN to demonstrate reuseable AI workflows and prompts. Use this guide alongside `README.md` to help automation-friendly agents reproduce the environment, extend notebooks responsibly, and ship reliable updates without guesswork.

## Setup commands
Create an isolated Python 3.11 environment, install notebook tooling, and add any ALAIN-specific extras your workflow requires.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip jupyterlab nbclient nbformat black isort pytest nbval
python -m pip install [INSTALL_COMMAND]  # replace with the ALAIN package or extra dependencies for this notebook set
```

## Code style
- Target Python 3.11 syntax and keep notebooks executable from top to bottom with deterministic seeds where randomness appears.
- Prefer standard libraries and well-supported ML/DS packages (numpy, pandas, matplotlib) unless a notebook documents an ALAIN-specific plugin.
- Use descriptive notebook filenames (`topic-intent.ipynb`), lowercase module names inside JSON assets, and snake_case variables.
- Strip unnecessary notebook output before committing; retain lightweight visuals that explain intent.
- Format embedded Python cells with `black` (line length 88) and sort imports with `isort` using the `black` profile.

## Testing instructions
- Execute notebooks in a clean environment before pushing to ensure metadata and outputs match expectations.
- Run automated checks to catch regressions in code cells and documentation snippets.

```bash
pytest --nbval --current-env
jupyter nbconvert --to notebook --execute path/to/notebook.ipynb --output /tmp/validation.ipynb
```

- For long-running notebooks, add lightweight smoke tests or parameterized subsets and document any skipped cells.

## PR instructions
- Branch from `main`, keep PRs focused on a coherent notebook or feature set, and summarize agent-facing changes in the description.
- Follow Conventional Commits (e.g., `feat: add summarization walkthrough`) so downstream automations can parse history.
- Before requesting review, run the setup and testing commands, ensure notebooks re-execute, and confirm that outputs are trimmed.
- Mention any external data sources or secrets an agent must provision separately; never commit tokens or personally identifiable content.

## Resources
- `README.md` — high-level project overview to pair with this agent guide.
- https://agents.md/ — specification and examples for interoperable AGENTS documentation.
- https://alain.run/docs — reference for ALAIN workflows, CLI options, and notebook exports.
- Example ALAIN recipes: https://github.com/search?q=alain+notebooks&type=repositories (review before proposing structural changes).
