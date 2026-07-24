# Repository Guidelines

## Project Structure & Module Organization

`cs336_alignment/` contains the Python package and prompt utilities. Put reusable implementation code there. `tests/` contains pytest modules, fixtures, adapter stubs, and NumPy snapshots. Implement assignment-facing adapters in `tests/adapters.py` as directed by the handout. `data/` stores evaluation datasets; `scripts/` contains evaluation helpers. The root PDFs are authoritative specifications; `README.md` covers setup and `CHANGELOG.md` records releases.

## Build, Test, and Development Commands

- `uv sync --no-install-package flash-attn`: install the base environment first.
- `uv sync`: finish dependency installation.
- `uv run pytest`: run the complete test suite configured under `tests/`.
- `uv run pytest tests/test_grpo.py -v`: run the required GRPO tests with verbose output.
- `uv run pytest tests/test_metrics.py -k entropy`: run a focused test selection while iterating.
- `bash test_and_make_submission.sh`: run GRPO tests and replace `code.zip`; use only when preparing a submission.

Python 3.12 is required. There is no separate build step; `uv` installs the package from the repository.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8 spacing, and type hints for public functions. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Document tensor shapes and return keys. Write comments in English. No formatter or linter is configured, so match nearby style.

## Testing Guidelines

Tests use pytest and `test_<behavior>` naming. Add cases to the matching `tests/test_*.py` module and reuse `tests/conftest.py` fixtures. Snapshots live in `tests/_snapshots/`; update them only for intentional behavior changes. No coverage threshold is configured. Run targeted tests during development and the full suite before a PR.

## Commit & Pull Request Guidelines

History favors short descriptive subjects. For new commits, use Conventional Commit prefixes such as `fix: correct response mask alignment` or `docs: clarify GRPO setup`. PRs must explain context, implementation, testing, and known limitations; link relevant issues and include screenshots only for visual output changes.
