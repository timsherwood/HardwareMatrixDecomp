# CLAUDE.md

## Project

Hardware Matrix Decomposition — experimental Python project for decomposing NN weight matrices into smaller tiles for weight-stationary hardware execution.

## Stack

- Python 3.12+, managed with `uv`
- Core deps: numpy, torch
- Dev tools: pytest, ruff, mypy

## Commands

- `uv run pytest` — run tests
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run mypy src/` — type check

## Conventions

- Source code lives in `src/hardware_matrix_decomp/`
- Tests live in `tests/`
- Use type annotations throughout
- Follow ruff's default style (configured in pyproject.toml)
