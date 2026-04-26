# Hardware Matrix Decomposition

Decompose neural network weight matrices into sets of smaller matrices, each operating in a weight-stationary fashion suitable for hardware accelerator execution.

## Overview

Neural network inference relies on large matrix multiplications. This project explores decomposing those weight matrices into many smaller tiles that can each be mapped to a weight-stationary compute unit — keeping weights fixed in local storage while streaming activations through.

## Setup

```bash
uv sync
uv sync --extra dev  # include dev tools
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy src/
```
