#!/usr/bin/env sh
# Run every gate CI runs, in the same order. One green run here means a green CI.
# Usage: scripts/check.sh

set -e
cd "$(dirname "$0")/.."

sh scripts/uv.sh sync --locked --quiet
sh scripts/uv.sh run ruff check
sh scripts/uv.sh run ruff format --check
sh scripts/uv.sh run mypy
sh scripts/uv.sh run pytest --cov
