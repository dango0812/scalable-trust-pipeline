#!/usr/bin/env sh
# Run uv whether it is on PATH or only installed as a Python module (pip install uv).
# Usage: scripts/uv.sh <uv arguments...>

if command -v uv >/dev/null 2>&1; then
  exec uv "$@"
fi
if python -m uv --version >/dev/null 2>&1; then
  exec python -m uv "$@"
fi
echo "x uv is required (https://docs.astral.sh/uv/)." >&2
exit 1
