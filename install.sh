#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--cpu" ]]; then
  uv venv --allow-existing .venv
  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchvision==0.21.0
  uv pip install --python .venv/bin/python -r requirements.txt
elif [[ $# -eq 0 ]]; then
  uv sync --frozen --extra dev --extra metrics
else
  printf 'Usage: bash install.sh [--cpu]\n' >&2
  exit 2
fi
