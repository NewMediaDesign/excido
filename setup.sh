#!/usr/bin/env bash
# Excido setup for macOS / Linux. Same as:  uv sync && uv run excido setup
#
#   bash setup.sh               core models (~3.7 GB)
#   bash setup.sh all           all 16 models (~10 GB)
#   bash setup.sh none          no models (download them from the app)
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it, open a new terminal and run this script again:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv sync
uv run --frozen excido setup --models "${1:-core}"
