#!/usr/bin/env bash
# Start Excido on http://127.0.0.1:7860 and open the browser. Same as:  uv run excido
# Extra arguments are passed through, e.g.:  bash run.sh --port 8000 --no-browser --offline
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --frozen excido serve "$@"
