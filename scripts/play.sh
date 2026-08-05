#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/mjlab"
exec uv run python -m simtoolreal_g1.rlg.play "$@"
