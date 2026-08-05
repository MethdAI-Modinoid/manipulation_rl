#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/mjlab"

uv sync

echo
echo "Setup complete. Where each package resolves from:"
uv run python -c "
import mjlab, rl_games, simtoolreal_g1, gym, omegaconf
for m in (mjlab, rl_games, simtoolreal_g1, gym, omegaconf):
    print(f'  {m.__name__:16} {m.__file__}')
"
