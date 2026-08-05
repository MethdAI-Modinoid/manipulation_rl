#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: ./scripts/view.sh {view_scene|view_live|verify_pose} [args...]" >&2
    exit 1
fi

MODULE="$1"
shift

case "$MODULE" in
    view_scene|view_live|verify_pose) ;;
    *) echo "unknown target '$MODULE' (expected view_scene, view_live or verify_pose)" >&2
       exit 1 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/mjlab"
exec uv run python -m "simtoolreal_g1.$MODULE" "$@"
