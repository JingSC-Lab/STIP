#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: $0 <config> <checkpoint> [arguments...]" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
CONFIG="$1"
CHECKPOINT="$2"
shift 2
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" tools/test.py "$CONFIG" "$CHECKPOINT" \
    --launcher none --eval bbox --tmpdir .dist_test "$@"
