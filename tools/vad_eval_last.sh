#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 <run-directory | legacy-config-name> [arguments...]" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
RUN="$1"
shift
if [[ -d "$RUN" ]]; then
    CONFIG="$RUN/default.py"
    RUN_DIR="$RUN"
else
    # Retain the original config-name convention for existing callers.
    CONFIG="projects/configs/$RUN.py"
    RUN_DIR="work_dirs/$RUN"
fi
[[ -f "$CONFIG" && -f "$RUN_DIR/latest.pth" ]] || { echo 'Saved configuration or latest.pth is missing.' >&2; exit 2; }
exec bash tools/vad_eval.sh "$CONFIG" "$RUN_DIR/latest.pth" \
    --show-dir "$RUN_DIR/eval" "$@"
