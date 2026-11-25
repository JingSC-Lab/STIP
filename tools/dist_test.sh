#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 3 ]] || { echo "Usage: $0 <config> <checkpoint> <num-gpus> [arguments...]" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
CONFIG="$1"
CHECKPOINT="$2"
GPUS="$3"
shift 3
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
    --nproc_per_node="$GPUS" --master_port="${PORT:-29503}" \
    tools/test.py "$CONFIG" "$CHECKPOINT" --launcher pytorch --eval bbox "$@"
