#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: $0 <config> <num-gpus> [arguments...]" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
CONFIG="$1"
GPUS="$2"
shift 2
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
    --nproc_per_node="$GPUS" --master_port="${PORT:-28509}" \
    tools/train.py "$CONFIG" --launcher pytorch --deterministic "$@"
