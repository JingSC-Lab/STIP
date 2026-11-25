#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config-name> <num-gpus> [train.py arguments...]" >&2
    exit 2
fi

CONFIG_NAME="$1"
GPUS="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONFIG="${REPO_ROOT}/projects/configs/${CONFIG_NAME}.py"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
# Configuration files own defaults; only explicit environment values override them.
if [[ -n "${INIT_SEED:-}" || -n "${TRAIN_SEED:-}" ]]; then
    echo "INIT_SEED/TRAIN_SEED are no longer supported; use SEED or config seed." >&2
    exit 2
fi
OVERRIDE_ARGS=()
for setting in SEED EPOCHS HISTORY_MODE EMA SAFETY; do
    if [[ -n "${!setting:-}" ]]; then
        case "$setting" in
            SEED) flag=--seed ;;
            EPOCHS) flag=--epochs ;;
            HISTORY_MODE) flag=--history-mode ;;
            EMA) flag=--ema ;;
            SAFETY) flag=--safety ;;
        esac
        OVERRIDE_ARGS+=("$flag" "${!setting}")
    fi
done
CONFIG_LABEL="${CONFIG_NAME//\//_}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! "${GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num-gpus must be a positive integer: ${GPUS}" >&2
    exit 2
fi
if [[ -z "${PORT}" ]]; then
    PORT="$("${PYTHON_BIN}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
elif [[ ! "${PORT}" =~ ^[1-9][0-9]*$ ]] || (( PORT > 65535 )); then
    echo "PORT must be an integer in [1, 65535]: ${PORT}" >&2
    exit 2
fi

# Keep temporary files and generated caches inside the repository.
# A short TMPDIR also avoids multiprocessing's AF_UNIX address length limit.
LAW_RUNTIME_DIR="$(realpath -m -- "${REPO_ROOT}/work_dirs/.runtime")"
for runtime_subdir in tmp mpl kernels; do
    runtime_path="$(realpath -m -- "${LAW_RUNTIME_DIR}/${runtime_subdir}")"
    case "${runtime_path}/" in
        "${REPO_ROOT}/"*) ;;
        *) echo "Runtime path is outside the repository: ${runtime_path}" >&2; exit 2 ;;
    esac
    mkdir -p -- "${runtime_path}"
done
export TMPDIR="${LAW_RUNTIME_DIR}/tmp"
export MPLCONFIGDIR="${LAW_RUNTIME_DIR}/mpl"
export PYTORCH_KERNEL_CACHE_PATH="${LAW_RUNTIME_DIR}/kernels"
export PYTHONDONTWRITEBYTECODE=1
export LAW_TENSOR_SHARING="${LAW_TENSOR_SHARING:-file_system}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0  # The training process uses the resolved seed below.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "${REPO_ROOT}"
# Resolve the actual CLI/config values before naming the run.
RUN_METADATA="$("${PYTHON_BIN}" - "${CONFIG}" \
    "${OVERRIDE_ARGS[@]}" "$@" <<'PY_RUN'
import argparse
import math
import sys
from pathlib import Path
from mmcv import Config, DictAction

sys.path.insert(0, str(Path.cwd() / 'tools'))
from law_train_options import add_law_arguments, resolve_law_options

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--seed', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--cfg-options', nargs='+', action=DictAction)
parser.add_argument('--options', nargs='+', action=DictAction)
add_law_arguments(parser)
args, _ = parser.parse_known_args()
if args.options and args.cfg_options:
    parser.error('--options and --cfg-options cannot be both specified')
cfg = Config.fromfile(args.config)
overrides = args.options or args.cfg_options
if overrides is not None:
    cfg.merge_from_dict(overrides)
resolve_law_options(cfg, args)
clip = cfg.get('optimizer_config', {}).get('grad_clip')
if clip is None:
    clip_label = 'clipoff'
else:
    if not isinstance(clip, dict) or 'max_norm' not in clip:
        parser.error('grad_clip must be a dict with max_norm or actual None. '
                     'To disable via CLI, use optimizer_config._delete_=True '
                     'optimizer_config.type=OptimizerHook')
    threshold = float(clip['max_norm'])
    if not math.isfinite(threshold) or threshold < 0:
        parser.error('grad_clip.max_norm must be finite and non-negative')
    label = str(int(threshold)) if threshold.is_integer() else repr(threshold)
    clip_label = 'clip' + label
ema_label = f"emaon_d{cfg.ema_decay:g}" if cfg.ema_enabled else 'emaoff'
safety = cfg.model.safety_cfg
safety_label = 'safetyoff'
if safety['enabled']:
    safety_label = (f"safetyon_w{safety['weight']:g}_d{safety['delay_epochs']}"
                    f"_wu{safety['warmup_epochs']}_m{safety['margin']:g}")
print(f'{cfg.history_mode}_{ema_label}_{clip_label}_{safety_label}')
print(cfg.seed)
print(cfg.total_epochs)
lr = cfg.get('lr_config', {})
if lr.get('policy') == 'CosineRestart' and lr.get('by_epoch', True):
    print('cos' + '-'.join(str(p) for p in lr['periods']))
elif lr.get('policy') == 'CosineAnnealing':
    print(f'cos{cfg.total_epochs}')
else:
    print('lr' + str(lr.get('policy', 'fixed')))
PY_RUN
)"
mapfile -t RUN_FIELDS <<< "${RUN_METADATA}"
if [[ ${#RUN_FIELDS[@]} -ne 4 ]]; then
    echo "Unexpected configuration metadata: ${RUN_METADATA}" >&2
    exit 2
fi
RUN_CONFIG_LABEL="${RUN_FIELDS[0]}"
RESOLVED_SEED="${RUN_FIELDS[1]}"
RESOLVED_EPOCHS="${RUN_FIELDS[2]}"
LR_LABEL="${RUN_FIELDS[3]}"
SEED_LABEL="seed${RESOLVED_SEED}"
export PYTHONHASHSEED="${RESOLVED_SEED}"
RUN_NAME="${RUN_NAME:-${CONFIG_LABEL}_${RUN_CONFIG_LABEL}_${SEED_LABEL}_${RESOLVED_EPOCHS}epochs_${LR_LABEL}_${RUN_TIMESTAMP}}"
WORK_DIR="${WORK_DIR:-work_dirs/${RUN_NAME}}"

echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${GPUS}; ${SEED_LABEL}; epochs: ${RESOLVED_EPOCHS}; port: ${PORT}"
echo "Python: ${PYTHON_BIN##*/}; seed: ${RESOLVED_SEED}; schedule: ${LR_LABEL}"
echo "Resolved run configuration: ${RUN_CONFIG_LABEL}"
echo "Work directory: ${WORK_DIR}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${REPO_ROOT}/tools/train.py" "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    --deterministic \
    "${OVERRIDE_ARGS[@]}" "$@"
