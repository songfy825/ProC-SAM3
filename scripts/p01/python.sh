#!/usr/bin/env bash
set -euo pipefail
P01_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
P01_RUNTIME=${P01_RUNTIME:-/home/fengyi/CodeHome/Proc/runtime/p01}
mkdir -p "$P01_RUNTIME/cache" "$P01_RUNTIME/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$P01_RUNTIME/cache/hf" TORCH_HOME="$P01_RUNTIME/cache/torch"
export XDG_CACHE_HOME="$P01_RUNTIME/cache" TMPDIR="$P01_RUNTIME/tmp"
export TRITON_CACHE_DIR="$P01_RUNTIME/cache/triton" MPLCONFIGDIR="$P01_RUNTIME/cache/matplotlib"
export PYTHONPATH="$P01_CODE${PYTHONPATH:+:$PYTHONPATH}"
# CPU is the default, including contract tests. GPU requires a later explicit command.
if [ "${P01_ENABLE_GPU:-0}" != 1 ]; then export CUDA_VISIBLE_DEVICES=''; fi
exec /lib64/ld-linux-x86-64.so.2 "${P01_PYTHON:-/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python}" "$@"
