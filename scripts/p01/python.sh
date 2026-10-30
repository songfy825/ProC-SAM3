#!/usr/bin/env bash
set -euo pipefail
P01_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export P01_RUNTIME=${P01_RUNTIME:-"$P01_CODE/.p01-runtime"}
mkdir -p "$P01_RUNTIME/cache" "$P01_RUNTIME/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$P01_RUNTIME/cache/hf" TORCH_HOME="$P01_RUNTIME/cache/torch"
export XDG_CACHE_HOME="$P01_RUNTIME/cache" TMPDIR="$P01_RUNTIME/tmp"
export TRITON_CACHE_DIR="$P01_RUNTIME/cache/triton" MPLCONFIGDIR="$P01_RUNTIME/cache/matplotlib"
export PYTHONPATH="$P01_CODE"
# CPU is the default, including contract tests. GPU requires a later explicit command.
if [ "${P01_ENABLE_GPU:-0}" != 1 ]; then export CUDA_VISIBLE_DEVICES=''; fi
if [ -n "${P01_PYTHON:-}" ]; then
    P01_INTERPRETER=$P01_PYTHON
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    P01_INTERPRETER="$CONDA_PREFIX/bin/python"
elif [ -f "$P01_CODE/.p01-env/bin/python" ]; then
    P01_INTERPRETER="$P01_CODE/.p01-env/bin/python"
elif [ -f /home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python ]; then
    # Backward compatibility on the original host only; no remote dependency.
    P01_INTERPRETER=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python
else
    echo 'Set P01_PYTHON or activate the supplied p01-sam3 Conda environment.' >&2
    exit 2
fi
if [ -x "$P01_INTERPRETER" ]; then exec "$P01_INTERPRETER" "$@"; fi
if [ -f /lib64/ld-linux-x86-64.so.2 ]; then exec /lib64/ld-linux-x86-64.so.2 "$P01_INTERPRETER" "$@"; fi
echo 'Python is not executable; restore the environment on a Linux filesystem.' >&2
exit 2
