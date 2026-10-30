#!/usr/bin/env bash
set -euo pipefail
P02_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export P01_ENABLE_GPU=0 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export P01_RUNTIME=/home/fengyi/CodeHome/Proc/runtime/p02/compat
export P01_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec bash "$P02_CODE/scripts/p01/python.sh" "$@"
