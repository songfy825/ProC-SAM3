#!/usr/bin/env bash
set -euo pipefail
P02_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export P01_ENABLE_GPU=0 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Select only this machine's explicit/active/local environment. An invalid
# explicit choice must fail; never fall back to another machine's P01 path.
if [[ ${P02_PYTHON+x} ]]; then
  P02_INTERPRETER=$P02_PYTHON
elif [[ -n ${VIRTUAL_ENV:-} ]]; then
  P02_INTERPRETER=$VIRTUAL_ENV/bin/python
elif [[ -n ${CONDA_PREFIX:-} ]]; then
  P02_INTERPRETER=$CONDA_PREFIX/bin/python
elif [[ -f "$P02_CODE/.venv/bin/python" ]]; then
  P02_INTERPRETER=$P02_CODE/.venv/bin/python
elif [[ -f "$P02_CODE/.p01-env/bin/python" ]]; then
  P02_INTERPRETER=$P02_CODE/.p01-env/bin/python
else
  P02_INTERPRETER=$(command -v python3 || true)
fi
if [[ ! -f "$P02_INTERPRETER" ]]; then
  echo "P02_PYTHON must select an existing Python interpreter: $P02_INTERPRETER" >&2
  exit 2
fi
P02_BOOTSTRAP=("$P02_INTERPRETER")
if [[ ! -x "$P02_INTERPRETER" ]]; then
  if [[ -x /lib64/ld-linux-x86-64.so.2 ]]; then
    P02_BOOTSTRAP=(/lib64/ld-linux-x86-64.so.2 "$P02_INTERPRETER")
  else
    echo "P02_PYTHON is not executable: $P02_INTERPRETER" >&2
    exit 2
  fi
fi

# Stdlib-only bootstrap validates paths before P01's wrapper creates anything.
# Isolated mode also prevents inherited Python site hooks from loading models.
P02_RESOLVED_RUNTIME=$("${P02_BOOTSTRAP[@]}" -I -S -c '
import pathlib, sys
path = pathlib.Path(sys.argv[1]).expanduser()
if not path.is_absolute():
    sys.exit("P02_RUNTIME must be an absolute path (or start with ~)")
print(path.resolve())
' "${P02_RUNTIME-$P02_CODE/.p02-runtime}")
export P02_RUNTIME=$P02_RESOLVED_RUNTIME
export P01_RUNTIME=$P02_RUNTIME/compat
export P02_PYTHON=$P02_INTERPRETER P01_PYTHON=$P02_INTERPRETER
exec bash "$P02_CODE/scripts/p01/python.sh" "$@"
