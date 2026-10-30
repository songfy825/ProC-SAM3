#!/usr/bin/env bash
set -euo pipefail
P01_LAUNCH_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$P01_LAUNCH_CODE/scripts/p01/python.sh" -m proc_research.launch "$@"
