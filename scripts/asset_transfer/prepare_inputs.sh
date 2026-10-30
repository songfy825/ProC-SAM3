#!/usr/bin/env bash
set -euo pipefail
P01_ASSETS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P01_CODE=/home/fengyi/CodeHome/Proc/src/ProC-SAM3
P01_SCENES=${1:-/data/p01-inputs/reviewed-scene-map.json}
P01_LIMITS=${P01_SPLIT_LIMITS:-'{"calibration":50,"pilot":50,"confirmation":80}'}
if [ ! -s "$P01_SCENES" ]; then
  echo "Missing reviewed source-scene map: $P01_SCENES. No split or experiment was started." >&2
  exit 2
fi
export P01_RUNTIME=/data/p01-runs P01_ENABLE_GPU=0 CUDA_VISIBLE_DEVICES=''
cd "$P01_CODE"
bash scripts/p01/python.sh -m proc_research.prepare inventory \
  --manifest-dir /data/source-manifests --output /data/p01-inputs/inventory-v1.jsonl
bash scripts/p01/python.sh -m proc_research.calibration.run split \
  --inventory /data/p01-inputs/inventory-v1.jsonl --scene-map "$P01_SCENES" \
  --limits "$P01_LIMITS" --output /data/p01-inputs/splits-v1.jsonl
bash scripts/p01/python.sh "$P01_ASSETS/select_label_metadata.py" \
  --manifest /data/p01-inputs/splits-v1.jsonl --inventory "$P01_ASSETS/evaluation/label_inventory.jsonl" \
  --output /data/p01-inputs
bash scripts/p01/launch.sh preflight --settings /data/p01-inputs/4090-v1.json
