# BLOCKED

Engineering: **P2 fixes implemented; Astra Ultra re-review PENDING**. F1 prediction capacity checks and F2 report bank binding now have RED→GREEN regressions, and full CPU acceptance `cpu-p2-v1` passed 70 tests without skips. Strict sign-off and push await the requested independent re-review. The original findings and prior evidence are preserved in `research/p01/reviews/2026-10-30-astra-ultra/REVIEW.md`; repair evidence is in `research/p01/p2-fixes/evidence/`. F3 dataset-set validation remains a documented nonblocking P3, outside the requested two fixes.

Research gates PENDING by explicit user instruction: no GPU work this round. The initial audit recorded nvidia-smi exit 9; this is historical, not a current availability claim. This transfer follow-up did not probe GPU availability, run inference or allocate CUDA memory. Do not queue or wait for GPU availability.

Data provenance: existing LoveDA image manifests lack verified original scene IDs; actual split generation must wait for an evidence-backed scene map. iSAID parent image prefixes alone do not prove geographic independence. Released prompt pool construction images/splits/model revision are unknown; leakage-free construction is not proven.

GPU import/runtime and actual 504 support remain untested. CPU arithmetic evidence cannot close G0 or prove a scientific gain.

Target installation: requirements export and source-environment `pip check` are complete; installing those dependencies on a fresh Ubuntu 4090 machine remains untested. See TRANSFER_4090.md for the prepared commands and missing inputs.

Asset transfer update: required validation images/GT on 192.168.0.18 exactly match this machine (26,626 files). Dataset retransfer is unnecessary. Target weight is absent; verified weight and small metadata package are prepared locally. Target was accessed read-only, so actual upload and path aliases have not been applied. Use research/p01/asset-transfer/REMOTE_TRANSFER.md; source-scene evidence is still missing.
