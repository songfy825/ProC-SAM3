# BLOCKED

Engineering: **REQUEST_CHANGES after the 2026-10-30 Astra Ultra strict review**. Two confirmed P2 gaps remain: CPU prediction writes lack the 20 GiB reserve gate (F1), and report generation does not require the exact bank receipt used by its predictions (F2). The prior 60-test CPU acceptance and 15-test migration evidence remain valid historical results; they did not cover these cases. No implementation was changed in this review. See `research/p01/reviews/2026-10-30-astra-ultra/REVIEW.md`. The user has authorized pushing this task branch to `songfy825/ProC-SAM3`; no merge is requested.

Research gates PENDING by explicit user instruction: no GPU work this round. The initial audit recorded nvidia-smi exit 9; this is historical, not a current availability claim. This transfer follow-up did not probe GPU availability, run inference or allocate CUDA memory. Do not queue or wait for GPU availability.

Data provenance: existing LoveDA image manifests lack verified original scene IDs; actual split generation must wait for an evidence-backed scene map. iSAID parent image prefixes alone do not prove geographic independence. Released prompt pool construction images/splits/model revision are unknown; leakage-free construction is not proven.

GPU import/runtime and actual 504 support remain untested. CPU arithmetic evidence cannot close G0 or prove a scientific gain.

Target installation: requirements export and source-environment `pip check` are complete; installing those dependencies on a fresh Ubuntu 4090 machine remains untested. See TRANSFER_4090.md for the prepared commands and missing inputs.

Asset transfer update: required validation images/GT on 192.168.0.18 exactly match this machine (26,626 files). Dataset retransfer is unnecessary. Target weight is absent; verified weight and small metadata package are prepared locally. Target was accessed read-only, so actual upload and path aliases have not been applied. Use research/p01/asset-transfer/REMOTE_TRANSFER.md; source-scene evidence is still missing.
