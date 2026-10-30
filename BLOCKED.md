# BLOCKED

Engineering blockers: **无 / none for the requested F1/F2 repair scope**. Both P2 findings are CLOSED after RED→GREEN regressions, 70 passing CPU tests (no skips), and Astra Ultra independent re-review of `4013778`. The reviewer added six passing CPU boundary probes and found no new blocking issue. See `research/p01/reviews/2026-10-30-astra-ultra-rereview/REVIEW.md`. F3 dataset-set validation remains a documented nonblocking P3, outside the requested two fixes; it has not been marked fixed.

Research gates PENDING by explicit user instruction: no GPU work this round. The initial audit recorded nvidia-smi exit 9; this is historical, not a current availability claim. This transfer follow-up did not probe GPU availability, run inference or allocate CUDA memory. Do not queue or wait for GPU availability.

Data provenance: existing LoveDA image manifests lack verified original scene IDs; actual split generation must wait for an evidence-backed scene map. iSAID parent image prefixes alone do not prove geographic independence. Released prompt pool construction images/splits/model revision are unknown; leakage-free construction is not proven.

GPU import/runtime and actual 504 support remain untested. CPU arithmetic evidence cannot close G0 or prove a scientific gain.

Target installation: requirements export and source-environment `pip check` are complete; installing those dependencies on a fresh Ubuntu 4090 machine remains untested. See TRANSFER_4090.md for the prepared commands and missing inputs.

Asset transfer update: required validation images/GT on 192.168.0.18 exactly match this machine (26,626 files). Dataset retransfer is unnecessary. Target weight is absent; verified weight and small metadata package are prepared locally. Target was accessed read-only, so actual upload and path aliases have not been applied. Use research/p01/asset-transfer/REMOTE_TRANSFER.md; source-scene evidence is still missing.
