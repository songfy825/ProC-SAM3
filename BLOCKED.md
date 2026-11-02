# BLOCKED

2026-11-02 scope update: the target 4090 GPU is authorized for the standalone G0 gate and, only after it passes, calibration/pilot. See `research/p01/execution/GOAL_4090.md`; confirmation remains excluded. This source-machine filename repair uses CPU tests only. Historical CPU-only scope below does not revoke the target-machine authorization.

Engineering blockers: **无 / none for the requested F1/F2 repair scope**. Both P2 findings are CLOSED after RED→GREEN regressions, 70 passing CPU tests (no skips), and Astra Ultra independent re-review of `4013778`. The reviewer added six passing CPU boundary probes and found no new blocking issue. See `research/p01/reviews/2026-10-30-astra-ultra-rereview/REVIEW.md`. F3 dataset-set validation remains a documented nonblocking P3, outside the requested two fixes; it has not been marked fixed.

Research gates PENDING by explicit user instruction: no GPU work this round. The initial audit recorded nvidia-smi exit 9; this is historical, not a current availability claim. This transfer follow-up did not probe GPU availability, run inference or allocate CUDA memory. Do not queue or wait for GPU availability.

Data provenance: existing LoveDA image manifests lack verified original scene IDs; actual split generation must wait for an evidence-backed scene map. iSAID parent image prefixes alone do not prove geographic independence. Released prompt pool construction images/splits/model revision are unknown; leakage-free construction is not proven.

GPU import/runtime and actual 504 support remain untested. CPU arithmetic evidence cannot close G0 or prove a scientific gain.

Target installation: requirements export and source-environment `pip check` are complete; installing those dependencies on a fresh Ubuntu 4090 machine remains untested. See TRANSFER_4090.md for the prepared commands and missing inputs.

Asset transfer update: the 2026-10-30 read-only audit matched required validation images/GT on 192.168.0.18 (26,626 files). On 2026-11-02 the user confirmed the target weight is now at `/home/fengyi/sam3.pt`; target preflight must still verify its size/hash. Do not repeat the old weight upload based on the historical missing-file record. Use research/p01/asset-transfer/REMOTE_TRANSFER.md; source-scene evidence is still missing.
