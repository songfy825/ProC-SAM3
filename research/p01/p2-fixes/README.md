# P01 F1/F2 repairs

User-authorized CPU-only repairs following Astra Ultra review of `ba663c4`, starting from archived review commit `e8e07cb`. Independent re-review is required before push. F3 is a recorded nonblocking P3 and was not changed in this scope.

F1 now checks the full int64 prediction budget on the output filesystem before creating the decode run. It includes 64 KiB per file for NPY header/allocation and 1 MiB plus 64 KiB per image for metadata/index/receipt headroom. It checks the remaining budget before every prediction write and again before final metadata. Other processes can still change disk space between system calls; these guards do not reserve disk blocks. A failure retains incomplete CPU evidence, and the existing explicit launcher recovery reuses the completed upstream cache.

F2 validates the calibration bank's fit stage/kind, its exact completion-receipt hash against the prediction manifest, and any confirmation frozen-bank binding before reading evaluation results or creating report output. Same-experiment identity alone is insufficient.

## Actual verification

- `test_p2_regressions.py` was frozen before implementation: SHA256 `33a36d9a6365d83114d81560ab6168cd57e1b200554af420749b45dabb5884f4`.
- RED: 10 tests; 9 expected failing assertion records including two stage/kind subcases; no errors; exit 1. F1 GREEN: five tests passed. Combined GREEN: ten tests passed, unchanged assertions, exit 0.
- Full acceptance `cpu-p2-v1`: 70 tests passed, no skips; three deliberately corrupted subprocess checks exit 1 and restored checks exit 0; frozen contract, untouched original model, six CLI help entries and whitespace checks pass. Overall exit 0.
- Current production digest: `df52a7636ed6fb0de5a613ceea513c68632fdcca7f61c8d415ce900d45caab59`. METHOD/config hashes unchanged. Historical 15-test standalone packaging acceptance was not rerun because packaging is unchanged.
- Receipts and log hashes: `evidence/`; full logs: `/home/fengyi/CodeHome/Proc/runtime/p01/audit/p2-fixes-v1/` and `/home/fengyi/CodeHome/Proc/runtime/p01/audit/cpu-p2-v1/`. All new tensor/prediction/label data are CPU synthetic; no model construction or GPU acceptance.

Actual full acceptance command (already executed; use a new output path for future runs):

```bash
CUDA_VISIBLE_DEVICES='' P01_ENABLE_GPU=0 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  P01_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python \
  P01_RUNTIME=/home/fengyi/CodeHome/Proc/runtime/p01/audit/p2-fixes-v1/runtime \
  bash scripts/p01/python.sh scripts/p01/verify.py \
  --output /home/fengyi/CodeHome/Proc/runtime/p01/audit/cpu-p2-v1
```

On another host set P01_PYTHON to the installed environment. Changed production code has a new identity; old receipts cannot be relabeled as evidence for this code. No real GPU run exists for either revision. GPU G0, real scene provenance, pilot, confirmation and scientific conclusions remain PENDING.
