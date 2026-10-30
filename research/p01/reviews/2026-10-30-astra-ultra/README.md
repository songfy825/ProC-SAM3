# Astra Ultra review archive

User-requested independent review by `gpt-6-astra` / `ultra`, on 2026-10-30. Reviewed implementation: `ba663c4bfe65c047932aaf0271d6ca27503b4c50`. This archive and accompanying status corrections do not modify that implementation or fix the findings.

Read [REVIEW.md](REVIEW.md) for the complete verdict and precise findings. Method: conditional pass. Strict engineering: request changes (two confirmed P2 gaps). Actual experiment: not ready. Research: PENDING. Historical tests remain valid; they do not cover all newly found gaps.

`archive.json` hashes exact copies of the independent report and small evidence files. `review-final-verification.json` validates the review artifacts, not the implementation contract. `provenance.json` is the parent's 40-file historical evidence audit. `cpu-probe-results.json` and `boundary-checks.json` contain new CPU-only findings. The full log and synthetic arrays are retained under `/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-review-v1/`, outside Git. No real image/GT bytes, checkpoint, credentials or environment binary is included here.

## Optional reproduction after inspecting the archived evidence

The existing reproduction already exited 0. Do not rerun merely to reproduce passed work. If needed for a targeted fix, copy the exact script to a new runtime directory so its synthetic outputs remain outside Git. From the repository root on the current source machine:

```bash
P01_REVIEW_DIR=$(mktemp -d /home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-reprobe.XXXXXX)
cp research/p01/reviews/2026-10-30-astra-ultra/reproduce_cpu.py "$P01_REVIEW_DIR/"
CUDA_VISIBLE_DEVICES='' P01_ENABLE_GPU=0 PYTHONDONTWRITEBYTECODE=1 \
  P01_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python \
  P01_RUNTIME="$P01_REVIEW_DIR" \
  bash scripts/p01/python.sh "$P01_REVIEW_DIR/reproduce_cpu.py"
```

This is a reproduction script for the reviewed revision, not a post-fix acceptance suite: successful execution demonstrates the recorded gaps. It uses synthetic scores and labels, never constructs SAM3, and asserts CUDA remains uninitialized. After fixes, turn the relevant cases into tests requiring safe rejection, and record a new code hash and run ID.
