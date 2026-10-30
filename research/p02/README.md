# P02 CPU offline action-space diagnostic

All commands run in `/home/fengyi/CodeHome/Proc/src/ProC-SAM3-p02`.
The verified interpreter is `/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python` (Python 3.12, NumPy 1.26.4, torch 2.7.1+cu126). The wrapper uses the ELF loader when the filesystem disallows executable files. No environment installation is required.

```bash
cd /home/fengyi/CodeHome/Proc/src/ProC-SAM3-p02
bash scripts/p02/python.sh scripts/p02/analyze.py --help
bash scripts/p02/python.sh scripts/p02/verify.py --help
bash scripts/p02/python.sh scripts/p02/verify.py --output /home/fengyi/CodeHome/Proc/runtime/p02/audit/acceptance-new
```

The wrapper fixes `P01_ENABLE_GPU=0`, `CUDA_VISIBLE_DEVICES=''`, offline variables and `P01_RUNTIME=/home/fengyi/CodeHome/Proc/runtime/p02/compat`, overriding inherited GPU/runtime values. Entrypoints install CPU and network guards before optional library imports. Old transform-only tests import SAM3 processor code but construct no model or run inference. CUDA availability is a fixed False policy answer, not a hardware query. Original P01 tests and model source remain unchanged.

`verify.py` runs unchanged 60 P01 CPU contracts, all P02 hand/negative tests, both CLI help/usage failures, actual synthetic success/corrupt receipt/restoration/missing-cache CLI checks, and byte-compares actions after GT replacement/deletion. Outputs preserve exact commands, expected/observed exit codes, input/source/log hashes in `verification.json`. Synthetic fixtures are arithmetic only and explicitly named. Existing output directories fail rather than overwrite.

Resume real diagnosis only when the required evidence already exists. Copy `research/p02/input.real.template.json` to `runtime/p02/inputs/real.json`, fill each path and SHA256 using genuine existing records, following `input.PROVENANCE.md`. Do not synthesize an independent receipt to make a cache look real. Missing external evidence remains PENDING.

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest /home/fengyi/CodeHome/Proc/runtime/p02/inputs/real.json \
  --output /home/fengyi/CodeHome/Proc/runtime/p02/real-actions-v1
```

Optional GT enters the independent evaluator only:

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest /home/fengyi/CodeHome/Proc/runtime/p02/inputs/real.json \
  --labels /home/fengyi/CodeHome/Proc/runtime/p02/inputs/labels.jsonl \
  --output /home/fengyi/CodeHome/Proc/runtime/p02/real-evaluated-v1
```

Label JSONL rows: `dataset,image_id,path,sha256,image_sha256`, optional `scene_id`. Relative paths resolve beside the label manifest. Never put GT fields into development/cache selection manifests. Missing any selected GT preserves actions and makes that dataset's evaluation PENDING_GT; corrupt supplied GT fails. Both datasets keep separate pooled confusion matrices. Source scene IDs do not prove geographic independence.

Without inputs, the following command deliberately emits PENDING and exits 2:

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest research/p02/input.real.template.json \
  --output /home/fengyi/CodeHome/Proc/runtime/p02/real-pending-new
```

Exit codes: `0` completed diagnostics (synthetic or GT-free is not real scientific validation); `1` corrupt/invalid evidence or verification failure; `2` missing required cache/provenance or CLI usage error. Inspect `status.json` and the command log to distinguish them. All output destinations must be new directories under `runtime/p02`.

Artifacts: `actions.json` shared candidates/aliases/actual changes and per-region stats; `summary.json` separate dataset aggregates plus image/region stats and frozen selector outputs; `evaluation.json` pooled metrics, bounded oracle trajectories, damage/repair/per-class counts when GT exists; `status.json` input provenance and limitations; `output_hashes.json`. Read METHOD.md for every frozen rule, including deterministic ties and budgets. The oracle is a bounded reference, never a global optimum or upper bound. No inference command is provided by this round.
