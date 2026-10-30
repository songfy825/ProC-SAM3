# P02 CPU offline action-space diagnostic

P02 analyzes existing, independently evidenced 1008-view caches on CPU. It does not construct a model, produce GPU caches, download assets, or turn synthetic verification into a scientific result. Missing real evidence remains `REAL_ACTION_SPACE=PENDING`.

For cloning the P02 branch on another machine, dependencies, path configuration, and Codex handoff, read [REMOTE_RUN.md](REMOTE_RUN.md). Run commands from the checkout root with a compatible Python environment already active:

```bash
export P02_PYTHON="$(python -c 'import sys; print(sys.executable)')"
export P02_RUNTIME="$HOME/runs/proc-sam3-p02"
P02_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/p02/python.sh scripts/p02/analyze.py --help
bash scripts/p02/python.sh scripts/p02/verify.py --help
bash scripts/p02/python.sh scripts/p02/verify.py \
  --output "$P02_RUNTIME/audit/cpu-$P02_RUN_ID"
```

Interpreter selection: explicit `P02_PYTHON` → active `VIRTUAL_ENV` → `CONDA_PREFIX` → checkout `.venv` → checkout `.p01-env` → `python3` on PATH. An invalid explicit `P02_PYTHON` fails; it does not silently choose another environment. Selection does not guarantee that dependencies are installed. `P02_RUNTIME` must be absolute (a leading `~` is supported); its default is `<checkout>/.p02-runtime`. Every output must be a **new child directory** under the resolved runtime, never the runtime root itself.

The wrapper fixes `P01_ENABLE_GPU=0`, `CUDA_VISIBLE_DEVICES=''`, offline variables and `P01_RUNTIME=<resolved P02_RUNTIME>/compat`, overriding inherited P01 GPU/runtime/interpreter settings. Entrypoints install CPU and network guards before optional library imports. Old transform-only tests import SAM3 processor code but construct no model or run inference. CUDA availability is a fixed False policy answer, not a hardware query. Original P01 tests and model source remain unchanged.

`verify.py` runs unchanged 60 P01 CPU contracts, all P02 hand/negative tests, both CLI help/usage failures, actual synthetic success/corrupt receipt/restoration/missing-cache CLI checks, and byte-compares actions after GT replacement/deletion. Outputs preserve exact commands, expected/observed exit codes, input/source/log hashes in `verification.json`. Synthetic fixtures are arithmetic only and explicitly named. Use the current run's receipt for verification results; historical reports retain their original paths and limitations. Existing output directories fail rather than overwrite.

Resume real diagnosis only when the required evidence already exists. Copy [input.real.template.json](input.real.template.json) to `$P02_RUNTIME/inputs/real.json`, fill each path and SHA256 using genuine existing records, following [input.PROVENANCE.md](input.PROVENANCE.md). Relative manifest references resolve beside the manifest. The `R/runtime/p02` examples in that historical contract refer to the configured runtime on the new machine. Do not synthesize an independent receipt to make a cache look real. Missing external evidence remains PENDING.

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest "$P02_RUNTIME/inputs/real.json" \
  --output "$P02_RUNTIME/real-actions-$P02_RUN_ID"
```

Optional GT enters the independent evaluator only:

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest "$P02_RUNTIME/inputs/real.json" \
  --labels "$P02_RUNTIME/inputs/labels.jsonl" \
  --output "$P02_RUNTIME/real-evaluated-$P02_RUN_ID"
```

Label JSONL rows: `dataset,image_id,path,sha256,image_sha256`, optional `scene_id`. Relative paths resolve beside the label manifest. Never put GT fields into development/cache selection manifests. Missing any selected GT preserves actions and makes that dataset's evaluation PENDING_GT; corrupt supplied GT fails. Both datasets keep separate pooled confusion matrices. Source scene IDs do not prove geographic independence.

Without inputs, the following command deliberately emits PENDING and exits 2:

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest research/p02/input.real.template.json \
  --output "$P02_RUNTIME/real-pending-$P02_RUN_ID"
```

Exit codes: `0` completed diagnostics (synthetic or GT-free is not real scientific validation); `1` corrupt/invalid evidence or verification failure; `2` missing required cache/provenance or CLI usage error. Inspect `status.json` and the command log to distinguish them. Renew `P02_RUN_ID` for each new run; all outputs must be new child directories under the resolved `P02_RUNTIME`.

Artifacts: `actions.json` shared candidates/aliases/actual changes and per-region stats; `summary.json` separate dataset aggregates plus image/region stats and frozen selector outputs; `evaluation.json` pooled metrics, bounded oracle trajectories, damage/repair/per-class counts when GT exists; `status.json` input provenance and limitations; `output_hashes.json`. Read [METHOD.md](METHOD.md) for every frozen rule, including deterministic ties and budgets. The oracle is a bounded reference, never a global optimum or upper bound. A P02 GPU producer, production ledger, and independent production auditor remain future implementation work; this package provides CPU `analyze`/`verify` only.
