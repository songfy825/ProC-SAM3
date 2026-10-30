# Repository handoff: P02

Follow the user's latest task and scope. These notes describe the current P02 checkout and do not add approval gates or authorize unrelated work.

## Start here

- Check `git status --short`, branch, and HEAD before editing. The P02 branch is `codex/p02-mask-action-space` in `songfy825/ProC-SAM3`; preserve unrelated local changes.
- Read `research/p02/REMOTE_RUN.md`, `research/p02/README.md`, `research/p02/METHOD.md`, and `research/p02/input.PROVENANCE.md` for current operation and input contracts.
- Read `research/p02/migration/REPORT.md` and `research/p02/migration/run_manifest.json` for the portable-path implementation, independent clone verification, and publication evidence.
- Read `research/p02/REPORT.md`, `research/p02/PROGRESS.md`, `research/p02/BLOCKED.md`, and `research/p02/run_manifest.json` for historical evidence and limitations. Their old paths, local-only commit statements, and test counts describe that historical run; inspect current Git state and new verification receipts for current status.

## Current implementation

- P02 code: `proc_research/correction/`; CLI: `scripts/p02/`; tests: `tests/p02/`; method and evidence documentation: `research/p02/`.
- Available P02 operations are CPU offline `analyze.py` and `verify.py`. A P02 GPU producer, production ledger, and independent production auditor remain future implementation work. Root P01 GPU guides do not establish those P02 capabilities.
- Existing P01/model files are frozen by the P02 verifier against `ec7f57ef2a3ec5cb166149a2fa85f5eff4a2f566`. Preserve the original tests and assertions; do not weaken checks to pass a migration. A later user-authorized change of scope must explicitly account for this contract.

## Paths and CPU verification

- Use `bash scripts/p02/python.sh ...` for P02 commands, including tests. Do not assume a source-machine username/path or that system Python has torch.
- Python preference: explicit `P02_PYTHON`, active `VIRTUAL_ENV`, `CONDA_PREFIX`, checkout `.venv`, checkout `.p01-env`, then PATH `python3`. Invalid explicit configuration fails.
- Set `P02_RUNTIME` to an absolute path (leading `~` supported); default is `<checkout>/.p02-runtime`. Every output must be a new child directory under the resolved root. The root itself and existing output directories are rejected.
- P02 overrides inherited P01 interpreter/runtime/GPU settings, uses `<P02_RUNTIME>/compat`, and enforces CPU/offline guards. Do not bypass these guards for current analysis or verification.
- Run `bash scripts/p02/python.sh scripts/p02/verify.py --output "$P02_RUNTIME/audit/<new-run-id>"` after relevant changes. Read `verification.json` and logs before claiming success. Clone with history; the verifier needs the frozen base commit.
- `requirements.txt` was exported from Ubuntu x86_64 / Python 3.12.14 with torch 2.7.1+cu126. Fresh installation and target-machine real GPU execution remain unverified; do not imply otherwise.

## Evidence and results

- Real inputs must be existing genuine development/use-history, caches and producer/independent audit records, as specified by `input.PROVENANCE.md`. Missing evidence stays `PENDING`; do not invent provenance or silently substitute samples.
- Keep GT in a separate evaluator label manifest. It must not influence development selection or action construction.
- Preserve historical source records, hashes, reports, and receipts unchanged. Configure new-machine paths in new input manifests where permitted; do not bulk-rewrite archived evidence or relabel synthetic records as real.
- Exit `0` means diagnostics/verification completed, `1` means invalid/corrupt evidence or verification failure, and `2` means missing required evidence or CLI usage error. Confirm semantic status in logs/`status.json`. Synthetic engineering PASS is not real scientific PASS.
- The historical CUDA availability-query deviation remains disclosed in `REPORT.md`; later guarded tests cannot erase it. The bounded oracle is not a global optimum or upper bound.
- Keep data, labels, weights, caches, large logs, environment directories and runtime outputs out of Git. Stage only intended code/documentation changes and inspect the staged diff.
