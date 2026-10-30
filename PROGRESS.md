# P01 engineering ledger

Scope: 2026-10-30 user revision. CPU-only engineering; no GPU inference/tests, allocation, queues or waiting. GPU G0, pilot, confirmation and scientific conclusions PENDING.

## Resume
Read this file, BLOCKED.md, research/p01/run_manifest.json and evidence hashes before resuming. Do not repeat verified unchanged phases. All code belongs to this independent clone, branch codex/p01-scale-calibration, base 234c87e58fb08f02ba1cf163b2a73f5330d462e9. Source repo HEAD 742ce170e1686b51aab06f2afd22c93812cb06cb is clean and read-only. No source inference result counts as G0.

## Plan
- [x] Read historical plans and external progress; verify source commit, independent clone, weight and read-only CPU environment.
- [x] Freeze METHOD and acceptance tests before implementation.
- [x] Implement cache, geometry, provenance, data isolation and evaluation.
- [x] Implement official ECD parity and ECD-ProC multi-view controls.
- [x] Implement gated real-model adapter, G0, pilot/confirmation run entrypoints and RUN_GPU.md.
- [x] Run CPU contracts, deliberate negative audits, code review and corrective checks.
- [x] Commit locally; record commit, evidence and pending research gates.

## Findings / rulings
- User's explicit clone/scope instructions supersede historical plan-only text and worktree workflow. No new design approval required.
- Reuse old interpreter through ELF loader, read-only; torch 2.7.1+cu126, NumPy 1.26.4, Pillow 11.3.0. pytest absent; unittest is the CPU test runner (no weaker assertions).
- Configs request nonexistent *_pgrf_max* pools; use exact existing published pools and record hashes. No automatic prompt generation.
- LoveDA filenames do not prove original scene identity. Require evidence-backed scene map before real splits. iSAID tile prefixes identify parent images, not geographic independence.
- External Git fetch required approved network escalation after sandbox connection failure; fetched source only, no weights. ActiveSAM pinned 25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe.
- R physical path is /home/fengyi/E/CodeHome/Proc; logical path /home/fengyi/CodeHome/Proc.

### Acceptance freeze
METHOD.md and 26 CPU contracts written before implementation. SHA256 frozen in research/p01/evidence/contract-freeze.sha256. RED command: bash scripts/p01/python.sh -m unittest discover -s tests/research -p test_contracts.py -v, exit 1: 26 tests, one missing-module-file assertion and 25 missing-module errors. Raw log runtime/p01/audit/contracts-red.log. Additional tests may strengthen coverage; frozen tests/thresholds will not be removed or relaxed.

### Core implementation / CPU evidence
- 33 CPU contracts passed (runtime/p01/audit/contracts-stage2.log, exit 0, 28.595 s). Includes faithful single-view official oracle, hand discriminant example, shared priors, scene support and all three actual negative subprocess audits.
- Original model source files are untouched. Direct runner adapts only unused MMSeg base/registry imports and two asset paths in memory; method-body AST equality is tested. Existing environment lacks mmcv; torch, SAM3 and mmengine data carriers import through read-only interpreter. SAM3 import emits upstream CUDA-autocast-disabled warning with CUDA hidden; no model is constructed by CPU tests.
- Complete initial image inventory verification: LoveDA 1,669 and iSAID 11,644 image SHA256 values match external manifest. No GT bytes read; source-scene provenance still unverified.
- Added CPU-only synthetic pipeline test for fitting all controls, prediction output, separate evaluation and report. Its output must state PENDING_SYNTHETIC_ONLY. Never a G0 substitute.
- Independent review in progress, permitted by requesting-code-review skill; reviewer uses CPU checks only.

### Independent review / final CPU acceptance
- Independent reviewer completed read-only review and independently ran 33 contracts plus 3 integration tests. Five actionable issues were fixed with RED→GREEN tests: confirmation claim reuse, weakened scene support, staging-dependent code hash, cross-axis fallback counting, and resume disk overestimation. See research/p01/CODE_REVIEW.md.
- Additional fixes bind the sealed bank, require the G0 benchmark on resume, reject non-PASS receipts and altered released pools, and reject mismatched active vocabularies (routing is disabled).
- Final command: `bash scripts/p01/python.sh scripts/p01/verify.py --output /home/fengyi/CodeHome/Proc/runtime/p01/audit/cpu-final-v1`, exit 0. **50 tests passed**, no skipped tests. Three corrupted-input subprocesses each exited 1; restored inputs each exited 0. All five CLI help entries, original-source diff, frozen contract SHA256 and whitespace checks exited 0.
- Evidence: research/p01/evidence/verification.json, negative.json, log_catalog.json; full logs remain in R/runtime/p01/audit. All prior RED logs retained. Frozen METHOD/original 26 test assertions unchanged.
- CPU preflight: CUDA initialized=false, source status clean at 742ce170…, current weight hash equals initial, 81 environment package versions recorded. 166,061,670,400 free bytes at that audit (about 154.7 GiB). No dependency installation or external environment modification.
- Real GPU stages/online latency/VRAM/accuracy remain PENDING. There is no queued work, no automatic retry/availability polling. Actual source-scene map and prompt-pool construction provenance remain research limitations.
- RUN_GPU.md commands are prepared but have not been executed. No old project result was copied into G0.

### Local delivery
Implementation commit: `e28c65b15321008c6964fc84bdabec4355b7c8ae` on `codex/p01-scale-calibration`. Final verified code SHA256: `5f8c193e655531cc00e1898878848f3a86f9a94dac35d0b128478756e359e166`. A subsequent documentation-only receipt commit records this hash; no production code changed after the 50-test acceptance. No push or merge performed. This round engineering goal COMPLETE; full P01/GPU/research acceptance PENDING.

## 4090 transfer follow-up (user authorized branch push)
- Rechecked prior ledger and hashes; prior CPU delivery remains recorded, no GPU results inferred.
- Original remote is the read-only local source checkout. GitHub connector lacks access to the user's fork; the user supplied SSH URL `git@github.com:songfy825/ProC-SAM3.git`. Read-only SSH `ls-remote` succeeded (HEAD 742ce170…, target branch absent). Push only to a separate fork remote.
- Delivered portable paths/interpreter selection, pinned requirements and optional Conda metadata, CPU preflight and explicit stage launcher. User narrowed environment delivery to package names/requirements; binary archives are not part of delivery. No GPU activity this session.
- Real scene-map provenance and GPU G0 remain prerequisites; do not describe the current checkout as waiting only for spare GPU memory.

- User specified SSH fork git@github.com:songfy825/ProC-SAM3.git and Ubuntu 64-bit target; narrowed environment delivery to requirements.txt (no binary environment transfer required). Preserve exploratory pack artifacts locally, but do not upload or require them.
- Initial transfer tests correctly failed before implementation. First implementation run also exposed the active base environment lacking torch and the project filesystem lacking symlinks. Use explicit existing read-only P01 Python; the packaging-link test uses `/tmp`, matching the disposable staging location, with unchanged assertions. No source environment was modified.
- Existing environment `pip check` exited 0; requirements pins derive from its package metadata. No clean-room Conda/pip installation on Ubuntu or GPU validation is claimed.
- Added portable checkpoint/runtime/manifest paths, `scripts/p01/launch.sh`, staged G0/pilot/confirmation plans, and `TRANSFER_4090.md`. Confirmation remains explicit and all GPU commands remain unexecuted. Complete receipts are verified before reuse; changed inputs fail closed.
- Transfer code review found and fixed stale evaluation reuse, missing stage/dependency validation, and CPU-stage recovery. Four added regression tests retain their actual RED logs. GT hash verification runs only in the independent evaluator process. Final reviewer spot-check has no remaining important findings.
- CPU preflight `preflight-transfer-v1.json` exited 0: weight hash unchanged, CUDA initialized=false; existing dataset-manifest hashes match prior inventory. Unchanged 13,313 image bytes were not redundantly rescanned. Source S remains clean at 742ce170…. Pilot launcher dry-run exited 0 without inference.
- Final transfer acceptance: `P01_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python P01_RUNTIME=/home/fengyi/CodeHome/Proc/runtime/p01 bash scripts/p01/python.sh scripts/p01/verify.py --output /home/fengyi/CodeHome/Proc/runtime/p01/audit/cpu-transfer-v2`, exit 0. **60 tests passed, no skips** (41.947 s for the suite); all three corrupted subprocess checks exit 1 and restored checks exit 0, six CLI help entries pass, frozen hashes/original source/whitespace pass. `cpu-transfer-v1` is retained as intermediate evidence before the evaluator resume fix; only v2 is final acceptance.
- Final transfer code SHA256 `ca1816ca8606689c2f4077851d89819758b38dead540c6be86807563659eb9a3`; requirements SHA256 `027da4108d4aceaf6b4e25ef95499cbe2abcf274b5d7e6ee4d837b5a3c1fd5cb`. Receipts/log catalog: `research/p01/evidence/transfer-*`. Full logs remain under R/runtime/p01/audit. No large assets or environment binaries are included in Git.

Transfer implementation commit: `0ae88f2410f7784f620390fa1e085832c4acf769`. A documentation-only receipt commit records it; verified production code remains unchanged. Push target: `git@github.com:songfy825/ProC-SAM3.git`, branch `codex/p01-scale-calibration`; local source origin remains read-only.

## Asset delivery / target comparison

- User requested packaging weights/data with compatible paths, then authorized read-only SSH comparison to `fengyi@192.168.0.18` and narrowed large-file transport to SAM3 if data match. Credentials are not stored in files.
- Both machines' required LoveDA/iSAID validation image and GT bytes were fully hashed: 26,626 files / 30,509,113,565 bytes, exact match, zero missing/extra/mismatched files. Remote LoveDA directory is `loveDA`; existing-target alias helper handles that spelling without changing model code/config. Evidence: `research/p01/asset-transfer/remote-comparison.json`.
- Final local delivery: `/home/fengyi/CodeHome/Proc/runtime/p01/transfer/weight-only-v1/`. Verified `sam3.pt` (3.21 GiB) + `p01-metadata-v2.tar.gz` (4,071,005 bytes) + SHA256SUMS + instructions. No dataset images/GT images in final delivery; original files retained. Remote weight path was absent. No remote writes performed.
- Migration helpers have 15 passing CPU tests and independent review/fix/spot-check. Source model code/config, requirements and production digest remain unchanged from ec7f57e; previous 60-test production acceptance is retained, not rerun. All actual commands, RED/GREEN logs and interrupted-full-package evidence are recorded in the asset-transfer ledger/runtime audit.
- Actual scene map, final isolated splits, target dependency install and GPU G0/pilot/confirmation remain PENDING. Instructions and current transfer scope: `research/p01/asset-transfer/REMOTE_TRANSFER.md`.

## Astra Ultra strict review — 2026-10-30

- User explicitly requested an independent Astra Ultra review. Dispatched `gpt-6-astra` with `reasoning_effort=ultra`, reviewing `ba663c4bfe65c047932aaf0271d6ca27503b4c50`; no model inference, GPU activity, remote writes or implementation changes.
- Parent verified 40/40 historical receipt/log/test-source hashes (exit 0). The reviewer inspected method, code, tests and runbooks; existing 60/15 suites and large-asset scans were not repeated.
- Targeted CPU probes (exit 0, CUDA initialized=false) confirmed F1: decode wrote 44 synthetic predictions while the mocked free space was below the 20 GiB reserve; F2: report accepted bank B for predictions bound to bank A. A/B were deterministic fits to the same synthetic cache, so this proves missing receipt binding, not changed scientific metrics. F3: standalone config accepted a reduced dataset set (P3).
- Missing scene map and missing GPU authorization were correctly rejected (exit 2 and exit 1). Runtime arithmetic was independently reproduced; new-hardware timings remain hypothetical and hour ranges are planning buffers.
- Verdict: method CONDITIONAL_PASS; strict CPU engineering REQUEST_CHANGES for the two P2 gaps; real-run readiness NOT_READY; full research PENDING. No confirmed P0/P1. This supersedes the earlier unconditional engineering status without erasing earlier passing evidence.
- Exact report and small evidence copies are archived in `research/p01/reviews/2026-10-30-astra-ultra/`; full synthetic artifacts/logs remain in `R/runtime/p01/audit/astra-ultra-review-v1/`. Report SHA256 `bbf4370c1e900595f0f7a0b3e44126ccaf0417369278968feb0f1f4aa688c513`. Findings are recorded, not repaired, in this review-only follow-up.

## P2 repair follow-up — in progress

- User authorized fixing F1/F2, CPU regression tests and another Astra Ultra strict review; push only after that review passes. GPU remains forbidden.
- Resumed clean at `e8e07cb`; archived review and six evidence hashes match. Actual root causes confirmed at decode writes and report bank selection. Detailed acceptance plan: `research/p01/p2-fixes/task_plan.md`.
- Added and froze 10 CPU regression tests before implementation. RED exited 1: 10 tests, 9 expected assertion failures (including two subcases), no setup errors. Test SHA256 `33a36d9a6365d83114d81560ab6168cd57e1b200554af420749b45dabb5884f4`; logs/command/exit code in `R/runtime/p01/audit/p2-fixes-v1/red.*`.
- F1 implementation now budgets every original-grid int64 prediction plus file/metadata headroom before creating a decode run, and rechecks remaining writes on the destination filesystem before each NPY and final metadata. Targeted verification in progress; F2 not yet changed.
- F1 targeted GREEN: 5 tests passed, exit 0; insufficient total/header capacity and space loss stop writes, sufficient capacity preserves identical NPY bytes, explicit CPU retry retains failed output and reuses the same cache receipt. Evidence `p2-fixes-v1/f1-green.*`.
- F2 now rejects a report bank with the wrong stage/kind, a receipt different from the prediction's recorded bank, or a confirmation frozen-bank mismatch, before reading evaluation results or creating report output. Combined regression verification in progress.
- Combined GREEN: all 10 new CPU regression tests passed, exit 0 (15.432 s suite); test SHA matches the pre-implementation freeze. Full CPU acceptance is now running under new run ID `cpu-p2-v1`. No model, method, precision, original tests or config changed.
- Full acceptance `cpu-p2-v1`: 70 tests passed, no skips (58.553 s suite), overall exit 0; all three corrupt-input subprocesses exited 1 and restored inputs exited 0, six CLI helps/frozen contract/original model diff/whitespace passed. Code SHA256 `df52a7636ed6fb0de5a613ceea513c68632fdcca7f61c8d415ce900d45caab59`; verification SHA256 `0e0b19a7b107e0e969d8145690e76d490dc857ca3a4af030cd66056239ae7098`. Evidence copied to `research/p01/p2-fixes/evidence/`. Awaiting Astra Ultra re-review; no push yet.
- Local implementation commit `4013778dc6bd476bfa5182514498747e77c08ffe` was frozen for independent re-review. Post-commit check verified all six previous test files unchanged, all seven current test files and 11 acceptance logs matching hashes, and only the two intended production files changed. External source checkout remains clean.
- Astra Ultra (`gpt-6-astra`, ultra) re-review **PASS**: F1/F2 CLOSED, no new blocking issues; F3 retained as nonblocking P3. Reviewer independently verified 31 evidence hashes and the 364-file production digest, then ran six additional CPU boundary probes (exit 0, CUDA uninitialized), including within-image space loss, final-metadata space loss, heterogeneous original grids and early bank/seal rejection.
- Exact review and small evidence archived in `research/p01/reviews/2026-10-30-astra-ultra-rereview/`; report SHA256 `645d1a6f63c043eb0b992d1b540706a61a1452a6b80475296623dbaf9bb4ab17`. Full synthetic artifacts/logs remain under `R/runtime/p01/audit/astra-ultra-rereview-v1/`. Final receipt commit changes documentation/evidence only; verified production digest is unchanged. Push is authorized after PASS; command/remote verification receipt is stored at `R/runtime/p01/audit/p2-fixes-v1/delivery.json` once published. No GPU execution, allocation, polling or queues.
