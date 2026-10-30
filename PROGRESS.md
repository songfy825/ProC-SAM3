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
