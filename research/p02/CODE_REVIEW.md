# P02 independent implementation review

Reviewed 2026-10-30 against `METHOD.md` and `task_plan.md`. Scope: new correction modules, P02 entrypoints/tests, and existing acceptance-v1 logs. Read-only review except this report; no broad tests rerun, model execution, GPU probe, API, download, source-checkout modification, commit or push.

## Findings sent to the implementer

1. **P2 — Record controlled failure for supplied corrupt GT.** At review time, `scripts/p02/analyze.py:73` called `evaluate_dataset` outside the error handler at lines 32–51. A supplied GT hash mismatch, duplicate label key, or invalid label geometry raises after `actions.json` and `summary.json` are written, leaving no terminal `status.json`. The README promises inspection of that status to distinguish missing inputs from corrupt evidence. Extend controlled exception handling through the independent evaluation stage, retain serialized GT-free actions, and write `status=FAIL` with the evidence error before exiting 1. Add an actual CLI corrupted-GT assertion.

2. **P2 — Negative CLI verification must check the reason for failure.** At review time, `scripts/p02/verify.py:35` accepted every expected exit code without validating the result artifact. This already produced a false individual pass in acceptance-v1: `synthetic-corrupt-receipt.log` contains `TypeError: load_manifest() got an unexpected keyword argument 'stream'`, yet the command record passed because both intended rejection and unrelated crash exit 1. Require the expected `status.json` status and diagnostic reason for corrupt receipt/missing cache cases, and reject tracebacks. Retain the final fresh complete acceptance run after code stops changing.

## Reviewed behavior

- Margin selection uses the geometric original grid, stable row-major ties, 8-connectivity, decreasing-area component selection and four-region cap. No GT is read by action construction.
- Candidate ranking uses float64 region means and class/prompt ties. Double KEEP is removed before joint output-pair deduplication and the three-candidate cutoff; both arms use shared IDs. Effective output comparisons include all opposite-arm outputs and implicit KEEP.
- Majority uses the original prediction class share strictly above one half and retains the same candidate ID in both arms even when its MASK is a no-op.
- Dataset summary rates pool numerators and denominators. Evaluation uses existing full-image label mapping/confusion metrics and keeps datasets separate.
- Oracle starts all KEEP, sorts images, evaluates replacement image confusion in the current dataset total, retains the current choice on ties, and accepts strict pooled-mIoU improvements for at most three sweeps and 256 combinations per image. Chosen predictions are reconstructed for final effect reporting. Tests include a global-objective counterexample and an exhaustive 256-choice example.
- Missing any selected GT suppresses that dataset's oracle rather than cherry-picking. Synthetic outputs and absent real inputs retain PENDING real-evidence status; geometry novelty is not reported as effectiveness.
- Provenance checks cross-record bindings, producer completion, independent audit record location, selected exports, selected image bytes, published prompt/class metadata, and code/config/weight identity. Changing a top-level flag alone is insufficient. Authenticity of an external audit is an explicit trust assumption; consistent self-written records cannot prove historical inference.

## Verification limits

The follow-up adds a complete synthetic record graph for `_real_evidence` and tampered binding, missing selected export, and changed code-inventory cases. The graph is explicitly synthetic, and the normal loader retains PENDING real-evidence status. No genuine real cache was available. These tests exercise structural evidence validation; this review does not establish real admission, authentic historical inference, or real scientific results.

Acceptance-v1 ended FAIL because implementation changed during execution; its negative corrupt-receipt case cannot establish that rejection. Findings above describe the original reviewed versions and are resolved as recorded below.

## Resolution

Both P2 findings are resolved on scoped rereview. `analyze.py` now catches independent-evaluation evidence errors, records FAIL and hashes while retaining GT-free actions. `verify.py` requires the expected terminal status plus receipt/cache-specific reason, so the previous unrelated TypeError cannot pass that negative contract. `tests/p02/test_cli.py` covers corrupted GT and absent/unrelated/wrong status results. Acceptance-v2 completed PASS: all 60 unchanged P01 tests and 62 P02 tests, all CLI/status contracts, frozen-original checks, and GT-action byte invariance passed. Acceptance-v3 is a final repeat for the subsequent test-only strengthening below; it was still in progress at this review's close.

The new published-vocabulary helper directly loads the pure `core/prompt_bank.py` definitions and avoids the `core.__init__`/SAM3 import chain. The regression-strength concern is resolved: the CUDA-denial test now explicitly clears `_published_inventory`'s LRU cache before its guarded call, so an earlier test cannot bypass the path under test. No remaining actionable implementation defect was identified in the scoped rereview.

The implementer disclosed an earlier execution deviation in `progress.md`: an unguarded legacy-wrapper provenance test imported the old SAM3/autocast chain and queried `torch.cuda.is_available()`. The query cannot be undone. The current correction and later guarded tests must not be represented as proof that the entire session satisfied the historical zero-GPU-query requirement. No model inference or GPU compute is established by that incident; functional engineering evidence and the historical boundary deviation must remain separately stated.

Final review disposition: implementation complete with the disclosed historical execution deviation; `ENGINEERING=COMPLETE_WITH_DEVIATION`, `IMPLEMENTATION=COMPLETE`, and `REAL_ACTION_SPACE=PENDING` accurately separate these conclusions. This is not an unqualified historical-compliance PASS or evidence of real effectiveness.

Final implementer evidence: acceptance-v3/verification.json status PASS; 60 P01 + 62 P02 tests, no skips, expected CLI states/reasons and GT invariance all pass. Source hashes rechecked against delivered files. Historical CUDA-query deviation remains unresolved as a past event, fully disclosed; it is not converted into compliance by these results.
