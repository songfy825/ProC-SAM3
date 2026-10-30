# P01 P2 repair plan — 2026-10-30

Scope: repair Astra Ultra F1/F2, CPU regressions, Astra Ultra re-review, then push to the user's fork only after PASS. No GPU activity. Baseline `e8e07cb7859fc4130d2ffddcb9c386bdd558721d`; original review targets `ba663c4`. F3 is a recorded nonblocking P3, outside these two repairs. METHOD, model precision, datasets and experimental policies remain frozen.

- [x] Read progress/blockers/run manifest; verify archived review and six evidence hashes; inspect actual write/report paths.
- [x] Freeze meaningful regression assertions and run RED against the unfixed implementation: 10 tests, 9 expected failing assertions/subcases, exit 1, no errors; evidence in runtime/p01/audit/p2-fixes-v1.
- [x] F1: budget all prediction output before starting decode and recheck each file before writing on the output filesystem; retain failed CPU evidence and reuse upstream cache. Five targeted CPU tests PASS, exit 0.
- [x] F2: require exact prediction bank receipt, valid bank stage/kind, and confirmation frozen-bank binding before report output. Combined 10-test GREEN passed, unchanged test hash, exit 0.
- [x] Targeted GREEN and full acceptance `cpu-p2-v1`: 70 tests, no skips, all negative/restored checks and six CLI helps PASS, overall exit 0. RED evidence and frozen assertions retained.
- [x] Independent `gpt-6-astra` / `ultra` re-review PASS on `4013778`: F1/F2 CLOSED, six additional CPU boundary probes PASS; F3 stays nonblocking P3, no new blocking findings.
- [x] Update PROGRESS/BLOCKED/decision/run manifest and hash catalogs; implementation committed as `4013778`, final evidence receipt commit follows.
- [ ] Push only to `fork` after review PASS and verify remote SHA; publication receipt is recorded outside Git at runtime/p01/audit/p2-fixes-v1/delivery.json.

Acceptance: low total space and later space loss reject before the affected predictions; sufficient space produces identical outputs; fresh CPU retry retains previous evidence and reuses upstream cache. Report A/A succeeds; A/B, missing binding, invalid bank stage/kind and confirmation seal mismatch fail before creating a report. No research/GPU PASS inferred.
