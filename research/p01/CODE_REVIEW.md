# Independent P01 code review — 2026-10-30

Reviewer: separate `/root/p01_review` agent, read-only CPU review. Baseline commit `234c87e58fb08f02ba1cf163b2a73f5330d462e9`; reviewed added research modules, frozen method and tests. No GPU activity. Reviewer independently ran 33 contracts plus the 3 pipeline tests available then.

## Findings and disposition

| Priority | Finding | Disposition and regression evidence |
|---|---|---|
| P1 | Renaming a seal permitted a second confirmation run | Central per-image claim registry under R/runtime/p01; exact run path, seal and config identity required for resume. Reordered/overlapping manifests cannot recycle confirmation images. `test_confirmation_claim_cannot_change_with_seal_filename`, `test_reordered_manifest_cannot_reuse_confirmation_images`: RED→GREEN. |
| P2 | Production config allowed min_scenes=1 | Production config must match frozen METHOD constants; low values remain available only in direct synthetic unit fixtures. `test_production_minimum_scene_support_cannot_be_weakened`: RED→GREEN. |
| P2 | Code digest depended on Git staging and omitted untracked launcher | Explicit production file inventory hashes current contents regardless of staging. Excludes evidence/ledger files. `test_code_identity_tracks_untracked_runtime_shell`: RED→GREEN. |
| P2 | Conditional fallback counts confused view axis with signature axis | Count the union of all views × both signature winners. `test_row_fallback_counts_all_views_and_signature_winners`: RED→GREEN. |
| P2 | Resume disk estimate charged already cached views again | Estimate missing image/view entries only; every actual write still enforces 20GiB reserve. `test_resume_capacity_only_accounts_for_missing_views`: RED→GREEN. |

Additional author checks: non-PASS receipts rejected; confirmation bound to exact sealed bank; G0 resume cannot omit its benchmark; tampered published pool rejected before model construction. These received added regression tests and recorded failing runs before fixes.

## Review focus and limits

- Scene/hash overlap checks are correct, but evidence strings cannot prove their own truth. Real source-scene map review is still required. No actual splits or independence claim were fabricated.
- No confirmed coordinate drift: original logit interpolation→sigmoid→PGRF→aggregation retained, actual input hook and transform tests prepared. Real 504 features and GPU parity remain PENDING.
- Faithful single-view ECD agrees with unmodified pinned official code. Conditioned signatures keep global priors, global uncalibrated flags and null mass. Shuffles mix record/scale association while preserving bucket counts.
- Initial pipeline fixture fell back for every class. Added separate calibrated fixtures now exercise changed class decisions, disagreement fallback, null competition and sufficient distinct-scene support.
- Differing active subsets could lose discriminant precision when averaged with a large finite floor. P01 routing is disabled and production uses all mapped classes; runtime now explicitly rejects differing active vocabularies, with RED→GREEN regression. Empty/zero evidence remains supported. This is an explicit unsupported extension, not a silent numerical approximation.
- Hook cleanup breaks the full-model reference cycle. Original PromptBank retains bounded text-embedding cache across the two dataset vocabularies; actual VRAM measurements remain pending and will include retained tensors.

No unresolved Critical/Important findings remain after the documented fixes. No second reviewer pass is claimed: the fix pass is checked by new failing-then-passing tests and the final entire CPU suite. No baseline model outputs were mocked, and no synthetic result counts as real G0.
