# Verified causes

- F1: initial launcher capacity budget is reduced to reserve-only after G0 exists. Cache writes enforce reserve, but `decode_stage` never calls a capacity guard before `np.save`; standalone decode bypasses launcher too. The shared decode entry is the appropriate fix boundary.
- F2: decode stores `bank_receipt_sha256`, but report only compares experiment-wide identity fields. Report then reads support/resources from any supplied same-experiment bank. Require the saved exact receipt before reading report inputs or creating output.
- Prior review/evidence hashes match; no need to rescan weights/data or repeat unchanged migration packaging tests. New core code changes justify complete CPU acceptance after targeted regression tests.
