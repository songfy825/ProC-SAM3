# ECD provenance and parity evidence

Reference: [VILA-Lab/ActiveSAM, commit 25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe](https://github.com/VILA-Lab/ActiveSAM/tree/25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe).

Unmodified official `ecd.py` and `signatures.py` are under proc_research/vendor with MIT license and frozen SHA256. No ActiveSAM model or its precomputed signatures are used by P01.

`test_official_ecd_parity` runs 20 deterministic CPU cases against the official decoder (background enabled/disabled, restricted active IDs, missing calibration class, all-zero pixel, different R_a/R_e). Labels must match exactly. `test_ecd_independent_hand_calculation` uses S=(0.8,0.2), R1=(0.7,0.3), R2=(0.1,0.9), equal zero log priors and sigma=0.5: full discriminants are -0.04 and -1.96. This checks the continuous interface independently of the vendored implementation.

Official signature estimation is reused verbatim. P01 adds per-scene support and row-level fallback around it. Multi-view ECD-ProC is explicitly an adaptation: per-view continuous full squared-distance scores, shared global priors, null competition for each signature, equal-view fusion, agreement/uncalibrated fallback to raw response mean, and raw-domain ProC threshold. It is not named a faithful reproduction of ActiveSAM's full system. METHOD.md gives every formula and the threshold/RMS adaptation.

Additional calibrated CPU cases exercise a changed prediction, disagreement reverting to the mean baseline, null winning, and five-scene support. No model inference occurred. Actual real-model G0 and ECD-ProC effects are PENDING.
