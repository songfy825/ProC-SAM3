# P01 decision — CPU engineering delivery

Research status: **PENDING / 证据不足**. No real P01 model inference, G0, pilot or confirmation has been executed in this round. There is no measured accuracy gain, loss, online latency or VRAM claim. The research hypothesis is neither supported nor refuted by CPU synthetic tests.

Engineering status: **CPU implementation and review PASS; local commit being finalized. 50 CPU tests passed with no skips.** The branch contains the complete standalone implementation, all specified controls, a frozen method contract, synthetic contract/integration tests, actual negative audits, and an independently reviewed fix pass. See PROGRESS.md and the final verification receipt for completion status.

Scientific blockers for the later phase:

1. New user authorization is required before any GPU work; this round explicitly forbids it. No queued job or automatic availability polling exists.
2. Real 1008 direct/export/repeat parity (exact labels, score error <=1e-6) and actual 504 input/features are untested.
3. LoveDA source-scene provenance is missing from existing manifests. iSAID parent tile names alone do not prove geographic independence. A reviewed source-scene map is needed before generating real isolated splits.
4. Released prompt-pool construction images/splits/revision are unknown. No leakage-free construction claim is made.

Measured engineering evidence and resource facts are in research/p01/evidence: source clean at 742ce170…, clone based at 234c87e5…, weight 3,450,062,241 bytes SHA256 9999e234…cf8c9e, 13,313 image hashes matched their existing inventories. Read-only Python 3.12 / torch 2.7.1+cu126 ran CPU checks; no packages installed. Actual CUDA initialization was false in the CPU preflight. Prompt expansion produced 28 LoveDA and 24 iSAID prompts from the frozen released pools.

The independent review and fixes are recorded in research/p01/CODE_REVIEW.md. The final CPU receipt records test count, exit codes, exact input/config/code/METHOD hashes and log locations. No artifact outside this project was intentionally modified; model weights, original data, old environment and source checkout were used read-only. No push or merge is authorized or performed.
