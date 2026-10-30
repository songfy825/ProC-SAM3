# P01 decision — CPU engineering delivery

Research status: **PENDING / 证据不足**. No real P01 model inference, G0, pilot or confirmation has been executed in this round. There is no measured accuracy gain, loss, online latency or VRAM claim. The research hypothesis is neither supported nor refuted by CPU synthetic tests.

Engineering status: **本轮工程目标 COMPLETE. CPU implementation, 60 tests (no skips), independent review and transfer preparation PASS.** The original 50-test delivery is preserved; the transfer follow-up adds 10 tests, portable paths, a stage launcher, pinned requirements and explicit recovery. All specified controls, frozen method contracts, negative audits and reviewer fixes remain in the branch. See PROGRESS.md and `research/p01/evidence/transfer-verification.json` for final acceptance.

Scientific blockers for the later phase:

1. New user authorization is required before any GPU work; this round explicitly forbids it. No queued job or automatic availability polling exists.
2. Real 1008 direct/export/repeat parity (exact labels, score error <=1e-6) and actual 504 input/features are untested.
3. LoveDA source-scene provenance is missing from existing manifests. iSAID parent tile names alone do not prove geographic independence. A reviewed source-scene map is needed before generating real isolated splits.
4. Released prompt-pool construction images/splits/revision are unknown. No leakage-free construction claim is made.

Measured engineering evidence and resource facts are in research/p01/evidence: source clean at 742ce170…, clone based at 234c87e5…, weight 3,450,062,241 bytes SHA256 9999e234…cf8c9e, 13,313 image hashes matched their existing inventories. Read-only Python 3.12 / torch 2.7.1+cu126 ran CPU checks; no packages installed. Actual CUDA initialization was false in the CPU preflight. Prompt expansion produced 28 LoveDA and 24 iSAID prompts from the frozen released pools.

The independent reviews and fixes are recorded in research/p01/CODE_REVIEW.md. The final CPU receipt records test count, exit codes, exact input/config/code/METHOD hashes and log locations. No artifact outside this project was intentionally modified; model weights, original data, old environment and source checkout were used read-only. The user subsequently authorized pushing this branch to their fork `songfy825/ProC-SAM3`; no merge is requested.

Implementation commit: `e28c65b15321008c6964fc84bdabec4355b7c8ae`. Final delivery HEAD may additionally contain the documentation-only receipt commit. Verified implementation code hash is in research/p01/evidence/verification.json.

Transfer decision: ready to install dependencies and prepare inputs on Ubuntu x86_64 using TRANSFER_4090.md. This is conditional execution readiness, not proof that the experiment only needs spare GPU memory. Actual scene provenance, fresh-environment installation and GPU G0 still need validation. Requirements and code hashes are recorded in the run manifest; no binary environment archive is needed under the revised user request.

Transfer implementation commit: `0ae88f2410f7784f620390fa1e085832c4acf769`; subsequent receipt commit changes only the ledger.

Asset decision: **reuse target datasets, transfer only the large SAM3 weight plus small metadata/helpers**. Read-only full validation-pool comparison found 26,626/26,626 image/GT files identical between this machine and 192.168.0.18. Target LoveDA spelling is `loveDA`; prepared aliases preserve original paths and existing settings. Weight copy and metadata archive are locally verified. Remote upload/alias creation were not performed. This proves byte equality, not scene independence or model accuracy; research remains PENDING.
