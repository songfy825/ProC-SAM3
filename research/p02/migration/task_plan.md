# P02 portability and branch publication
User authorization: configure machine paths; adjust only machine-path test assumptions; preserve algorithms and safety assertions; after tests commit and push codex/p02-mask-action-space to songfy825/ProC-SAM3. Provide remote Codex guide. No GPU production implementation/execution in this migration task.
Base: cc5942a3759a3783891875cf5de6f63de5053901, clean D independent checkout. Local origin points to C; publish via separate fork remote. C/other repositories read-only.
- [x] Inspect current paths, branch, interpreter and old evidence.
- [x] Write migration regressions and observe RED.
- [x] P02 runtime/interpreter configuration preserving CPU/offline/output containment.
- [x] Docs and Codex handoff guide; no historical receipt rewriting.
- [x] Original CPU regression + new tests, relocated clean snapshot and safety negative cases; independent review.
- [x] Local commit, explicit branch push, remote SHA verification; deliver clone/run commands.
Design: P02_RUNTIME absolute or expanded tilde; default checkout/.p02-runtime; ignore inherited P01_RUNTIME. P02_PYTHON explicit, then active virtualenv/conda, local .venv/.p01-env, then PATH python3; always override P01_PYTHON. All safety knobs remain forced. Artifacts of this task remain R/runtime/p02/migration. No installs/downloads/inference/GPU queries.
