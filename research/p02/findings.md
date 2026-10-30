# Findings
C HEAD equals pinned commit; source untracked assets recorded outside C. D did not exist.
Logical R resolves to /home/fengyi/E/CodeHome/Proc. P01 cache top-level empty.
CPU Python via existing bash scripts/p01/python.sh and read-only repro interpreter: numpy 1.26.4, torch 2.7.1+cu126. No CUDA calls made in dependency check.
Existing cache SampleCache stores CPU FP32 prompt_scores and class scores on original grid; class_ids tuple; prompt_to_class stores score channel indices. Parent external receipt required.
Existing evaluator maps LoveDA raw 0/255 ignored, 1..7→0..6, iSAID 255 ignored. Pooled confusion metrics exclude absent union.
Historical 60 tests are not current evidence. Some old imports use CUDA availability helpers: guard at Python level before imports; never call hardware query.
No graphify-out graph exists; no graph generation/API/download performed under offline task scope.

Current metadata audit (runtime/p02/audit/data-history.json): original repro LoveDA inference manifest 1669 rows, iSAID 11644, both hashes match historical P01 assets record. Fields are height,id,image,image_sha256,width only: no split, scene or eligibility/use-history. These inventories cannot be relabelled a legal development list. No image/GT/old cache tree scanned. Existing weights were not reread; their old audit hash is history, not a new real-model receipt.

Independent clone: git remote origin is local C, no objects/info/alternates; clone invoked with --no-hardlinks. Source C's HEAD/index/tracked worktree unchanged at end snapshot, but untracked asset-transfer files were added concurrently. Do not describe entire C filesystem as byte-identical; claim only no writes by this task and audited tracked/branch invariants.
