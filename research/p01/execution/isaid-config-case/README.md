# iSAID configuration filename repair — 2026-11-02

`model_options('iSAID')` requested `configs/cfg_isaid.py`, while Git tracks
`configs/cfg_iSAID.py`. The source checkout's case-insensitive filesystem hid the
failure that a case-sensitive Linux checkout would encounter. The production
change corrects only that filename in `proc_research/baseline.py`.

The new CPU regression copies the released configurations, class files and
prompt pools into case-sensitive `/tmp`, relocates the adapter's asset root, and
calls the real configuration loader and prompt inventory. It does not construct
SAM3 or replace model outputs. It checks iSAID's 16 classes / 24 prompts and
LoveDA's 7 classes / 28 prompts, plus the frozen options used by the adapter.

- RED: two tests ran; iSAID failed on the missing lower-case filename, LoveDA passed.
- GREEN: the identical test file passed both tests after the one-line fix.
- Full P01 CPU acceptance: **72 tests passed, zero skipped**; all 11 acceptance
  commands exited zero, including negative audits, frozen contracts, original
  model comparison and CLI checks.
- Independent review found no blocking issue. Existing tests, model files,
  configurations, prompt pools, METHOD and the acceptance runner were not changed.

[Evidence](evidence.json), [RED output](red.txt), [GREEN output](green.txt), and
[full acceptance receipt](verification.json) preserve commands, exit codes and
hashes. The full logs remain at the source-machine runtime paths in the receipt.
The retained upstream CUDA-autocast warning occurs with CUDA hidden during CPU
imports; these runs did not execute model inference or a real GPU test.

The user confirmed the target checkpoint at `/home/fengyi/sam3.pt`; the transfer
settings template now uses that path. This task did not verify its remote bytes.
Target preflight must still check the frozen size/hash before G0. No weight upload
or alias creation is needed based on the superseded missing-file record.

This fix changes `code_sha256`. Use the updated branch for a new target run;
do not attach old-code GPU receipts to this implementation. GPU authorization,
the separate G0 gate, the ban on confirmation, and P02 cache preservation remain
as specified in [the execution Goal](../GOAL_4090.md).
