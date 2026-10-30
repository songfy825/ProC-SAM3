# P01 asset transfer

Authorized: package the existing weights, required LoveDA/iSAID data, and metadata; preserve original paths using target-machine symlinks. No GPU work. Existing experiment code/configuration must remain byte-identical.

- [x] Read PROGRESS/BLOCKED/run manifest and identify original paths/size.
- [x] Freeze migration assertions; add standalone packaging and target-link helpers outside production code identity.
- [x] User-directed final scope: copy and verify only the weight; full dataset archive cancelled. Original assets unchanged, >=20 GiB free.
- [x] Read-only SSH comparison: all 26,626 required image/GT files match; target has loveDA spelling and lacks sam3.pt.
- [x] Re-read final metadata archive and verify every member/footer; 15 CPU tests cover path aliases, conflicts, corrupt archives and comparison completeness.
- [x] Record delivery location, bytes/hash, missing inputs and completed independent code review. Commit/push evidence is recorded in the main PROGRESS ledger and runtime delivery receipt.

Design: one gzip tar archive with weights, both existing validation pools and GT, original inference manifests, separate label inventory, released prompt/vocabulary copies, and an unchanged copy of transfer.example.json. Full pool transport is not authorization to infer more than the frozen 600-image cap. No training/test pools needed. Target helper preserves original /home/fengyi paths and /data aliases used by the shipped settings, refuses conflicts, never overwrites existing paths. Runtime and input folders are created in the extracted bundle, outside Git. No scene map or pilot/confirmation labels are fabricated.

Superseding design after user steering: verified weight copy plus a small metadata tar; reuse target datasets. Keep source paths through aliases, including LoveDA→loveDA. Remote remains read-only in this turn; upload/link commands are prepared for later execution. Final instructions: REMOTE_TRANSFER.md. RESTORE.md records the superseded generic full-payload approach.

Acceptance: original hashes must match bytes read into the archive; archive readback must match every recorded member hash/size; link setup must be idempotent and reject conflicts before any mutation; malformed/corrupt packages must fail. Production code hash must remain ca1816ca8606689c2f4077851d89819758b38dead540c6be86807563659eb9a3.
