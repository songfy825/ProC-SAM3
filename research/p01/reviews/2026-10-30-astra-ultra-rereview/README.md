# Independent Astra Ultra re-review

Exact reviewer artifacts are preserved unchanged. Read [REVIEW.md](REVIEW.md) for the signed assessment of commit `4013778dc6bd476bfa5182514498747e77c08ffe`: F1/F2 CLOSED, CPU engineering PASS; F3 remains nonblocking P3; real launch/research NOT_READY/PENDING.

`archive.json` hashes the seven copied files. `final-verification.json` also pins the two full logs, retained only under `/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/`. The 31-file evidence check and six new boundary checks are separate from the already completed 70-test CPU suite; do not sum them as a single test run. No GPU was used.

The standalone probe/evidence scripts preserve the exact review-time commands and paths. They are evidence snapshots, not the production acceptance entry. Do not rerun them in place or overwrite their outputs; use a new runtime directory if a future investigation needs them. Full synthetic arrays/labels remain outside Git.
