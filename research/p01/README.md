# P01 engineering package

This is an opt-in standalone path. Original ProC/SAM3 source is unchanged. New inference defaults off. No training, prompt generation, model download or API inference.

- `METHOD.md`: frozen scientific/engineering contract and ECD-ProC formulas.
- `RUN_GPU.md`: unexecuted real-model stages and dependencies.
- `proc_research/baseline.py`: original direct predictor adapter, two real resolutions, hooks, export and timing.
- `proc_research/cache.py`, `geometry.py`, `protocol.py`: FP32 integrity, coordinate and scene contracts.
- `proc_research/calibration/`: official-estimator wrapper, all 11 methods, immutable stage runner.
- `proc_research/evaluate.py`: only GT-reading module.
- `proc_research/report.py`: paired scene bootstrap, protected-class changes, failures and qualified conclusions.
- `proc_research/prepare.py`, `audit.py`: assets/capacity and actual failing/restored checks.
- `tests/research/`: CPU synthetic contracts and integration. No model substitution.
- `scripts/p01/verify.py`: repeatable acceptance with commands, exit codes and hashes.

Actual logs, environments and tensors belong in `/home/fengyi/CodeHome/Proc/runtime/p01`. Git stores manifests/hashes and code only. No GPU jobs are scheduled by this package. Read PROGRESS.md, BLOCKED.md and run_manifest.json before continuation.

Primary CPU command:

```bash
bash scripts/p01/python.sh -m unittest discover -s tests/research -v
```

Full CPU receipt (use a new output directory):

```bash
bash scripts/p01/python.sh scripts/p01/verify.py --output /home/fengyi/CodeHome/Proc/runtime/p01/audit/cpu-next
```
