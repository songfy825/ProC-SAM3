# Deferred GPU runbook — NOT EXECUTED

**当前执行入口（2026-11-02）：[4090 最终 Goal](research/p01/execution/GOAL_4090.md)。用户已授权目标 4090 的 G0 及通过后的 calibration/pilot；G0 必须单独执行并逐项验证。本轮禁止 confirmation，并保留 P02 缓存。** 以下未授权描述是此前阶段的记录；本轮范围以 Goal 为准。本次仅发布任务书，没有新增 GPU 实验结果。

Every GPU command below is **unverified on a real model**. This round did not load SAM3 weights into a model, run GPU inference/tests, reserve VRAM, queue work, or wait for GPU availability. Execute only after a new user instruction. GPU G0, actual 504 feature behavior, online resources, pilot and confirmation remain PENDING. CPU tests certify software contracts, not model accuracy.

**Ubuntu / 4090 transfer:** start with [TRANSFER_4090.md](TRANSFER_4090.md). It supplies the portable requirements and stage launcher. The expanded commands below document each stage and its acceptance criteria; their original-host paths are examples, not target-machine defaults.

## Dependencies and independent scene evidence

Work on `codex/p01-scale-calibration`. `bash scripts/p01/python.sh` uses explicit `P01_PYTHON`, an active Conda environment, or a local `.p01-env`; the original host also supports its read-only interpreter through the ELF loader. Set `P01_RUNTIME` to the persistent experiment root. Actual source package versions are in `research/p01/environment-lock.txt`; installable pins are in `requirements.txt`. Required imports include torch, torchvision, NumPy, Pillow, einops, iopath, timm and the checkout's SAM3/mmengine. No dependency installation or external environment repair occurred. CUDA driver must work when separately authorized; no automatic polling or queue is provided.

Assets: `/home/fengyi/sam3.pt` (SHA256 in evidence), local `sam3/assets/bpe_simple_vocab_16e6.txt.gz`, shipped LoveDA/iSAID `.pkl` pools. The adapter replaces only MMSeg base/registry wiring and two asset paths in memory; all original algorithm methods remain unchanged. No Qwen or prompt-generation entry is invoked. Missing or non-expanding published pool is an error.

Before resuming, read PROGRESS.md, BLOCKED.md, research/p01/run_manifest.json and the desired run's manifest/complete.json. Verify every referenced hash. Completed unchanged stages are reused. `--resume` is only for an incomplete GPU run with an identical identity; complete runs verify and return without inference. Orphaned unindexed cache directories cause an error and are retained; use a new run ID after diagnosis, without deleting evidence.

All commands below assume:

```bash
cd /home/fengyi/CodeHome/Proc/src/ProC-SAM3
P01_R=/home/fengyi/CodeHome/Proc/runtime/p01
export P01_RUNTIME="$P01_R"
export P01_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python
mkdir -p "$P01_R/inputs" "$P01_R/runs"
```

CPU preparation (safe this round, but unique output filenames are required):

```bash
bash scripts/p01/python.sh -m proc_research.prepare preflight --output "$P01_R/preflight-new.json"
bash scripts/p01/python.sh -m proc_research.prepare inventory --output "$P01_R/inputs/inventory.jsonl"
```

Supply a **reviewed source-scene map**, with every selected inventory key `dataset/image_id` mapped to exactly `source_scene_id` and `scene_evidence`. Evidence must identify the actual source metadata/document, not an invented independence assertion. Example schema (NOT a valid real map):

```json
{"LoveDA/2522.png":{"source_scene_id":"REQUIRES_SOURCE_EVIDENCE","scene_evidence":"REQUIRES_SOURCE_EVIDENCE"}}
```

LoveDA numeric filenames do not establish parent scene IDs. iSAID tile prefixes can identify parent images but do not establish geographic independence. The software validates supplied IDs, image hashes and nonempty evidence fields; it cannot verify the truth of a provenance assertion. Actual map review remains required. Do not use the example placeholders.

```bash
bash scripts/p01/python.sh -m proc_research.calibration.run split \
  --inventory "$P01_R/inputs/inventory.jsonl" \
  --scene-map "$P01_R/inputs/reviewed-scene-map.json" \
  --limits '{"calibration":50,"pilot":50,"confirmation":80}' \
  --output "$P01_R/inputs/splits-v1.jsonl"
bash scripts/p01/python.sh -m proc_research.prepare capacity \
  --manifest "$P01_R/inputs/splits-v1.jsonl" --runtime "$P01_R" \
  --output "$P01_R/capacity-v1.json"
```

Counts are maxima per dataset, not a significance guarantee. The 80-image confirmation example is below the authorized 200 cap because storing all FP32 responses and all predictions for 600 images can exceed this disk. Choose final counts from this **GT-free** capacity estimate, leave >=20GiB free, then freeze the manifest before inference. Do not silently reduce precision, delete other files or remove methods. All three splits must be disjoint in original scenes and image SHA256. Inference manifests may contain only the documented fields; GT is rejected.

## Future explicit enablement and G0

Only after new user authorization, copy the tracked config to an immutable runtime input and set its sole `enabled` field to `true`. Other constants must match METHOD v1. Record the config hash. For example:

```bash
bash scripts/p01/python.sh -c 'import json; from pathlib import Path; p=Path("/home/fengyi/CodeHome/Proc/runtime/p01/inputs/config-v1.json"); c=json.loads(Path("research/p01/config.json").read_text()); c["enabled"]=True; p.open("x").write(json.dumps(c,sort_keys=True,indent=2)+"\n")'
```

The two explicit switches below do not themselves grant authorization. They are deliberate invocation guards. **These commands have not been run.** They are serial and contain no scheduler.

```bash
P01_ENABLE_GPU=1 bash scripts/p01/python.sh -m proc_research.calibration.run gpu \
  --allow-gpu --stage g0 --config "$P01_R/inputs/config-v1.json" \
  --manifest "$P01_R/inputs/splits-v1.jsonl" --output "$P01_R/runs" --run-id g0-v1
```

G0 selects at most 2 calibration images per dataset. For each image/view it runs original direct `predict`, export inference and repeat direct inference. Required: exact labels, max score difference <=1e-6, lossless cache replay, exactly one encode per view, actual encoder input `[1,3,504,504]` / `[1,3,1008,1008]`, recorded real feature sizes. The encoder pre-hook rejects silent resetting before accepting the output. A failed real view is a blocker, never substituted with a synthetic result. It also runs 5 warmup pairs and 3 timed pairs per dataset and records synchronized timings/peak memory. No enlargement is allowed without the completed G0 receipt.

## Calibration, bounded pilot, evaluation

```bash
P01_ENABLE_GPU=1 bash scripts/p01/python.sh -m proc_research.calibration.run gpu \
  --allow-gpu --stage calibration --config "$P01_R/inputs/config-v1.json" \
  --manifest "$P01_R/inputs/splits-v1.jsonl" --g0-run "$P01_R/runs/g0-v1" \
  --output "$P01_R/runs" --run-id calibration-v1
bash scripts/p01/python.sh -m proc_research.calibration.run fit \
  --cache-run "$P01_R/runs/calibration-v1" --output "$P01_R/runs" --run-id banks-v1
P01_ENABLE_GPU=1 bash scripts/p01/python.sh -m proc_research.calibration.run gpu \
  --allow-gpu --stage pilot --config "$P01_R/inputs/config-v1.json" \
  --manifest "$P01_R/inputs/splits-v1.jsonl" --g0-run "$P01_R/runs/g0-v1" \
  --output "$P01_R/runs" --run-id pilot-cache-v1
bash scripts/p01/python.sh -m proc_research.calibration.run decode \
  --cache-run "$P01_R/runs/pilot-cache-v1" --bank-run "$P01_R/runs/banks-v1" \
  --output "$P01_R/runs" --run-id pilot-pred-v1
```

All 11 methods consume shared caches. Bank artifacts include R_a/R_e, quality mass, background mass, uncalibrated flags, per-class distinct-scene support, RMS factors, exact shuffled assignments (seeds 0/1/2), and scene-subset stability. Inspect every seed and support table. A non-improving mean does not stop calibration. If all conditional rows fall back, the report must remain inconclusive.

An independent evaluator alone receives a separate JSONL manifest with `dataset,image_id,path,sha256`, where `path` is the GT image. Its hashes must be verified from source; do not add these fields to inference manifests. LoveDA raw 0/255 are ignored and 1..7 become 0..6; iSAID ignores 255.

```bash
bash scripts/p01/python.sh -m proc_research.evaluate \
  --predictions "$P01_R/runs/pilot-pred-v1/prediction_index.json" \
  --labels "$P01_R/inputs/pilot-labels.jsonl" --output "$P01_R/runs/pilot-eval-v1"
bash scripts/p01/python.sh -m proc_research.report \
  --prediction-run "$P01_R/runs/pilot-pred-v1" --evaluation-dir "$P01_R/runs/pilot-eval-v1" \
  --bank-run "$P01_R/runs/banks-v1" --output "$P01_R/runs/pilot-report-v1"
```

Evaluation produces full confusion matrices, dataset mIoU, class IoU, scene metrics and hash-pinned artifacts. Report produces every baseline difference, per-scene bootstrap (2000, seed20261030), all protected-class drops, fallback/support and deterministic random/worst failure-case IDs. The pilot can inform a documented new method version/new run IDs; never overwrite prior evidence. Do not inspect confirmation labels during this process.

## Freeze and one confirmation

```bash
bash scripts/p01/python.sh -m proc_research.calibration.run freeze \
  --config "$P01_R/inputs/config-v1.json" --manifest "$P01_R/inputs/splits-v1.jsonl" \
  --g0-run "$P01_R/runs/g0-v1" --pilot-run "$P01_R/runs/pilot-pred-v1" \
  --bank-run "$P01_R/runs/banks-v1" --seal "$P01_R/inputs/confirmation-seal-v1.json"
P01_ENABLE_GPU=1 bash scripts/p01/python.sh -m proc_research.calibration.run gpu \
  --allow-gpu --stage confirmation --config "$P01_R/inputs/config-v1.json" \
  --manifest "$P01_R/inputs/splits-v1.jsonl" --g0-run "$P01_R/runs/g0-v1" \
  --freeze "$P01_R/inputs/confirmation-seal-v1.json" --output "$P01_R/runs" --run-id confirmation-cache-v1
bash scripts/p01/python.sh -m proc_research.calibration.run decode \
  --cache-run "$P01_R/runs/confirmation-cache-v1" --bank-run "$P01_R/runs/banks-v1" \
  --output "$P01_R/runs" --run-id confirmation-pred-v1
bash scripts/p01/python.sh -m proc_research.evaluate \
  --predictions "$P01_R/runs/confirmation-pred-v1/prediction_index.json" \
  --labels "$P01_R/inputs/confirmation-labels.jsonl" --output "$P01_R/runs/confirmation-eval-v1"
bash scripts/p01/python.sh -m proc_research.report \
  --prediction-run "$P01_R/runs/confirmation-pred-v1" --evaluation-dir "$P01_R/runs/confirmation-eval-v1" \
  --bank-run "$P01_R/runs/banks-v1" --output "$P01_R/runs/confirmation-report-v1"
```

The exact sealed bank is required. Persistent per-image confirmation claims under `$P01_RUNTIME/confirmation_claims` prevent reuse by renaming seals, reordering manifests or changing run/config names. Preserve the entire runtime and claims when moving machines; changing its root does not authorize fresh confirmation on exposed images. Only exact interrupted-run resume is allowed. Do not remove claims to retry tuning. A crash keeps evidence; investigate before resuming.

Superiority requires each relevant paired CI lower bound >0; all classes are protected and any >1pp decline restricts the conclusion. Too few independent scenes, all-global fallback or wide intervals means insufficient evidence. Keep model online timing and VRAM/encode counts separate from CPU calibration and cache IO. Never report cached replay speed as online inference speed. The prompt-pool provenance gap remains a limitation even with positive confirmation.

## Actual CPU acceptance commands

```bash
bash scripts/p01/python.sh -m unittest discover -s tests/research -v
bash scripts/p01/python.sh -m proc_research.audit negative --output /home/fengyi/CodeHome/Proc/runtime/p01/audit/negative-new.json
sha256sum -c research/p01/evidence/contract-freeze.sha256
git diff --check
```

Use `research/p01/evidence/verification.json` for actual command exit codes, test count and log hashes. GPU commands above have **no measured GPU results**.
