# P01 runtime budget — estimate, not a new GPU measurement

2026-10-30. No GPU query, allocation or inference was performed. Assumes successful G0, serial views, unchanged precision and all controls sharing score caches. Model/data setup and human pilot review time are excluded.

Historical source machine is TITAN RTX, confirmed by external `audit/initial-system.txt`. At source commit 742ce170…, original 1008-view ProC recorded:

- LoveDA: 5,839.825152 s / 1,669 images = 3.498996 s/image.
- iSAID: 34,382.894939 s / 11,644 images = 2.952842 s/image.

Sources: `/home/fengyi/CodeHome/ProC-SAM3-repro/results/full/proc/{LoveDA,iSAID}/run.json`. Those times include initialization/file handling in the old runner; they are not isolated GPU kernel measurements and cannot count as current P01 G0.

Official TITAN RTX specs give 4,608 CUDA cores and 1.77 GHz boost; FP32 peak is calculated as 4608×1.77×2/1000 = 16.31232 TFLOPS. RTX4090 official FP32 peak is 82.6 TFLOPS. Ideal ratio = 5.0636574. Memory bandwidth ratio is only 1008/672 = 1.5; both have 24 GB. Current encoder is BF16, decoder FP32, TF32 disabled. TITAN Turing lacks native BF16 arithmetic, and PyTorch 2.7.1's local is_bf16_supported includes an emulation path. Therefore no uniform real speedup factor or guaranteed cross-device compatibility is inferred.

Primary hardware references:

- [NVIDIA TITAN RTX specifications](https://www.nvidia.com/en-us/titan/titan-rtx/)
- [NVIDIA Ada whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

For a transparent arithmetic scenario only, set t504/t1008 between 0.25 and 1.0. Pixel count alone does not establish this ratio: full prompt decoding and original-grid postprocessing remain. Use equal dataset counts. Current G0 adds 28 paired-view equivalents: 4 images × 3 direct/export/repeat pairs, plus 2 datasets × 8 warmup/timed pairs.

| Scope | Main views | TITAN old-run extrapolation, including G0 | 4090 ideal FP32-ratio scenario |
|---|---:|---:|---:|
| Current preparation example: each dataset 50 calibration + 50 pilot + 80 confirmation = 360 images | 720 | about 26–42 min | about 5–8 min |
| Authorized maximum: each dataset 50 + 50 + 200 = 600 images | 1,200 | about 42–68 min | about 8–14 min |

These columns exclude added score-cache hash/IO, CPU calibration, all eleven decoders, evaluator/report/bootstrap, repeated model-load overhead and troubleshooting. They are scenarios, not confidence intervals or guaranteed bounds. The 11 methods do not each require another SAM3 forward pass.

For the full 600-image procedure, retain the earlier feasibility review's **4090 2–8 h machine-time reservation**. If CPU/storage overheads are comparable between hosts, reserve roughly **TITAN 3–9 h**; this is a planning buffer, not a measured prediction. The GPU-only arithmetic saves roughly half to one hour, while unchanged CPU/IO overhead is not divided by five. For the 360-image example, a rough smaller reservation is 4090 1–5 h / TITAN 1.5–6 h; fixed calibration/G0 work prevents exact linear scaling. A reliable end-to-end estimate requires future authorized G0 plus a small measured cache/calibration/decode sample.

Formula: total = sum over selected images of (t1008+t504) + G0/model initialization + cache writes/verification + CPU fit/decode/evaluation/report. Once actual G0 pair timings and cache/CPU costs exist, replace the hypothetical ratio and planning buffers with those measurements. Full scientific acceptance remains PENDING.
