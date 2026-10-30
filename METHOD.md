# P01 frozen method contract v1 — 2026-10-30

Frozen before pilot labels. This is engineering of **ECD-ProC adaptation**, not a claim of reproducing the whole ActiveSAM system. GPU and scientific acceptance are PENDING. The user-authorized scope is CPU-only this round.

## Frozen model and input

Base ProC commit: `234c87e58fb08f02ba1cf163b2a73f5330d462e9`. No learned weights, training, Qwen, MLLM, API inference, prompt generation, new prompts or routing. Read the shipped LoveDA/iSAID pools; fail closed if absent. Published pool construction provenance is incomplete (`transductive=unknown`).

Run original cached ProC with PGRF then max class aggregation independently for 1008 and 504. Processor resizes the original PIL image directly to each square; no preliminary 504 image subsequently enlarged by a 1008 processor. Serial one-image execution, compile off, FP32 decoder, original configured BF16 image encoder (explicit, never silently changed), FP32 exported responses. A strict all-FP32 configuration requires a new run identity. Hook the actual ViT input and output, assert the input is `[1,3,s,s]`, and record feature shapes. No weights or modules are replaced.

Keep original reducer order: interpolate raw semantic/instance logits to original image grid using bilinear/align_corners=False, sigmoid, object-score weighting/filtering, original PGRF prompt fusion, original max class aggregation. Extra geometry operates on continuous class responses only; same grid is a no-op. No bilinear interpolation of labels. Views cover the full image, with no cropping, padding, partial visibility or extra augmentation in v1. Reject unsupported transforms.

## Cache and identity

One lossless FP32 cache per image/view: prompt and class maps, prompt/class presence, active class IDs, prompt-to-class mapping, original prediction, original dimensions, actual encoder input/features, geometry, image/weight/pool/config/code identities. Cache content SHA256 and immutable metadata; validate before loading and on resume. No GT paths in inference manifests. G0 requires separate direct and export inference at 1008, labels exactly equal and score max absolute difference <=1e-6, plus cache roundtrip; repeat inference twice. Synthetic arithmetic tests do not constitute G0.

## Reference ECD

Official source: https://github.com/VILA-Lab/ActiveSAM/tree/25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe
Vendored unmodified `activesam/ecd.py`, `signatures.py` and MIT license, with hashes. CPU oracle tests use that code independently of the adaptation.

Let S be the active joint class response. Official estimation pools to floor(H/4) x floor(W/4), clamps [0,1], rounds to uint8 and rescales. R_a is the mean response vector over argmax winners. R_e uses exclusive winners (S_k>0.5, all competitors<0.3) if >=2000 pooled pixels, else confident pixels if >=2000, else argmax pixels. Missing means are 0.5, clamped [1e-6,1-1e-6]. Quality mass is the global share of argmax pixels, clamped >=1e-6; uncalibrated means <2000 argmax pixels. Presence for a ProC class is max over its published prompts; all mapped classes are active (there is no EPG). Scenes count only a class winning a positive pooled response, never mere pixel count.

Single-view faithful decoder: D^t_k=log(presence_k*m_k+0.001)+(R^t_k dot S - ||R^t_k||²/2)/sigma², sigma=0.15, restricted to S_k>0. Double-signature agreement selects a class; disagreement or uncalibrated argmax/selection keeps original argmax. Null background compares log(max(m_bg,1e-6))+1-||S||²/(2sigma²) with log(max(presence_k,0.001)*m_k)-||S-R_a,k||²/(2sigma²), and cannot override uncalibrated fallback. This ordering is tested against the unmodified official decoder, separately from the multiview adaptation.

## Multi-view ECD-ProC

Same two calibration views and images for global, conditional and shuffled methods. Global R uses both views; conditional R uses per-scale observations. For each class/condition with fewer than 5 distinct original scenes, replace its entire R_a/R_e row with the global row. **Only signature rows change**: global quality mass, global uncalibrated flags, background mass, presence rule, sigma, prior, null rule, thresholds and fusion are shared. Low scene support is reported separately from uncalibrated pixels.

Expose continuous full squared-distance discriminants (including the common -||S||² term) for both signatures and a separate null score, so views share a comparable domain. Mean the two per-view discriminants with weights 1/2. Eligibility is positive response in at least one view; use a finite per-view floor for an absent class. Decide R_a and R_e separately, apply their null competition (using the official null-comparison prior log(max(presence,0.001)*mass)), then require agreement. Disagreement, any globally uncalibrated winning/fallback class, or no active evidence falls back to **uncalibrated two-view average**. Null-background comparison is disabled for such fallback pixels. Prior-free control sets foreground mass=1 and null mass=1/C, preserving presence and all other rules.

Uncalibrated baseline B=(S_1008+S_504)/2, argmax followed by max(B)<0.1 -> background 0. For ECD-ProC use this same raw B<0.1 gate, never apply a probability threshold to log scores. RMS: f_c=sqrt(mean_calibration S_c²), zero ->1, positive floor 1e-6. Compare S_c/f_c, with per-class threshold 0.1/f_c for the winning class; for two views compare averaged normalized response against the winning class's mean(0.1/f_v,c). Raw and RMS single-view baselines use 1008 only. Single-view ECD-ProC applies the faithful decoder then the original raw max<0.1 gate. Report faithful-oracle parity separately.

Shuffle seeds 0,1,2 independently permute condition labels over canonical `(dataset,scene,image,view)` records, preserving exact bucket counts. Reject identity, a mere global scale-name swap, or any assignment that keeps every scale internally homogeneous. Fit and application assignments are separately generated without labels and persisted with their hashes. Fewer than four scene-view records cannot support this control and must fail explicitly.

## Isolation, evaluation and resource gates

At most 50 calibration, 50 pilot and 200 confirmation images per dataset; at most 600 overall, LoveDA/iSAID only. Split by verified original source_scene_id; hash/scene overlap must both be zero. Require source evidence, never infer LoveDA scene independence from numeric filenames. Deterministic selection reads image metadata/hashes and scene evidence only. Geographic independence remains unproven unless separately documented. Labels belong solely to the evaluator and are unavailable to prediction, signature fitting or selection.

Evaluator: full-image confusion matrices aggregated to dataset IoU/mIoU, absent union excluded, LoveDA raw 0/255 ignored and raw 1..7 mapped to 0..6; iSAID 255 ignored. Pair by scene, resample 2000 times with seed 20261030, compute metrics from resampled summed confusion matrices, 95% percentile CI. Lower bound >0 required for superiority to each named baseline; <=1 independent scene -> INCONCLUSIVE. All classes protected: any >1pp decline limits conclusion. Report each shuffle seed, fallback counts, scene support, random and worst failure cases, signature stability under scene subsets.

Serial batch caches shared by every method; estimate uncompressed bytes before writing, keep >=20GiB free, fail rather than delete or change precision. Resource report separates synchronized online latency, warmup (5), timed repeats (3), peak allocated/reserved VRAM, encodes and prompt work from cache IO and CPU calibration/decode costs. Immutable run IDs and input/config/code/METHOD/weight/pool hashes; changed identity needs a new run. Confirmation requires sealed config and G0/pilot receipts, then one execution without retuning. This round supplies these entrypoints but does not execute GPU commands or claim experimental gains.
