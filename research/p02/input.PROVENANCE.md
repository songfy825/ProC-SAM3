# Offline input evidence contract

`input.real.template.json` is intentionally incomplete. Copy it into `R/runtime/p02/inputs/`, then fill references from **existing records**. Every reference has exactly `path` and the independently recorded lowercase SHA256. Relative references resolve beside the input manifest. Do not synthesize cache provenance, use synthetic fixtures as real data, create a new development list by selecting favorable outputs, or generate model outputs in this task.

`load_manifest(path)` returns `{kind, samples, selected, evidence}`. Missing required files/records or per-prompt maps raise `PendingEvidence` (CLI exit 2). A supplied corrupt hash, receipt, geometry, mapping or association raises `ValueError` (exit 1). Valid synthetic bundles remain `REAL_ACTION_SPACE=PENDING`.

## Reference files

| Manifest key | Required content |
| --- | --- |
| `development` | Existing legal development/pilot JSONL, one image per row; schema below. Entire list is pinned and sorted before cache availability is consulted. |
| `production_index` | Original P01 `cache_index.json`: `entries` with dataset, image_id, scene_id, view, path, receipt. Read only selected view 1008 caches; never traverse old cache trees. |
| `production_manifest` | Original run `manifest.json`, with `status=PASS`, `identity` and `identity_sha256`. Identity must contain `evidence_kind=real_model`, config and image-production code/weight/pool/config hashes. |
| `production_receipt` | Original run `complete.json`, `status=PASS`, kind `real_gpu_cache` or `real_gpu_g0`, and `files` hashes binding the manifest and index. This describes historical production only; it never authorizes inference now. |
| `independent_receipt` | Existing independent audit outside the producer run directory, schema below. Merely copying cache receipt to a new filename is not independent evidence. |
| `production_log` | Nonempty preserved producer execution log, independently bound by the audit. |
| `use_history` | Existing documented eligibility and previous use record, schema below. |
| `producer_config` | Exact JSON config used by producer; hash and content must match producer identity. It may describe other cached views, but P02 reads only 1008 and never calibrates. |
| `producer_code_inventory` | Archived `{files:{relative_source_path:sha256,...}}`; canonical digest of mapping must equal producer `code_sha256`. No traversal of old caches or inference needed. |
| `asset_receipt` | Independent existing weight audit with `weight.sha256`, matching the producer identity. Weight bytes are not reread this round. |

Hashes establish artifact integrity and cross-record consistency; origin still depends on an authentic external audit/production record. Software cannot turn self-written assertions into evidence of real inference. The supplied repository has no real caches, so no real admission has been exercised. Existing published prompt pools are checked using the repository's frozen SHA256 values and exact original vocabulary expansion; no prompt generation/model call occurs. Prompt-pool construction data history remains unknown.

## Development row

Required: `dataset` (`LoveDA` or `iSAID`), nonempty string `image_id`, `split` (`development` or `pilot`), `image_path`, `image_sha256`, positive integer `height`, `width`. Optional: `source_scene_id`, `scene_evidence`, `city_or_flight`. No GT/label/performance fields. Relative image paths resolve beside this JSONL. The first 50 lexicographically sorted IDs **per dataset** are fixed before loading any cache or GT. Missing selected cache makes the run PENDING; it never substitutes image 51. Missing scene identity must remain empty; no filename-derived geographic independence is claimed.

## Independent audit schema

- `kind`: `independent_cache_audit`; `status`: `PASS`.
- Nonempty `auditor`, `recorded_at`, `producer_command` identifying the independent audit and historical production invocation.
- `bindings`: SHA256 values keyed by `development`, `production_index`, `production_manifest`, `production_receipt`, `production_log`, `use_history`, `producer_config`, `producer_code_inventory`, `asset_receipt`.
- `exports`: one object for each audited `(dataset,image_id,view)` with `receipt` (`meta_sha256`,`arrays_sha256`), exact `geometry`, and `provenance` containing image/weight/pool/config/code SHA256 fields. Every selected 1008 export must match its actual cache and index.

Use genuine producer/auditor records as the basis of an adapter if their historical schema differs. Do not invent these attestations to make the CLI pass. Both local cache receipt and external parent receipt are checked, followed by `proc_research.cache.load_sample(expected=...)`.

## Use-history schema

`{development_sha256, selection_basis:"existing-development-list", rows:[...]}`. Each selected row records `dataset`, `image_id`, matching `image_sha256`, `development_eligible:true`, a nonempty `eligibility_evidence` reference/explanation, and `prior_uses` list (empty only if documented as unused). When source scene is provided, `source_scene_id` and `scene_evidence` must match the development row. Missing source scene never blocks geometry-only diagnostics, but remains insufficient for independent-scene claims. Calibration/confirmation samples cannot be silently relabeled as development.

## Synthetic fixture and labels

Tests expose `tests/p02/test_provenance.py:write_fixture(directory)`, returning a complete tiny `synthetic-input.json` with separately named synthetic development/index/cache records. It proves algorithm/CLI contracts only. Every synthetic sample/index retains a synthetic marker; changing just the top-level kind is rejected.

GT is a **separate** optional CLI label manifest, never a field of this input manifest or development rows. The evaluation module alone checks label association/mapping. Geometry validity is full original image for supported full_resize caches; GT ignore masks never enter selection.
