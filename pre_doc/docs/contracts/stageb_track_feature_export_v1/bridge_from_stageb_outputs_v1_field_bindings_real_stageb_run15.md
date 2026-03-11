# Bridge Field Bindings from Real Sample: run15 (`gpu4090d`)

## 1) Scope and authority

This document is a concrete binding addendum for one real sample run:
- host/path: `gpu4090d:/home/zyy/code/wsovvis_runner/runs/wsovvis_seqformer/15`
- inspected on: 2026-02-28

It refines concrete source paths/keys against:
- `docs/contracts/stageb_track_feature_export_v1/bridge_from_stageb_outputs_v1.md`
- `docs/contracts/stageb_track_feature_export_v1/schema_track_feature_export_v1.md`

It does **not** redefine frozen bridge semantics. This is inspection + mapping only.
Gap-resolution decisions for D1-D4 are documented in `docs/contracts/stageb_track_feature_export_v1/bridge_gap_resolution_for_real_stageb_to_bridge_input_v1.md`.

## 2) Sample provenance context

- Run metadata source: `run15/run.json`
- Runtime config sources: `run15/config.json`, `run15/cfg_runtime.json`
- Inference result sources: `run15/d2/inference/results.json`, `run15/d2/inference/instances_predictions.pth`
- Split-domain candidate sources referenced by `config.json`:
  - `/home/zyy/code/wsovvis_runner/data/LV-VIS/annotations/lvvis_val_agnostic.json`
  - `/home/zyy/code/wsovvis_runner/outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json`

## 3) Real output layout inventory (run15)

- Run root contains: run metadata/config (`run.json`, `config.json`, `cfg_runtime.json`), logs (`cout.txt`, `d2/log.txt`), checkpoints (`d2/model_*.pth`), and inference outputs (`d2/inference/*`).
- Per-video Stage B JSON files: not observed in run15 sample.
- Inference result pattern: aggregated files only.
  - `d2/inference/results.json` (single large JSON line)
  - `d2/inference/instances_predictions.pth` (list of prediction records)

Observed inference record shape (from `instances_predictions.pth`):
- top-level: `list` of length `8370`
- item keys: `video_id`, `score`, `category_id`, `segmentations`
- `segmentations[*]` keys: `size`, `counts`
- `track_id`, `embedding`, `runtime_status`, `tracks` fields: not observed in this sample

## 4) Runtime status / processing-state encoding

Observed explicit run-level status signals:
- `run.json.status = "COMPLETED"`
- `d2/last_checkpoint = model_final.pth`
- no explicit error markers (`ERROR`, `Traceback`, `Exception`, `failed`) found in searched run logs

Observed per-video runtime status tokens:
- not observed in run15 sample (`runtime_status` key absent in inspected inference records)

Mapping implication:
- `processed_with_tracks` can be inferred for this sample domain because each val video ID has records.
- `processed_zero_tracks`, `failed`, `unprocessed` are not directly evidenced in run15 output files.

## 5) Concrete binding table (run15)

| Real source location (run15) | Example type/value | P3.1a abstract placeholder | P3.1b normalized key | Transform / coercion | Requiredness for bridge | Missing / malformed behavior | Notes (fact vs assumption) |
|---|---|---|---|---|---|---|---|
| `config.json.data.val_json` | `"data/LV-VIS/annotations/lvvis_val_agnostic.json"` | `split_domain_manifest` source path | `split_domain_video_ids` (derived) | Resolve relative to runner repo; load JSON `videos[*].id`; sort/unique if needed | `REQUIRED` | Hard-fail if file missing/unreadable/invalid JSON | `FACT` in run15; used as nearest split-domain source |
| `config.json.data.train_json` | `"outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json"` | `producer.pseudo_tube_manifest_*` candidate source | `producer.pseudo_tube_manifest_id/hash` | Use as provenance reference (ID/hash derivation still implementation-specific) | `REQUIRED` by bridge contract | Hard-fail if required producer fields cannot be populated | `FACT` path exists; ID/hash derivation `OPEN` |
| `lvvis_val_agnostic.json.videos[*].id` | int range `0..836` | `split_domain_manifest.video_ids[]` | `split_domain_video_ids[]` | Cast to canonical video ID type for bridge domain (string/int policy `OPEN`) | `REQUIRED` | Hard-fail on duplicates/invalid IDs | `FACT` keys observed |
| `lvvis_val_agnostic.json` (top-level has no `split`) | no explicit split key | `split_domain_manifest.split` | `split` | Obtain split from run config context (`DATASETS.TEST`/path semantics) | `REQUIRED` | Hard-fail if unresolved | `OPEN` concrete field source in run15 |
| `d2/inference/instances_predictions.pth[*].video_id` | int | `stageb_video_result.video_id` (nearest equivalent) | `stageb_video_results[*].video_id` | Group records by `video_id` | `REQUIRED` for result records | If absent in a record -> drop/fail per parser policy | `FACT` key observed |
| `d2/inference/instances_predictions.pth[*].score` | float in approx `[0.0175,0.9495]` | `stageb_track.objectness_score` candidate | `stageb_video_results[*].tracks[*].objectness_score` | Map directly as float score | `DEFAULT_FOR_V1` candidate | Missing/non-numeric -> track invalid | `OPEN`: interpretation as objectness/confidence needs confirmation |
| `d2/inference/instances_predictions.pth[*].category_id` | int (`1`) | optional class metadata | (none required in P3.1b core) | keep/ignore for v1 export | optional | ignore if unused | `FACT` present; currently not required by bridge core |
| `d2/inference/instances_predictions.pth[*].segmentations` | list of RLE masks (`size`,`counts`) | possible track temporal support evidence | could inform `start_frame_idx/end_frame_idx/num_active_frames` | derive frame support from non-empty segmentation entries | needed only if used as track representation | if derivation fails -> track invalid / `OPEN` | `OPEN`: derivation rule not frozen in this sample |
| `d2/inference/results.json` | aggregated JSON with same visible key prefix (`video_id/score/category_id/segmentations`) | alternate source for same semantics | same as above | use only if `.pth` unavailable | optional fallback | hard-fail if neither source parseable | `FACT` prefix observed; full parse not performed |
| `run.json.status` | `"COMPLETED"` | run-level runtime status source | (diagnostic only) | map to run success/failure signal | recommended | if missing, infer from logs/checkpoints with caution | `FACT` observed |
| `run.json.start_time/stop_time` | ISO timestamps | provenance metadata | optional producer audit fields (outside P3.1b required keys) | copy string | optional | none | `FACT` observed |
| `config.json.d2_cfg_path` + `d2_opts` + `cfg_runtime.json` | config path + option list | `producer.stage_b_config_ref` candidate | `producer.stage_b_config_ref` | use concrete config ref path/string | `REQUIRED` | hard-fail if unresolved | `FACT` source exists |
| `d2/last_checkpoint` + checkpoint files | `model_final.pth` and `model_*.pth` | `producer.stage_b_checkpoint_id/hash` candidate | `producer.stage_b_checkpoint_id/hash` | ID from filename/path; hash from file digest | `REQUIRED` | hard-fail if required provenance cannot be filled | `FACT` files exist; hash extraction `OPEN` |
| Inference records: no `runtime_status` field | not present | `stageb_video_result.runtime_status` | `stageb_video_results[*].runtime_status` | infer from run-level and per-video record existence if allowed | `REQUIRED` in current P3.1b input schema | if unresolved, parser must fail or apply explicit adapter mode | `OPEN` mismatch to current normalized schema |
| Inference records: no `track_id` field | not present | `stageb_video_result.tracks[*].track_id` | `stageb_video_results[*].tracks[*].track_id` | synthesize deterministic ID or derive from upstream track linkage | `REQUIRED` | missing -> track invalid / hard-fail depending policy | `OPEN` |
| Inference records: no embedding vector field | not present | `stageb_video_result.tracks[*].track_feature_vector` | `stageb_video_results[*].tracks[*].embedding` | unavailable in run15 outputs; requires another artifact/source | `REQUIRED` | hard-fail for export path needing embeddings | `OPEN` critical blocker for P3.1c.1 if only run15 artifacts are used |

## 6) Edge-case inventory and policy impact

Observed in run15:
- All val videos covered by prediction records: `837/837` (`missing=0`, `extra=0` relative to `val_json` IDs).
- Prediction multiplicity: exactly 10 records per video in this output.
- No explicit per-video failure markers found.

Coverage target status from this sample:
- `processed_with_tracks`: observed/inferable
- `processed_zero_tracks`: **not observed in run15 sample**
- `failed`: **not observed in run15 sample**
- `unprocessed`: **not observed in run15 sample**

Policy impact:
- Frozen defaults remain unchanged: normalization default `none`, duplicate Stage-B results hard-fail default, non-finite embedding reject-only.
- Because concrete `embedding` and `runtime_status` fields are absent in run15 outputs, parser implementation must either:
  - consume an additional upstream artifact with these fields, or
  - introduce an explicitly documented inference mode (outside this doc-only task).

## 7) Unresolved mismatches / OPEN items

- `OPEN-1`: Concrete per-video `runtime_status` field not found in run15 inference artifacts.
- `OPEN-2`: Concrete `track_id` field not found in run15 inference artifacts.
- `OPEN-3`: Concrete per-track embedding/feature-vector field not found in run15 inference artifacts.
- `OPEN-4`: Concrete source of `split` string for bridge top-level (`split_domain_manifest.split`) is indirect in run15 (`val_json` has no `split` key).
- `OPEN-5`: Exact derivation of `start_frame_idx`, `end_frame_idx`, `num_active_frames` from segmentation-only outputs is not frozen by existing contract.
- `OPEN-6`: Provenance hash extraction locations (`checkpoint_hash`, `config_hash`, `pseudo_tube_manifest_hash`) are not directly materialized in run15 metadata files.

## 8) P3.1c.1 readiness checklist

- [x] Real sample root inspected with reproducible commands.
- [x] Artifact layout and representative structures captured.
- [x] Split-domain candidate source identified and cross-checked against result domain.
- [x] Concrete source keys mapped where observed.
- [ ] Runtime-status / track-id / embedding concrete fields resolved for this sample.
- [x] OPEN blockers listed for parser implementation planning.
