# Stage B Track Feature Export Schema v1 (Implementation-Ready)

## 1) Scope and Authority
This document is the authoritative v1 export contract for Stage B track features consumed by Stage C attribution.

- In-scope: artifact layout, metadata/array schema, deterministic ordering/alignment invariants, provenance, compatibility rules.
- Out-of-scope: implementation code, training logic, Stage C algorithm design.

Normative language:
- `REQUIRED`: must be present/true for v1-compliant producer or consumer.
- `OPTIONAL (future)`: allowed extension point, not required for v1 compliance.

## 2) Human-Frozen v1 Decisions (Normative)
The following are fixed for v1 and not optional:

1. Embedding form: `REQUIRED` pooled per-track embedding only.
2. No sequence embeddings: per-frame / variable-length embeddings are not part of required v1 export.
3. Artifact granularity: `REQUIRED` split-level manifest + per-video payload files.
4. Track identity: `track_id` uniqueness scope is video-local only; Stage C primary key is `(video_id, track_id)`.
5. Zero-track handling: zero-track videos must remain in manifest with explicit zero-track processed status.
6. Serialization: `REQUIRED` JSON metadata + NPZ arrays.

## 3) Recommended v1 Layout (Required Recommendation)

### 3.1 Directory layout
`<export_root>/track_feature_export_v1/<split>/`

Required files per split:
- `manifest.v1.json`
- `videos/<video_id>/track_metadata.v1.json` (for processed videos)
- `videos/<video_id>/track_arrays.v1.npz` (for processed videos)

Example:
```text
track_feature_export_v1/
  train/
    manifest.v1.json
    videos/
      vid_000001/
        track_metadata.v1.json
        track_arrays.v1.npz
      vid_000002/
        track_metadata.v1.json
        track_arrays.v1.npz
```

### 3.2 Path rules
- All paths inside JSON are `REQUIRED` relative to split root (`<export_root>/track_feature_export_v1/<split>/`).
- Producers must not emit absolute paths.

## 4) Serialization Contract (JSON + NPZ)

### 4.1 JSON responsibilities
JSON stores structure and semantics:
- dataset/split/export provenance
- per-video processing status and payload paths
- per-track scalar metadata and row binding

### 4.2 NPZ responsibilities
NPZ stores dense numeric arrays:
- pooled embeddings
- optional numeric vectors that require fast batched loading

### 4.3 Required NPZ keys (v1)
`track_arrays.v1.npz` must include:
- `embeddings`: `float32`, shape `[N, D]`, pooled per-track vectors.
- `track_row_index`: `int64`, shape `[N]`, exact values `0..N-1`.

May include now (still optional):
- `objectness_score`: `float32`, shape `[N]`.

## 5) Split Manifest Schema (`manifest.v1.json`)

### 5.1 Top-level required fields
- `schema_name` (`string`): must equal `"stage_b_track_feature_export"`.
- `schema_version` (`string`): must equal `"1.0.0"` for this v1 spec.
- `split` (`string`): split identifier (e.g., `train`, `val`, `test`).
- `embedding_dim` (`integer > 0`): D for all processed videos in this split artifact.
- `embedding_dtype` (`string`): must be `"float32"` in v1.
- `embedding_pooling` (`string`): must be `"track_pooled"` in v1.
- `embedding_normalization` (`string enum`): `"none" | "l2"`.
- `producer` (`object`): required provenance block (see Section 8).
- `videos` (`array`): one record per video expected in split processing domain.

### 5.2 Per-video record required fields
- `video_id` (`string`)
- `status` (`string enum`):
  - `"processed_with_tracks"`
  - `"processed_zero_tracks"`
  - `"failed"`
  - `"unprocessed"`
- `num_tracks` (`integer >= 0`)
- `track_metadata_path` (`string | null`)
- `track_arrays_path` (`string | null`)

### 5.3 Per-video status invariants
- If `status == "processed_with_tracks"`:
  - `num_tracks > 0`
  - `track_metadata_path` and `track_arrays_path` are non-null and must exist.
- If `status == "processed_zero_tracks"`:
  - `num_tracks == 0`
  - `track_metadata_path` and `track_arrays_path` are non-null and must exist.
- If `status in {"failed", "unprocessed"}`:
  - `num_tracks == 0`
  - `track_metadata_path == null` and `track_arrays_path == null`.

## 6) Per-Video Track Metadata Schema (`track_metadata.v1.json`)

### 6.1 Top-level required fields
- `schema_name` (`string`): `"stage_b_track_feature_export_video"`
- `schema_version` (`string`): `"1.0.0"`
- `split` (`string`)
- `video_id` (`string`)
- `num_tracks` (`integer >= 0`)
- `tracks` (`array` length `num_tracks`)

### 6.2 Per-track required fields (v1)
- `row_index` (`integer >= 0`): row in NPZ arrays.
- `track_id` (`string | integer`): unique within this `video_id`; serialized type must be consistent within a split artifact.
- `start_frame_idx` (`integer >= 0`, inclusive)
- `end_frame_idx` (`integer >= start_frame_idx`, inclusive)
- `num_active_frames` (`integer > 0`)
- `objectness_score` (`number`, recommended range `[0,1]`, no hard calibration assumption)

### 6.3 Per-track optional future fields (not required v1)
- `mask_quality_score` (`number`)
- `track_stability_score` (`number`)
- `bbox_ref` (`object`) for external geometry references
- `mask_ref` (`object`) for external mask references
- `is_filtered_candidate` (`boolean`) for downstream candidate filtering flags
- `unknown_or_background_hint` (`string enum`), future Stage C extension

## 7) Deterministic Ordering and Alignment Invariants (Critical)

### 7.1 Video ordering in manifest
`videos` array must be sorted lexicographically by `video_id` ascending.

### 7.2 Track ordering in per-video metadata
`tracks` array must be sorted by this key chain:
1. `start_frame_idx` ascending
2. `end_frame_idx` ascending
3. `track_id` ascending using the split-consistent serialized type rule:
   - integer `track_id`: numeric ascending
   - string `track_id`: lexicographic ascending

Tie-breaking must be deterministic and stable. Producer must not emit nondeterministic order.

### 7.3 JSON-to-NPZ row alignment
For each per-video payload:
- `tracks[i].row_index == i` for all `i in [0, N-1]`.
- `track_arrays.v1.npz["track_row_index"][i] == i`.
- Embedding for track `tracks[i]` is `embeddings[i, :]`.

Consumer validation requirement:
- Hard-fail the video payload if any row alignment invariant is violated.

### 7.4 Count invariants
For each processed video (`processed_with_tracks` or `processed_zero_tracks`):
- `video_record.num_tracks == track_metadata.num_tracks == embeddings.shape[0] == len(track_row_index)`.

## 8) Provenance Metadata (Required)
Manifest `producer` object must include:
- `stage_b_checkpoint_id` (`string`): model/checkpoint identifier.
- `stage_b_checkpoint_hash` (`string`): checksum or immutable hash.
- `stage_b_config_ref` (`string`): config path or ID.
- `stage_b_config_hash` (`string`): config content hash.
- `pseudo_tube_manifest_id` (`string`): source pseudo-tube dataset/manifest ID.
- `pseudo_tube_manifest_hash` (`string`): immutable hash of source manifest.
- `split` (`string`): duplicate for provenance clarity.
- `extraction_settings` (`object`) with required fields:
  - `frame_sampling_rule` (`string`)
  - `pooling_rule` (`string`, must match track-pooled semantics)
  - `min_track_length` (`integer >= 1`)

Recommended (non-blocking) additional provenance:
- `export_timestamp_utc`
- `git_commit`
- `host`

## 9) Stage C Minimum Consumer Contract (v1)
A Stage C v1 consumer must require at least:
- Split-level: `schema_name`, `schema_version`, `split`, embedding contract fields, `producer`, `videos`.
- Per-video: `video_id`, `status`, `num_tracks`, payload paths.
- Per-track: `(video_id, track_id)` identity, temporal span fields, `row_index`, `objectness_score`.
- Array: `embeddings` aligned to metadata rows.

Filtering compatibility:
- Stage C may filter by `objectness_score` and/or temporal length.
- Filtering is applied after loading aligned rows, never by reordering rows.

## 10) Zero-Track and Failure Handling

### 10.1 Zero-track processed video (required representation)
- Must be present in manifest with `status = "processed_zero_tracks"` and `num_tracks = 0`.
- Must have valid per-video JSON/NPZ payloads with empty track tables/arrays.
- NPZ embeddings shape must be `[0, D]`.

### 10.2 Failed or unprocessed video
- Must be present if in split processing domain.
- Must use `status = "failed"` or `"unprocessed"`.
- Must have `num_tracks = 0` and null payload paths.

## 11) Versioning and Compatibility Policy

### 11.1 Version field
- `schema_version` is semantic version string `MAJOR.MINOR.PATCH`.
- This spec fixes v1 at `1.0.0`.

### 11.2 Backward-compatible changes within MAJOR=1
Allowed without major bump:
- add new optional fields
- add new optional NPZ keys
- add new optional status metadata that does not alter required semantics

### 11.3 Breaking changes (require MAJOR bump)
- changing required field names/types/semantics
- changing ordering or row-alignment rules
- changing primary embedding form from pooled per-track
- changing identity semantics from `(video_id, track_id)`

Consumer rule:
- Consumers implementing v1 must hard-fail on unknown MAJOR.
- Consumers should ignore unknown optional fields when `MAJOR == 1`.

## 12) Concrete Synthetic Snippet

### 12.1 Manifest snippet
```json
{
  "schema_name": "stage_b_track_feature_export",
  "schema_version": "1.0.0",
  "split": "train",
  "embedding_dim": 256,
  "embedding_dtype": "float32",
  "embedding_pooling": "track_pooled",
  "embedding_normalization": "l2",
  "producer": {
    "stage_b_checkpoint_id": "seqformer_b_ckpt_042",
    "stage_b_checkpoint_hash": "sha256:abc123",
    "stage_b_config_ref": "configs/stage_b/seqformer_b.yaml",
    "stage_b_config_hash": "sha256:def456",
    "pseudo_tube_manifest_id": "ptubes_train_v2",
    "pseudo_tube_manifest_hash": "sha256:987xyz",
    "split": "train",
    "extraction_settings": {
      "frame_sampling_rule": "uniform_stride_2",
      "pooling_rule": "mean_over_active_frames",
      "min_track_length": 2
    }
  },
  "videos": [
    {
      "video_id": "vid_000001",
      "status": "processed_with_tracks",
      "num_tracks": 2,
      "track_metadata_path": "videos/vid_000001/track_metadata.v1.json",
      "track_arrays_path": "videos/vid_000001/track_arrays.v1.npz"
    },
    {
      "video_id": "vid_000002",
      "status": "processed_zero_tracks",
      "num_tracks": 0,
      "track_metadata_path": "videos/vid_000002/track_metadata.v1.json",
      "track_arrays_path": "videos/vid_000002/track_arrays.v1.npz"
    },
    {
      "video_id": "vid_000003",
      "status": "failed",
      "num_tracks": 0,
      "track_metadata_path": null,
      "track_arrays_path": null
    }
  ]
}
```

### 12.2 Per-video track metadata snippet
```json
{
  "schema_name": "stage_b_track_feature_export_video",
  "schema_version": "1.0.0",
  "split": "train",
  "video_id": "vid_000001",
  "num_tracks": 2,
  "tracks": [
    {
      "row_index": 0,
      "track_id": 3,
      "start_frame_idx": 12,
      "end_frame_idx": 30,
      "num_active_frames": 19,
      "objectness_score": 0.91
    },
    {
      "row_index": 1,
      "track_id": 8,
      "start_frame_idx": 35,
      "end_frame_idx": 52,
      "num_active_frames": 18,
      "objectness_score": 0.77
    }
  ]
}
```

## 13) Considered but Not Selected for v1
- Single monolithic split NPZ containing all videos.
- Global `track_id` uniqueness across split/dataset.
- Per-frame or variable-length sequence embedding as required export.

## 14) Future / Non-blocking Open Questions
These are explicitly deferred and not needed for v1 implementation:
- Whether to standardize calibration semantics for `objectness_score` across checkpoints.
- Whether to require geometry/mask references in v2.
- Whether to add optional per-track uncertainty vectors in NPZ.
- Whether to add shard-level manifests for very large splits.
