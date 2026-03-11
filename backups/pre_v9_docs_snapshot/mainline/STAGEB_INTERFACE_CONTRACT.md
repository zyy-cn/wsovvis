# WSOVVIS Stage B Interface Contract

This file migrates the essential Stage B export / bridge / consumer contract into the new automation layer.
It is authoritative for preserving compatibility with the existing codebase.

## 1. Scope
This contract covers the minimum Stage B → Stage C interface required by the current code:
- export artifact shape
- bridge-input expectations
- Stage C consumer assumptions
- provenance requirements needed for reproducibility

## 2. Export artifact form
Required v1 form:
- split-level manifest + per-video payload files
- JSON metadata + NPZ arrays
- pooled per-track embeddings only
- no sequence embeddings as required v1 payload

Recommended layout under an export root:
- `track_feature_export_v1/<split>/manifest.v1.json`
- `track_feature_export_v1/<split>/videos/<video_id>/track_metadata.v1.json`
- `track_feature_export_v1/<split>/videos/<video_id>/track_arrays.v1.npz`

## 3. Identity and alignment invariants
Required invariants:
- Stage C primary key is `(video_id, track_id)`
- `track_id` uniqueness is video-local
- processed zero-track videos remain explicit in the manifest
- metadata track ordering and array row ordering must be deterministic
- metadata rows and array rows must align exactly by `row_index`

## 4. Required per-track payload
At minimum, Stage C depends on:
- `track_id`
- `row_index`
- pooled embedding vector
- temporal support fields or equivalents (`start_frame_idx`, `end_frame_idx`, `num_active_frames`) when available
- `objectness_score` when available

## 5. Bridge-input expectations
The bridge layer must provide enough normalized information for Stage C loading and attribution.
Boundary rules:
- downstream normalized runtime status remains in a reduced success/failure domain
- split-domain reconciliation is downstream logic, not an upstream export token
- export/bridge contracts should not silently widen status taxonomy

## 6. Required provenance
The run-level metadata must preserve at least:
- Stage B checkpoint reference/hash
- Stage B config reference/hash
- pseudo-tube manifest reference/hash
- extraction settings such as sampling / pooling / min-track rules

## 7. Change policy
Do not change Stage B schema, bridge semantics, or consumer assumptions in the mainline unless:
1. the current gate explicitly requires it,
2. this file is updated, and
3. related Stage B tests are updated in the same change.
