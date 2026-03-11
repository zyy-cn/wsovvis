# WS-OVVIS Stage B and Artifact Transition Contract

This file preserves the essential contract discipline from the current repository while extending it toward the v9 mainline.
It is authoritative for the minimum structure-side artifacts needed by the v9 control plane.

## 1. Scope
This contract covers the minimum transition from the current Stage B ecosystem to the v9 artifact stack:
- local tracklet export
- clip-level global track bank
- track semantic cache
- provenance requirements needed for reproducibility
- evidence and provenance hooks needed for worked examples

## 2. Local tracklet artifact form
Required v9-oriented form:
- split-level manifest plus per-video payload files
- JSON metadata plus NPZ arrays or equivalent structured caches
- identity preserved at least by `(video_id, local_track_id)`

Required per-tracklet payload at minimum:
- `local_track_id`
- `row_index`
- window or clip provenance
- temporal support fields
- masks or mask references sufficient for downstream stitching
- local query feature or equivalent linking-side representation
- local mask quality or objectness-related fields when available

Evidence and provenance additions strongly preferred:
- stable identifiers for worked-example selection
- direct references to the source pseudo-tube or pseudo-instance record when available

## 3. Global track bank artifact form
Required invariants:
- primary key is `(video_id, global_track_id)`
- `global_track_id` uniqueness is video-local
- global-track membership over local tracklets is deterministic
- per-frame or temporal support information is sufficient for crop or feature extraction
- zero-track videos remain explicit in the manifest

Evidence and provenance additions strongly preferred:
- merge-membership trace or equivalent loggable structure for worked examples
- score-matrix or matching-summary sidecar when available

## 4. Track semantic cache form
Required per-global-track payload at minimum:
- `global_track_id`
- `z_tau` or a reserved field for it before the extractor is active
- `o_tau` or sufficient components to derive it
- temporal support fields
- crop or pooling provenance needed for reproducibility

Evidence and provenance additions strongly preferred:
- worked-example fields or sidecars for crop boxes, visible-frame indices, and aggregation weights

## 5. Provenance requirements
Run-level metadata should preserve at least:
- Stage B checkpoint reference or hash
- Stage B config reference or hash
- pseudo-tube manifest reference or hash
- stitching policy reference or hash
- DINO feature extraction settings
- protocol reference or hash

## 6. Change policy
Do not change these contracts in the mainline unless:
1. the current gate explicitly requires it,
2. this file is updated, and
3. related tests or contract checks are updated in the same change.
