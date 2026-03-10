# WSOVVIS V9 Stage B and Artifact Transition Contract

This file preserves the essential contract discipline from the current automation layer while extending it toward v9.
It is authoritative for the minimum structure-side artifacts needed by the v9 mainline.

## 1. Scope
This contract covers the minimum transition from the current Stage B ecosystem to the v9 artifact stack:
- local tracklet export
- clip-level global track bank
- track semantic cache
- provenance requirements needed for reproducibility
- evidence/provenance hooks needed for worked examples

## 2. Local tracklet artifact form
Required v9-oriented form:
- split-level manifest + per-video payload files
- JSON metadata + NPZ arrays or equivalent structured cache
- identity preserved at least by `(video_id, local_track_id)`

Required per-tracklet payload at minimum:
- `local_track_id`
- `row_index`
- window / clip provenance
- temporal support fields
- masks or mask references sufficient for downstream stitching
- local query feature or equivalent linking-side representation
- local mask quality / objectness-related fields when available

Evidence/provenance additions strongly preferred:
- stable identifiers for worked-example selection
- direct references to the source pseudo-tube or pseudo-instance record when available

## 3. Global track bank artifact form
Required invariants:
- primary key is `(video_id, global_track_id)`
- `global_track_id` uniqueness is video-local
- global track membership over local tracklets is deterministic
- per-frame or temporal support information is sufficient for crop/feature extraction
- zero-track videos remain explicit in the manifest

Evidence/provenance additions strongly preferred:
- merge-membership trace or equivalent logable structure for worked examples
- score-matrix or matching-summary sidecar when available

## 4. Track semantic cache form
Required per-global-track payload at minimum:
- `global_track_id`
- `z_tau` (or reserved field for it before the extractor is active)
- `o_tau` or sufficient components to derive it
- temporal support fields
- crop/pooling provenance needed for reproducibility

Evidence/provenance additions strongly preferred:
- worked-example fields or sidecars for crop boxes, visible-frame indices, and aggregation weights

## 5. Provenance requirements
Run-level metadata should preserve at least:
- Stage B checkpoint reference/hash
- Stage B config reference/hash
- pseudo-tube manifest reference/hash
- stitching policy reference/hash
- DINO feature extraction settings
- protocol reference/hash

## 6. Change policy
Do not change these contracts in the mainline unless:
1. the current gate explicitly requires it,
2. this file is updated, and
3. related tests or contract checks are updated in the same change.
