# Stage C Consumer Requirements for Stage B Track Feature Export v1

## 1) Required Inputs
For each split, Stage C must read:
- `manifest.v1.json`
- For each video with status `processed_with_tracks` or `processed_zero_tracks`:
  - `track_metadata.v1.json`
  - `track_arrays.v1.npz`

## 2) Required Validation Before Use
Stage C loader must hard-fail when any of the following is violated:
- `schema_name` mismatch
- unsupported major schema version
- missing required manifest/per-video/per-track fields
- invalid status/num_tracks/path consistency
- NPZ missing `embeddings` or `track_row_index`
- row alignment mismatch:
  - `tracks[i].row_index != i`
  - `track_row_index[i] != i`
  - `len(tracks) != embeddings.shape[0]`
- embedding shape mismatch with split-level `embedding_dim`

## 3) Identity and Join Rules
- Primary identity key is `(video_id, track_id)`.
- `track_id` is only video-local; Stage C must not treat it as globally unique.
- If Stage C creates a flat table, it must materialize both keys.

## 4) Zero-Track / Failure Semantics
- `processed_zero_tracks` videos are valid processed outputs and must remain visible to downstream accounting.
- `failed` and `unprocessed` videos are non-consumable for attribution and must not produce track candidates.
- Split-level counts should separately report:
  - processed_with_tracks videos
  - processed_zero_tracks videos
  - failed/unprocessed videos

## 5) Candidate Filtering Compatibility
Stage C may apply filtering after loading aligned rows, for example:
- `objectness_score >= threshold`
- `num_active_frames >= min_length`

Filtering must not reorder surviving rows in a way that breaks provenance traceability to source row indices.

## 6) v1 Assumptions for MIL/OT/EM
- Pooled per-track embedding vectors are the only required feature form.
- No per-frame sequence features are assumed in v1.
- Consumer code should be written to tolerate optional future fields while preserving current required path.

## 7) Future / Non-blocking Extensions (Not Required for v1)
- Optional geometry/mask references for richer cost terms.
- Optional unknown/background hints.
- Optional uncertainty vectors.
