# Bridge Spec v1: Real Stage B Outputs -> Export Producer Input Semantics

## 1) Scope Boundary and Authority
This document defines the **upstream bridge mapping contract** from real Stage B runtime outputs into the existing Stage B export producer input semantics used by:
- `tools/build_stageb_track_feature_export_v1.py`
- `wsovvis/track_feature_export/v1_core.py`

This document **does not redefine** the export artifact schema or Stage C consumer behavior. Those remain authoritative in:
- `docs/contracts/stageb_track_feature_export_v1/schema_track_feature_export_v1.md`
- `docs/contracts/stageb_track_feature_export_v1/consumer_requirements_stage_c.md`

What this bridge spec adds:
- accepted Stage B-side input forms for bridge v1
- deterministic field-level mapping/derivation rules to producer input
- deterministic video status classification and split reconciliation rules
- provenance extraction and fallback policy
- adapter obligations for type/order consistency before producer handoff

Out of scope:
- export artifact schema redesign
- Stage C algorithm/loss/training design
- code implementation details of future adapter

Bridge vs producer validation boundary (`FROZEN`):
- This bridge contract defines pre-handoff obligations only: split-domain reconciliation, deterministic status classification, required producer-field population, and bridge hard-fail categories in Sections 7-11.
- Producer/schema validation may re-check overlapping constraints as defense-in-depth.
- Nothing in this document weakens or expands the existing producer/schema constraints.

## 2) Decision Labels Used in This Doc
- `FROZEN`: fixed for bridge v1
- `DEFAULT_FOR_V1`: bounded v1 default; may be revisited in later versions
- `OPEN`: unresolved; future adapter task must either resolve or explicitly narrow scope

## 3) Bridge Input Domain (Real Stage B Side)

### 3.1 Accepted input forms in bridge v1
Bridge v1 accepts a per-split collection of Stage B outputs represented by these semantic objects (exact field names may differ by runtime implementation):
- `split_domain_manifest`: authoritative split-domain video ID set
- `stageb_video_results[*]`: one per video attempted by Stage B (may be absent for unprocessed videos)
- `stageb_video_results[*].tracks[*]`: per-track summaries (IDs, temporal span, score, feature source references)
- `stageb_video_results[*].failure_info` or equivalent execution/error summary
- provenance sources: checkpoint metadata, config metadata, pseudo-tube manifest metadata, extraction settings metadata

### 3.2 Source placeholder convention
If exact runtime keys are not frozen, this spec uses abstract placeholders such as:
- `stageb_video_result.video_id`
- `stageb_video_result.tracks[*].track_id`
- `stageb_video_result.tracks[*].track_feature_vector`
- `stageb_video_result.runtime_status`

These placeholders are semantic and must be mapped to concrete keys by adapter implementation.

### 3.3 Bridge-input requiredness by semantic source
| Semantic source | Expected location (abstract) | v1 requirement | Missing behavior |
|---|---|---|---|
| Split domain video list | `split_domain_manifest.video_ids[]` | `REQUIRED` (`FROZEN`) | Hard-fail (cannot classify `unprocessed` deterministically) |
| Split name | `split_domain_manifest.split` or adapter arg | `REQUIRED` (`FROZEN`) | Hard-fail |
| Stage B results collection/index source | `stageb_video_results` indexed by `video_id` | `REQUIRED` (`FROZEN`) | Hard-fail if collection/index source unavailable |
| Per-video Stage B result record (split-domain video) | `stageb_video_results[video_id]` | `OPTIONAL` (`FROZEN`) | Missing record -> classify `unprocessed` |
| Per-track core metadata | `stageb_video_result.tracks[*]` | `REQUIRED` for processed tracks (`FROZEN`) | Invalid track rows dropped only if recoverable policy allows; else video `failed` |
| Track feature source | `stageb_video_result.tracks[*].track_feature_vector` or derivable frame-level features | `REQUIRED` (`DEFAULT_FOR_V1`) | If unavailable for a video with otherwise processed tracks -> video `failed` |
| Provenance checkpoint/config/manifest metadata | run metadata / sidecar metadata | `REQUIRED` (`FROZEN`) | Hard-fail if required producer fields cannot be populated |
| Failure diagnostics | `stageb_video_result.failure_info` | `RECOMMENDED` (`DEFAULT_FOR_V1`) | If absent, still classify via deterministic status rules |
| Optional calibration/auxiliary scores | stageb optional fields | `UNAVAILABLE_IN_V1` (`FROZEN`) | Ignored by bridge v1 |

## 4) Canonical Mapping Target (Producer Input Semantics)
Bridge output target is the existing producer input semantic shape (not necessarily a public file format), with required fields summarized here by reference to existing schema/producer contracts:

Top-level required fields:
- `split`
- `embedding_dim`
- `embedding_dtype` (`float32`)
- `embedding_pooling` (`track_pooled`)
- `embedding_normalization` (`none|l2`)
- `producer`
- `videos[]`

Per-video required fields:
- `video_id`
- `status`
- `tracks[]` only for processed statuses

Per-track required fields:
- `track_id`
- `start_frame_idx`
- `end_frame_idx`
- `num_active_frames`
- `objectness_score`
- `embedding` (length `embedding_dim`)

## 5) Required Mapping Table (All Producer Input Fields)

### 5.1 Top-level mapping
| Target field | Producer requiredness | Source/derivation | Transform rule | Decision status | Missing/invalid behavior |
|---|---|---|---|---|---|
| `split` | `REQUIRED` | `split_domain_manifest.split` (preferred) else adapter input arg | Normalize to non-empty string; must match producer provenance split | `FROZEN` | Hard-fail |
| `embedding_dim` | `REQUIRED` | Derived from first valid track embedding length across split | Must be integer > 0; all processed-track embeddings must match | `FROZEN` | Hard-fail on empty-dim or mismatch |
| `embedding_dtype` | `REQUIRED` | Constant | Set to `float32` | `FROZEN` | Hard-fail only if adapter cannot cast |
| `embedding_pooling` | `REQUIRED` | Bridge pooling semantics outcome | Set to `track_pooled` | `FROZEN` | Hard-fail if semantics not track-pooled |
| `embedding_normalization` | `REQUIRED` | Adapter config or explicit bridge mode | `none` or `l2` only | `DEFAULT_FOR_V1` | Hard-fail on unsupported value |
| `producer` | `REQUIRED` | Provenance extraction rules (Section 9) | Populate full required block | `FROZEN` | Hard-fail on missing required field |
| `videos[]` | `REQUIRED` | Split-domain reconciliation over Stage B results | One record per split-domain video; sorted by `video_id` before handoff | `FROZEN` | Hard-fail on duplicate split-domain IDs |

### 5.2 Per-video mapping
| Target field | Producer requiredness | Source/derivation | Transform rule | Decision status | Missing/invalid behavior |
|---|---|---|---|---|---|
| `videos[*].video_id` | `REQUIRED` | Split domain video ID | Non-empty string | `FROZEN` | Hard-fail on empty/duplicate |
| `videos[*].status` | `REQUIRED` | Deterministic classifier from runtime evidence (Section 7) | One of 4 allowed statuses | `FROZEN` | Hard-fail if classifier preconditions unavailable |
| `videos[*].tracks` for all statuses | `CONDITIONAL_REQUIRED` | Valid mapped tracks for that video | Required non-empty for `processed_with_tracks`; required empty for `processed_zero_tracks`; optional for `failed`/`unprocessed`, and if present MUST be `[]` | `FROZEN` | Violation -> hard-fail before producer |

`tracks` presence note (`FROZEN`): this rule is for bridge pre-producer-handoff payload semantics. Producer-side canonicalization/validation may enforce equivalent or stricter shape checks downstream.

### 5.3 Per-track mapping
| Target field | Producer requiredness | Source/derivation | Transform rule | Decision status | Missing/invalid behavior |
|---|---|---|---|---|---|
| `videos[*].tracks[*].track_id` | `REQUIRED` | `stageb_track.track_id` | Keep original serialized type; must be int or non-empty string | `FROZEN` | Invalid type/empty -> track invalid; if any mixed type across split -> hard-fail |
| `videos[*].tracks[*].start_frame_idx` | `REQUIRED` | `stageb_track.start_frame_idx` or min active frame index | Cast to int >= 0 | `FROZEN` | Missing/negative -> track invalid |
| `videos[*].tracks[*].end_frame_idx` | `REQUIRED` | `stageb_track.end_frame_idx` or max active frame index | Cast to int >= start | `FROZEN` | Missing/<start -> track invalid |
| `videos[*].tracks[*].num_active_frames` | `REQUIRED` | `stageb_track.num_active_frames` or count(active frame indices) | Cast to int > 0 | `FROZEN` | Missing/nonpositive -> track invalid |
| `videos[*].tracks[*].objectness_score` | `REQUIRED` | `stageb_track.objectness_score` (preferred) else `stageb_track.confidence` | Cast numeric to float | `DEFAULT_FOR_V1` | Missing/non-numeric -> track invalid |
| `videos[*].tracks[*].embedding` | `REQUIRED` | Track feature semantics (Section 6) | Produce finite `float32` vector length `embedding_dim`; apply optional normalization policy | `DEFAULT_FOR_V1` | Missing/wrong shape/non-finite unresolved by policy -> track invalid or video `failed` per Sections 7 and 11 |

## 6) Track Feature Semantics

### 6.1 Source selection
- Preferred source: `stageb_track.track_feature_vector` if Stage B emits one vector per track.
- Fallback source: `stageb_track.frame_feature_vectors` pooled into one vector.
- If multiple candidate representations exist, `track_feature_vector` takes precedence.

Decision: source precedence is `DEFAULT_FOR_V1` (exact concrete key names remain `OPEN`).

### 6.2 Pooling semantics
When only frame/step-level vectors exist, use:
- `mean_over_active_frames` over valid active-frame vectors for each track.
- Active frame set must correspond to the same temporal support used for `num_active_frames`.

Decision: pooling rule is `DEFAULT_FOR_V1` (`mean_over_active_frames`).

### 6.3 Normalization semantics
- Allowed output normalization modes are `none` and `l2`.
- If mode is `l2`, apply after pooling and before final dtype cast writeout.
- Zero vector under `l2`: keep zero vector unchanged (do not divide by zero).

Decision (`DEFAULT_FOR_V1`): if upstream does not declare normalization, adapter default is `none`; handling rule is `FROZEN`.

### 6.4 dtype / NaN / Inf handling
- Final exported per-track embedding must be finite `float32`.
- Cast sequence: source numeric -> intermediate float -> normalization (if any) -> `float32`.
- NaN/Inf handling default: reject affected track as invalid; do not silently clip/replace.

Decision (`DEFAULT_FOR_V1`): reject-only on non-finite embeddings (no repair/clip/replace).

## 7) Deterministic Video Status Classification
Classification precedence is deterministic and must be applied in order:

1. If video ID is not in split domain: exclude from manifest domain; handle as extra output (Section 8).
2. If video ID is in split domain and no Stage B result exists: `unprocessed`.
3. If Stage B result indicates pipeline/runtime failure for that video: `failed`.
4. Else evaluate mapped valid tracks:
- count(valid_tracks) > 0 -> `processed_with_tracks`
- count(valid_tracks) == 0 and processing completed without runtime failure -> `processed_zero_tracks`

Clarification (`FROZEN`):
- `processed_zero_tracks` means Stage B successfully completed for the video but yielded no valid exportable tracks after required filtering/validation.
- `failed` means runtime failure or unusable output state prevented successful completion.

Partial/interrupted outputs:
- If runtime marks interrupted/error -> `failed`.
- If artifacts are partial but runtime success marker is absent/ambiguous -> classify `failed` (`DEFAULT_FOR_V1`).

## 8) Split Domain and Manifest Coverage Reconciliation

### 8.1 Manifest domain authority
Manifest video domain is exactly `split_domain_manifest.video_ids[]`.
Decision: `FROZEN`.

### 8.2 Missing Stage B outputs
For any split-domain video without Stage B result, include video with status `unprocessed`.
Decision: `FROZEN`.

### 8.3 Extra Stage B outputs (video not in split domain)
Do not include in manifest `videos[]`; log and count as `extra_video_results`.
Decision: `DEFAULT_FOR_V1`.

### 8.4 Duplicate handling
- Duplicate video IDs in split domain source: hard-fail.
- Duplicate Stage B result records for same video: hard-fail by default in v1 (no implicit supersession selection).
- Duplicate track IDs within a video after mapping: hard-fail.

### 8.5 Ordering prior to producer handoff
Bridge must hand off `videos[]` sorted lexicographically by `video_id`.
Track ordering may be left unsorted and rely on producer canonicalization, but bridge should prefer pre-sort for deterministic debugging.
Decision: handoff sorted videos is `FROZEN`; track pre-sort by bridge is `DEFAULT_FOR_V1`.

## 9) Provenance Extraction and Fallback Policy (`producer.*`)
Fail-fast is default for required provenance fields.

| Producer field | Preferred source | Fallback source | Decision status | Missing behavior |
|---|---|---|---|---|
| `producer.stage_b_checkpoint_id` | run metadata checkpoint ID | checkpoint path basename | `FROZEN` | Hard-fail |
| `producer.stage_b_checkpoint_hash` | explicit checkpoint hash metadata | recomputed hash of checkpoint artifact | `FROZEN` | Hard-fail |
| `producer.stage_b_config_ref` | run config reference | resolved config path string | `FROZEN` | Hard-fail |
| `producer.stage_b_config_hash` | config hash metadata | recomputed config content hash | `FROZEN` | Hard-fail |
| `producer.pseudo_tube_manifest_id` | pseudo-tube manifest metadata | adapter input arg | `FROZEN` | Hard-fail |
| `producer.pseudo_tube_manifest_hash` | pseudo-tube manifest hash metadata | recomputed hash of referenced manifest | `FROZEN` | Hard-fail |
| `producer.split` | split domain manifest split | top-level split field | `FROZEN` | Hard-fail if mismatch |
| `producer.extraction_settings.frame_sampling_rule` | Stage B runtime extraction config | adapter config | `FROZEN` | Hard-fail |
| `producer.extraction_settings.pooling_rule` | explicit applied pooling rule | derived from bridge mode (`track_feature_vector_direct` or `mean_over_active_frames`) | `DEFAULT_FOR_V1` | Hard-fail |
| `producer.extraction_settings.min_track_length` | Stage B runtime filtering config | adapter config | `FROZEN` | Hard-fail |

## 10) Ordering and Type-Consistency Obligations
Before producer handoff, bridge must enforce:
- split-wide `track_id` serialized type consistency (all int or all string): `FROZEN`
- no duplicate `video_id` in handoff payload: `FROZEN`
- no duplicate `track_id` within video: `FROZEN`
- each processed video uses only valid tracks meeting per-track field rules: `FROZEN`

Mixed `track_id` types across split:
- default action: hard-fail bridge run (do not coerce) (`DEFAULT_FOR_V1`).
- coercion policy remains `OPEN` for future versions.

## 11) Error Handling and Observability Contract

### 11.1 Hard-fail categories
- split domain source unavailable/invalid
- required provenance field unavailable or inconsistent
- split-level embedding dimension inconsistency
- split-wide track_id type inconsistency
- duplicate video IDs in authoritative split domain

### 11.2 Recoverable per-video categories (`DEFAULT_FOR_V1`)
- malformed track records may be dropped for that video only if runtime success exists and at least classification remains deterministic
- if recoverable drops lead to zero valid tracks and no runtime failure -> `processed_zero_tracks`
- if video becomes semantically ambiguous (cannot determine processed vs failed) -> classify `failed`

### 11.3 Required counters/log signals
Bridge implementation must emit at least:
- `total_split_videos`
- `processed_with_tracks_videos`
- `processed_zero_tracks_videos`
- `failed_videos`
- `unprocessed_videos`
- `extra_video_results`
- `duplicate_video_results`
- `malformed_track_rows_dropped`
- `videos_with_nonfinite_embeddings`

## 12) Synthetic Mapping Example

### 12.1 Stage-B-like input fragment (schematic)
```json
{
  "split_domain_manifest": {
    "split": "train",
    "video_ids": ["vid_a", "vid_b", "vid_c"]
  },
  "stageb_video_results": [
    {
      "video_id": "vid_a",
      "runtime_status": "ok",
      "tracks": [
        {
          "track_id": 11,
          "start_frame_idx": 5,
          "end_frame_idx": 14,
          "num_active_frames": 10,
          "objectness_score": 0.92,
          "track_feature_vector": [0.1, 0.0, 0.3, 0.4]
        }
      ]
    },
    {
      "video_id": "vid_b",
      "runtime_status": "ok",
      "tracks": []
    }
  ]
}
```

### 12.2 Resulting producer-input fragment (schematic)
```json
{
  "split": "train",
  "embedding_dim": 4,
  "embedding_dtype": "float32",
  "embedding_pooling": "track_pooled",
  "embedding_normalization": "none",
  "producer": {
    "stage_b_checkpoint_id": "ckpt_42",
    "stage_b_checkpoint_hash": "sha256:...",
    "stage_b_config_ref": "configs/stageb.yaml",
    "stage_b_config_hash": "sha256:...",
    "pseudo_tube_manifest_id": "ptubes_train_v2",
    "pseudo_tube_manifest_hash": "sha256:...",
    "split": "train",
    "extraction_settings": {
      "frame_sampling_rule": "uniform_stride_2",
      "pooling_rule": "track_feature_vector_direct",
      "min_track_length": 2
    }
  },
  "videos": [
    {
      "video_id": "vid_a",
      "status": "processed_with_tracks",
      "tracks": [
        {
          "track_id": 11,
          "start_frame_idx": 5,
          "end_frame_idx": 14,
          "num_active_frames": 10,
          "objectness_score": 0.92,
          "embedding": [0.1, 0.0, 0.3, 0.4]
        }
      ]
    },
    {
      "video_id": "vid_b",
      "status": "processed_zero_tracks",
      "tracks": []
    },
    {
      "video_id": "vid_c",
      "status": "unprocessed",
      "tracks": []
    }
  ]
}
```

Resulting status outcomes:
- `vid_a` -> `processed_with_tracks`
- `vid_b` -> `processed_zero_tracks`
- `vid_c` -> `unprocessed` (in split domain, no Stage B result)

## 13) Explicit Unresolved Decisions
- `OPEN`: exact concrete Stage B key paths/artifact filenames for each abstract placeholder.
- `OPEN`: whether future bridge version may coerce mixed-type track IDs to a canonical type.

## 14) Implementation Readiness Notes for Future Tier 2 Adapter
This bridge spec is intended to be sufficient for adapter implementation without re-deciding:
- producer-input required field coverage
- status classification precedence
- split reconciliation and provenance fail-fast behavior
- track feature pooling/casting baseline defaults

P3.1b frozen defaults now in force (`DEFAULT_FOR_V1`):
- Upstream-unspecified normalization defaults to `none`.
- Duplicate Stage B results hard-fail (no implicit supersession).
- Non-finite embedding policy is reject-only (no repair).
