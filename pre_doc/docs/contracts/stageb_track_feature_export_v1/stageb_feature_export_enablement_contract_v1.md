# Stage B Feature Export Enablement Contract v1

## 1) Scope and purpose
This contract defines the minimum upstream Stage B feature-export artifacts required to unblock bridge-input construction for real run15-style outputs where embeddings are currently absent.

This is an upstream enablement contract only. It specifies what Stage B MUST emit so downstream parser/builder tasks can construct normalized bridge-input records deterministically.

Non-goals:
- No runner implementation details.
- No parser algorithm design.
- No redefinition of existing export artifact schema v1.

## 2) Evidence/context basis (why this contract is needed)
Evidence anchors:
- `docs/contracts/stageb_track_feature_export_v1/bridge_gap_resolution_for_real_stageb_to_bridge_input_v1.md`
- `docs/contracts/stageb_track_feature_export_v1/bridge_from_stageb_outputs_v1_field_bindings_real_stageb_run15.md`

Run15 findings that motivate this contract:
- Existing aggregated outputs expose `video_id`, `score`, `category_id`, `segmentations`.
- Per-track embedding vectors are not present in run15-style outputs.
- Concrete per-track `track_id` and per-video runtime markers are not consistently present.
- Gap-resolution D3 conclusion is `BLOCKED_BY_UPSTREAM`: bridge-ready embedding production is blocked without upstream artifact enablement.

## 3) Relationship to existing contracts
This contract is additive and upstream-facing:
- It enables production of upstream artifacts consumed by bridge/parser tasks.
- It does not replace bridge mapping rules in `bridge_from_stageb_outputs_v1.md`.
- It does not replace D1-D4 decisions in `bridge_gap_resolution_for_real_stageb_to_bridge_input_v1.md`.
- It does not replace export schema definitions in `schema_track_feature_export_v1.md`.

Semantic boundary (`FROZEN`):
- Downstream normalized bridge-input `runtime_status` token domain remains `success|failed` only.
- Downstream `unprocessed` remains a split-domain reconciliation/video-status outcome, not an upstream token in this contract.

## 4) E1 Artifact topology and placement
Recommended v1 topology (`DEFAULT_FOR_V1`):

| Level | Path pattern (relative to Stage B run root) | File type | Required? | Notes |
|---|---|---|---|---|
| Run manifest | `d2/inference/feature_export_v1/manifest.json` | JSON | Yes | Single per-run entrypoint with run metadata and shard index |
| Per-video shard | `d2/inference/feature_export_v1/videos/<video_id>.json` | JSON | Yes | One record per video with track entries and runtime evidence |
| Optional compact index | `d2/inference/feature_export_v1/video_index.json` | JSON | No | Optional acceleration/index copy; MUST be consistent with manifest |

Naming/versioning (`FROZEN`):
- Directory name MUST include `feature_export_v1`.
- Manifest MUST include `contract_name="stageb_feature_export_enablement_contract_v1"` and `contract_version="v1"`.

Minimum required to unblock upstream implementation:
- Emit `manifest.json`.
- Emit per-video shard files listed in manifest.
- Include required fields from E2/E3/E4/E5.

## 5) E2 Minimal data model (run/video/track)

### 5.1 Run-level manifest fields
| Field | Type | Required? | Description |
|---|---|---|---|
| `contract_name` | string | Yes | Must equal `stageb_feature_export_enablement_contract_v1` |
| `contract_version` | string | Yes | Must equal `v1` |
| `run_id` | string | Yes | Stable identifier for producing run |
| `split` | string | Yes | Dataset split name used for this export |
| `embedding_dim` | integer | Yes | Positive dimension used by all track embeddings in the run |
| `embedding_dtype` | string | Yes | Must be `float32` for v1 expectations |
| `embedding_normalization` | string | Yes | Allowed values in E3 |
| `video_shards` | array[string] | Yes | Relative paths to per-video shard files |
| `created_at_utc` | string | No | ISO-8601 timestamp |
| `notes` | string | No | Human-readable run notes |

### 5.2 Per-video shard fields
| Field | Type | Required? | Description | Missing/error behavior |
|---|---|---|---|---|
| `video_id` | string or integer | Yes | Video identifier used by downstream binding | Missing/empty is invalid |
| `runtime_evidence` | object | Yes | Upstream evidence used to support downstream `runtime_status` inference | Missing object is invalid |
| `tracks` | array[object] | Yes | Per-track export records; MAY be empty | Missing array is invalid |
| `export_warnings` | array[string] | No | Optional non-fatal diagnostics | Optional |

### 5.3 Per-track fields
| Field | Type | Required? | Description | Constraints |
|---|---|---|---|---|
| `track_id` | string or integer | Yes | Upstream track identifier if available; deterministic synthetic allowed by downstream policy when absent in older sources | Must be non-empty |
| `embedding` | array[number] | Yes | Track feature vector payload | Length=`embedding_dim`; finite; dtype expectation in E3 |
| `embedding_normalization` | string | Yes | Track-level normalization tag | Allowed values in E3; SHOULD match run-level default |
| `start_frame_idx` | integer | No | Optional temporal provenance | If present, >=0 |
| `end_frame_idx` | integer | No | Optional temporal provenance | If present, >= `start_frame_idx` |
| `num_active_frames` | integer | No | Optional support cardinality | If present, >0 |
| `objectness_score` | number | No | Optional quality/confidence signal | If present, finite |

## 6) E3 Embedding semantics and numeric contract
Representation form (`DEFAULT_FOR_V1`):
- `embedding` is stored as a JSON numeric array in each per-track record.

Numeric contract (`FROZEN` unless noted):
- All embedding values MUST be finite (`NaN`/`Inf` forbidden).
- Run-level `embedding_dtype` MUST be `float32`.
- All tracks in one run MUST have identical embedding length equal to manifest `embedding_dim`.
- `embedding_dim` MUST be an integer > 0.

Normalization metadata (`FROZEN`):
- Required field: `embedding_normalization`.
- Allowed values: `none`, `l2`.
- Manifest-level `embedding_normalization` defines run default.
- Track-level `embedding_normalization` MUST be present and SHOULD match manifest value.

Zero-vector policy (`FROZEN`):
- Zero vectors are allowed.
- If normalization is `l2`, zero vectors MUST remain unchanged (no divide-by-zero repair).

## 7) E4 Runtime-status evidence model
Purpose:
- Provide upstream evidence sufficient for downstream parser/builder to infer normalized `runtime_status` (`success|failed`) without introducing upstream `unprocessed` tokens.

Required runtime evidence object (`DEFAULT_FOR_V1` with explicit limits):

| Field | Type | Required? | Allowed values | v1 intent |
|---|---|---|---|---|
| `stageb_completion_marker` | string | Yes | `completed`, `failed`, `unknown` | Upstream per-video completion evidence |
| `evidence_source` | string | Yes | free-form non-empty | Where marker came from (artifact/log key) |
| `evidence_confidence` | string | No | `explicit`, `inferred` | Strength of evidence |
| `failure_reason` | string | No | free-form non-empty | Recommended when marker is `failed` |

Inference boundary statements (`FROZEN`):
- Upstream contract MUST NOT define or emit `unprocessed` as a runtime token.
- Downstream parser MAY infer `runtime_status=success` for observed video records with evidence `completed`.
- Downstream parser SHOULD emit `runtime_status=failed` only when explicit failure evidence is available.
- If explicit per-video failure evidence is not emitted in a v1 run, `failed` coverage is limited; this remains `OPEN` for stronger upstream guarantees.

## 8) E5 Provenance and traceability
Minimum provenance fields (`FROZEN`):

| Level | Field | Type | Required? | Purpose |
|---|---|---|---|---|
| Run manifest | `stageb_checkpoint_ref` | string | Yes | Identifies checkpoint source |
| Run manifest | `stageb_checkpoint_hash` | string | Yes | Reproducibility guard for checkpoint |
| Run manifest | `stageb_config_ref` | string | Yes | Identifies config source |
| Run manifest | `stageb_config_hash` | string | Yes | Reproducibility guard for config |
| Run manifest | `pseudo_tube_manifest_ref` | string | Yes | Links export to pseudo-tube manifest |
| Run manifest | `pseudo_tube_manifest_hash` | string | Yes | Reproducibility guard for pseudo-tube manifest |
| Run manifest | `extraction_settings` | object | Yes | Frame sampling/pooling/min-track settings used |
| Video shard | `source_artifacts` | object | No | Optional per-video trace to original outputs |

Provenance sufficiency statement:
- The required run-level provenance above is the minimum v1 set needed to satisfy downstream reproducibility/traceability obligations.

## 9) E6 Compatibility and boundary statements
Compatibility (`FROZEN`):
- This contract supports upstream artifact production needed to unblock downstream parser/builder work.
- This contract does not redefine bridge status taxonomy or split-domain reconciliation.
- This contract does not replace bridge input contract semantics (P3.1a/P3.1b lineage).
- This contract does not replace export artifact schema v1 and Stage C consumer requirements.

Boundary (`FROZEN`):
- Upstream evidence fields in this contract are inputs to downstream inference logic; they are not a replacement for downstream bridge fields.
- `unprocessed` remains downstream reconciliation output only.

## 10) Decision summary table
| Topic | Label | v1 choice | Rationale | Impact |
|---|---|---|---|---|
| E1 topology | `DEFAULT_FOR_V1` | Manifest + per-video shards under `d2/inference/feature_export_v1/` | Concrete and minimally sufficient for implementation | Unblocks upstream emitter task with deterministic layout |
| Contract identity/version keys | `FROZEN` | `contract_name` and `contract_version` mandatory | Avoids ambiguous artifact interpretation | Stable parser/builder entry checks |
| E2 core required fields | `FROZEN` | Required run/video/track minima in Section 5 | Needed for deterministic bridge-input construction | Prevents partial/ambiguous upstream outputs |
| E3 numeric invariants | `FROZEN` | finite values, float32 expectation, consistent dimension, allowed normalization values | Ensures downstream numeric safety | Prevents invalid embedding payloads |
| E4 runtime token boundary | `FROZEN` | Downstream `runtime_status` stays `success|failed`; no upstream `unprocessed` token | Preserves D2 semantic boundary | Avoids taxonomy collapse/regression |
| E4 failed-evidence guarantee strength | `OPEN` | Explicit failed evidence recommended but not guaranteed in every v1 run | Current run15-style evidence gaps remain | May limit `failed` inference coverage in early v1 |
| D3 blocker disposition | `BLOCKED_BY_UPSTREAM` | Run15-style outputs alone remain insufficient for embeddings | Aligned with gap-resolution D3 conclusion | Necessitates this upstream enablement contract |

## 11) Recommended next-task decomposition
1. Upstream implementation task (first, required).
- Implement production of `feature_export_v1` manifest + per-video shards.
- Enforce required fields and E3 numeric contract at export time.
- Emit E4 runtime evidence markers where available.

2. Downstream parser/builder task (second).
- Consume new upstream artifacts plus existing Stage B outputs.
- Infer normalized bridge-input `runtime_status` in `success|failed` domain only.
- Keep `unprocessed` exclusively in split-domain reconciliation stage.

3. Integration validation task (third).
- Validate end-to-end bridge construction using real-run artifacts.
- Confirm provenance completeness and embedding consistency checks.
- Confirm boundary behavior for `runtime_status` vs `unprocessed` remains intact.
