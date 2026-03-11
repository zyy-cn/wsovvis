# Gap Resolution Addendum v1: Real Stage B run15 -> Bridge Input

## 1) Scope and purpose
This addendum resolves or explicitly scopes the field-source gaps identified from real Stage B run15 artifacts so the next implementation task can be defined without ambiguity.

Scope is limited to D1-D4 decisions for real run15-style inputs and alignment with:
- `docs/contracts/stageb_track_feature_export_v1/bridge_from_stageb_outputs_v1.md`
- `docs/contracts/stageb_track_feature_export_v1/bridge_from_stageb_outputs_v1_field_bindings_real_stageb_run15.md`
- `docs/contracts/stageb_track_feature_export_v1/stageb_feature_export_enablement_contract_v1.md` (upstream artifact contract that resolves D3 blocker path)

This addendum is doc-only and MUST NOT redefine frozen P3.1a/P3.1b bridge/export semantics.

## 2) Evidence basis summary (run15 facts only)
Evidence source: `codex/2026022808_p3_1c_1_pre_bridge_gap_resolution_doc_first_tier1/inputs/04_output.txt` and the run15 field-binding doc.

Observed facts from run15:
- Artifacts are aggregated outputs: `d2/inference/results.json` and `d2/inference/instances_predictions.pth`.
- `instances_predictions.pth` records expose `video_id`, `score`, `category_id`, `segmentations`.
- No concrete `track_id` field in inspected prediction records.
- No concrete per-track embedding vector field in inspected prediction records.
- No concrete per-video `runtime_status` field in inspected prediction records.
- Split-domain IDs can be sourced from `val_json -> videos[*].id`; run15 observation reported exact domain match (`837/837`, `missing=0`, `extra=0`).
- Run-level completion markers exist (`run.json.status = "COMPLETED"`, `d2/last_checkpoint = model_final.pth`).

## 3) Gap summary table (D1-D3)
| Topic | Evidence from run15 | Decision label | v1 rule / scope outcome |
|---|---|---|---|
| D1 `track_id` source | No concrete `track_id` field observed in prediction records | `DEFAULT_FOR_V1` | Deterministic synthetic `track_id` generation is allowed for next parser/builder task under strict stability/uniqueness constraints (Section 4). |
| D2 `runtime_status` derivation | No concrete per-video `runtime_status` field observed; run-level completion exists | `DEFAULT_FOR_V1` + `OPEN` | Parser may infer `runtime_status=success` for observed videos only. Evidence standard for `failed` remains open (Section 5). |
| D3 embedding source | No per-track embedding vectors observed in run15 artifacts | `BLOCKED_BY_UPSTREAM` | Parser/builder is blocked for bridge-ready track embeddings until upstream artifact/field enablement is added (Section 6). |

## 4) D1 resolution/scope: track identity derivation
Decision: `DEFAULT_FOR_V1`

Rules for next parser/builder task:
- Deterministic synthetic `track_id` generation MAY be used when upstream `track_id` is absent.
- Stability domain MUST be per input dataset/run artifact set and deterministic for repeated parses of identical inputs.
- Uniqueness domain MUST be within each `video_id` (no duplicate track IDs per video).
- Generated IDs MUST be reproducible from deterministic ordering/materialized fields (no random source, no wall-clock dependence).
- If future upstream artifacts provide explicit `track_id`, parser MUST prefer upstream `track_id` and MUST NOT overwrite it.

Constraint note:
- This is a source-derivation default for real run15-style parsing only; it does not relax producer/export requirements that each track record has a valid `track_id`.

## 5) D2 resolution/scope: runtime status derivation
Decision: `DEFAULT_FOR_V1` with `OPEN` evidence rule for `failed`

Critical semantic clarification (`FROZEN`):
- In normalized bridge-input domain, `runtime_status` token values are `success|failed` only.
- `unprocessed` is NOT a normalized `runtime_status` token.
- `unprocessed` is a downstream split-domain reconciliation/video-status classification outcome for videos in split domain that have no corresponding result record.

Rules for next parser/builder task:
- For observed videos (videos with parsed result records), parser MAY infer `runtime_status=success` when no per-video failure evidence exists.
- Parser MUST NOT emit `runtime_status=unprocessed`.
- Parser SHOULD emit `runtime_status=failed` only when explicit per-video failure evidence exists.

Evidence required to assert `failed` (`OPEN`):
- A concrete per-video failure marker in upstream result artifacts, or
- A deterministic upstream contract that maps specific artifact/log signals to a specific `video_id` failure outcome.

Current run15 implication:
- Run15 provides no concrete per-video failure token in prediction records; therefore parser-side `failed` inference is not evidence-backed for run15-style artifacts alone.
- Absent split-domain videos remain handled later by reconciliation as `unprocessed`, not by parser `runtime_status` token emission.

## 6) D3 resolution/scope: embedding source (highest priority)
Decision: `BLOCKED_BY_UPSTREAM`

Evidence-backed sufficiency statement:
- Run15-style artifacts are insufficient for required bridge track embeddings because no concrete per-track embedding vector source is present in observed outputs.

Impact:
- P3.1c.1 parser/builder that targets bridge-ready export inputs cannot be completed from run15-style artifacts alone.

Viable upstream enablement paths:
1. Stage B inference/export patch to emit per-video per-track bridge sidecar with explicit `track_id` and embedding vectors.
2. Stage B model-side hook to export track embeddings during inference into a deterministic artifact keyed by `video_id` + track key.
3. Offline post-process feature extraction path that deterministically reconstructs track embeddings aligned to Stage B track records.

Recommended v1 path:
- Path 1 (explicit Stage B bridge sidecar export) is recommended for v1 due to highest determinism and lowest parser ambiguity.

Minimal upstream additions needed for recommended path:
- Artifact containing per-video records keyed by `video_id`.
- Per-track fields: `track_id`, `embedding` (finite float32 vector), and sufficient temporal/score fields required by bridge mapping.
- Per-video runtime outcome field or equivalent deterministic failure evidence mapping (if `failed` support is required in parser stage).

## 7) D4 scope boundary + recommended decomposition
Decision: `FROZEN`

Next implementation decomposition:
1. Upstream enablement task first (`BLOCKED_BY_UPSTREAM` dependency removal)
- Define and implement Stage B-side embedding export artifact/fields (recommended path in Section 6).
- Include concrete field contract for per-video/per-track keys required by bridge parsing.

2. Parser/builder task second (P3.1c.1)
- Consume run artifacts + enabled upstream embedding artifact.
- Apply D1 deterministic `track_id` default only where upstream `track_id` is absent.
- Apply D2 rules: infer parser `runtime_status` only in `success|failed` domain; never emit `unprocessed` token.
- Leave split-domain reconciliation/classification (`processed_with_tracks|processed_zero_tracks|failed|unprocessed`) to downstream bridge/export stage per existing contract.

P3.1c.1 WILL do:
- Deterministic parsing/binding from concrete Stage B artifacts into normalized bridge-input records.
- Emit normalized `runtime_status` tokens in allowed domain.

P3.1c.1 WILL NOT do:
- Invent embedding vectors without an upstream source.
- Redefine status taxonomies from existing bridge contract.
- Collapse downstream `unprocessed` classification into parser token domain.

## 8) Decision summary by label
| Label | Decisions |
|---|---|
| `FROZEN` | D4 decomposition order (upstream enablement before parser/builder); D2 semantic boundary that normalized `runtime_status` is `success|failed` only and excludes `unprocessed`. |
| `DEFAULT_FOR_V1` | D1 synthetic deterministic `track_id` allowance (when upstream missing); D2 parser-side `success` inference for observed videos without failure evidence. |
| `OPEN` | D2 concrete evidence contract for per-video `failed` inference from run15-style outputs. |
| `BLOCKED_BY_UPSTREAM` | D3 embedding source gap for run15-style artifacts. |

## 9) Open prerequisites / upstream requirements
- Upstream must provide a concrete per-track embedding source artifact/field contract.
- If parser-side `failed` emission is required, upstream must provide per-video failure evidence that is deterministically mappable to `video_id`.

## 10) Recommended next task decomposition (actionable)
1. Create upstream feature-export enablement task pack.
- Define sidecar schema/field contract for `video_id` + per-track embedding payload.
- Implement export path in Stage B run pipeline.
- Provide one real-run artifact sample for contract verification.

2. Create parser/builder implementation task pack (P3.1c.1).
- Bind concrete artifacts to normalized bridge-input schema.
- Implement D1 deterministic synthetic `track_id` fallback.
- Implement D2 parser runtime-status inference (`success|failed` only) and explicit non-emission of `unprocessed`.

3. Create focused reconciliation validation task (post-parser).
- Verify downstream split-domain classification still yields `unprocessed` only via missing-result reconciliation.
