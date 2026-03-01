# WSOVVIS Project Progress

This document tracks project-level implementation milestones and next steps.
For long-lived Codex workflow rules and Git/remote-verify operational guidance, see:
- `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
- `codex/specs/*`

---

## 2026-02-28 — P1 Protocol v2 completed

### Scope
WS-OVVIS labelset protocol tooling (protocol-side preprocessing)

### Completed
- Added `long_tail` missing protocol support to `tools/build_wsovvis_labelset_protocol.py`
- Preserved `uniform` protocol behavior compatibility
- Expanded manifest statistics (protocol metadata + class-level stats + missing summaries)
- Expanded tests from v1 baseline coverage to include long-tail behavior and determinism checks

### Validation
- Canonical remote validation passed on `gpu4090d`
- Targeted pytest completed successfully for protocol tooling tests

### Notes
- P1 protocol tooling is now suitable for downstream experimental protocol generation (uniform + long-tail v2 baseline)

---

## 2026-02-28 — P3-Prep completed (Stage B export schema v1, doc-only)

### Scope
Define a stable Stage B track feature export contract before exporter implementation

### Completed
- Authored Stage B export schema v1 contract docs and promoted canonical copies under `docs/contracts/stageb_track_feature_export_v1/`
- Included `consumer_requirements_stage_c.md` (Stage C consumer-side requirements) in the canonical contract set
- Included `sample_manifest_stub.json` (manifest example/stub) in the canonical contract set
- Locked v1 schema decisions (pooled per-track embeddings, split-manifest + per-video payloads, `(video_id, track_id)` identity, zero-track handling, JSON+NPZ serialization)

### Validation
- Doc-only task (human review)

### Canonical contract location
- `docs/contracts/stageb_track_feature_export_v1/`

### Notes
- This removed major ambiguity for Stage B → Stage C integration and enabled focused producer/validator implementation.

---

## 2026-02-28 — P3 core completed (Stage B track feature export v1 producer/validator/tests)

### Scope
Implement schema v1 export core and validation tooling for Stage B track feature exports

### Completed
- Implemented Stage B track feature export v1 producer core (manifest + per-video JSON + NPZ export flow)
- Implemented export artifact validator (schema/order/alignment/status consistency checks)
- Added focused tests for `tests/test_stageb_track_feature_export_v1.py`
- Integrated Codex-managed Git flow with sandbox push routing via `github-via-gpu`
- Completed canonical remote validation workflow with commit-consistency verification

### Validation
- Canonical remote validation passed on `gpu4090d`
- Target: `tests/test_stageb_track_feature_export_v1.py`
- Result: `10 passed`
- Passing run validated intended commit consistency (remote HEAD matched local/pushed target commit)

### Key references
- Implementation branch: `codex/p3-stageb-track-export-v1-core`
- Passing intended commit observed in workflow logs: `04633059...` (see task execution output/logs)

### Notes
- Some initial test assertions were refined after remote validation feedback; final result remained within task scope (test robustness fixes, no scope creep).

---

## 2026-02-28 — Workflow hardening update (long-term reference refactor)

### Scope
Reduce repeated impl prompt content by introducing reusable policy specs and stabilizing workflow references

### Completed
- Strengthened `codex/WSOVVIS_CODEX_WORKFLOW_README.md` as long-term workflow reference
- Added reusable specs under `codex/specs/` for:
  - Git-managed impl policy
  - canonical remote validation policy
  - doc-only / impl prompt contract
- Established layered prompt strategy:
  - workflow/common (global)
  - specs (reusable execution policy)
  - task-local docs/prompts (task-specific constraints only)

### Notes
- Future impl prompts can be substantially shorter while preserving stability and execution quality.

---

## 2026-02-28 — P3.1a completed (Bridge Spec)

### Scope
Define and freeze the bridge contract from real Stage B outputs to the existing exporter producer-input semantics (doc-only), without redefining export schema v1.

### Completed
- Authored canonical bridge spec under `docs/contracts/stageb_track_feature_export_v1/bridge_from_stageb_outputs_v1.md`
- Added explicit mapping semantics for real Stage B outputs -> producer input fields
- Documented track feature semantics, video status classification, split-domain reconciliation, provenance mapping, and error-handling/observability expectations
- Completed an r1 doc patch to remove contract ambiguities (`failed/unprocessed` + `tracks`, requiredness wording, bridge-vs-producer validation boundary)
- Synchronized frozen default decisions into the bridge contract (normalization default, duplicate result policy, non-finite handling)

### Validation
- Doc-only task (human review + targeted patch review)

### Notes
- This closed the largest integration ambiguity between actual Stage B outputs and the exporter/validator path, and enabled adapter implementation.

---

## 2026-02-28 — P3.1b completed (normalized bridge-input adapter + CLI/tests)

### Scope
Implement the bridge adapter that converts normalized Stage-B bridge input payloads into the existing export producer-input domain, with CLI entry and tests.

### Completed
- Implemented bridge adapter core (`wsovvis/track_feature_export/bridge_from_stageb_v1.py`)
- Added builder/validation CLI flow for bridge-input-based export generation
- Added focused tests for bridge adapter semantics and contract enforcement
- Updated bridge contract docs to reflect frozen defaults (including runtime-status strictness and non-finite reject-only policy)
- Added a minimal negative test for invalid `embedding_normalization` during validation-recovery pass

### Validation
- Canonical remote validation passed on `gpu4090d` after workflow-compliant recovery
- Targeted pytest selector:
  - `tests/test_stageb_track_feature_export_v1.py`
  - `tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py`
- Result: `21 passed`
- Remote HEAD matched intended local/pushed commit (workflow-compliant canonical validation)

### Key references
- Canonical validation branch used in recovery: `codex/p3-stageb-track-export-v1-core`
- Passing intended commit observed in logs: `b9fc82c1...`

### Notes
- The primary recovery work was workflow/validation completion (not major code rework): local env lacked pytest/numpy, so canonical remote verification was used per workflow rules.

---

## 2026-02-28 — P3.1c.0 completed (real Stage B output inventory + field binding, run15)

### Scope
Inspect a real Stage B run (run15) and produce concrete field-binding documentation from real artifacts to bridge placeholders, without implementing parser code.

### Completed
- Collected remote evidence for real Stage B run15 artifact layout
- Authored task-local evidence appendix for inspection commands and observed structures
- Authored canonical field-binding doc:
  - `docs/contracts/stageb_track_feature_export_v1/bridge_from_stageb_outputs_v1_field_bindings_real_stageb_run15.md`
- Confirmed run-level metadata/config/checkpoint artifacts and aggregated prediction outputs
- Confirmed split-domain coverage reconstruction via dataset `val_json` vs observed prediction `video_id` set (run15 sample matched exactly in inspection)
- Identified key gap: run15-style outputs lacked direct per-track embeddings / explicit `track_id` / explicit per-video runtime status fields for bridge input construction

### Validation
- Inspection + doc task (no parser implementation)

### Notes
- This established evidence-backed field names/layouts and directly motivated the subsequent gap-resolution and upstream feature-export enablement work.

---

## 2026-02-28 — P3.1c.1-pre completed (bridge gap resolution, doc-first)

### Scope
Resolve or explicitly scope the remaining field-source gaps (especially embeddings) before implementing real Stage B + sidecar -> bridge-input builder.

### Completed
- Authored gap-resolution addendum:
  - `docs/contracts/stageb_track_feature_export_v1/bridge_gap_resolution_for_real_stageb_to_bridge_input_v1.md`
- Resolved/specified D1–D4 decision topics:
  - track identity derivation scope
  - runtime-status derivation scope
  - embedding source gap status
  - next-task decomposition
- Explicitly preserved the semantic boundary:
  - normalized bridge-input `runtime_status` token domain is `success|failed`
  - `unprocessed` is downstream reconciliation output, not an input token
- Marked embedding-source insufficiency for current run15-style outputs as an upstream blocker, motivating a dedicated enablement contract/implementation path

### Validation
- Doc-only task (human review)

### Notes
- This step converted the previous ambiguity into a clean, executable task split (upstream feature-export enablement first, parser/builder next).

---

## 2026-02-28 — P3.1d.0 completed (Stage B feature-export enablement contract v1, doc-only)

### Scope
Define the upstream feature-export enablement contract needed to unblock real Stage B -> normalized bridge-input building.

### Completed
- Authored canonical enablement contract:
  - `docs/contracts/stageb_track_feature_export_v1/stageb_feature_export_enablement_contract_v1.md`
- Defined v1 artifact topology and placement, minimal run/video/track data model, embedding numeric contract, runtime-evidence model, provenance minima, and compatibility boundaries
- Preserved downstream semantic boundaries (including `runtime_status` vs `unprocessed`) and bridge/export contract alignment

### Validation
- Doc-only task (human review)

### Notes
- This contract enabled a staged upstream implementation path (fixture-first writer, then real-run integration) without reopening bridge/export semantics.

---

## 2026-03-01 — P3.1d.1 completed (feature-export enablement writer/CLI/tests, fixture-validated)

### Scope
Implement the upstream feature-export enablement writer path and CLI per contract v1, with focused tests and canonical validation, using fixture/synthetic evidence first.

### Completed
- Implemented upstream writer:
  - `wsovvis/track_feature_export/feature_export_enablement_v1.py`
- Added CLI entry:
  - `tools/build_stageb_feature_export_enablement_v1.py`
- Added focused tests:
  - `tests/test_stageb_feature_export_enablement_v1.py`
- Updated exports in `wsovvis/track_feature_export/__init__.py`
- Enforced contract-critical behaviors in writer/tests:
  - explicit `track_id` requirement
  - `unprocessed` runtime-evidence boundary rejection
  - finite embedding checks (`NaN/Inf` hard-fail)
  - normalization mode validation (`none|l2`)
  - deterministic ordering

### Validation
- Canonical remote validation passed on `gpu4090d`
- Branch: `codex/p3-1d1-stageb-feature-export-enablement-v1`
- Passing intended commit observed in logs: `8a7a141f...`
- Targeted pytest: `tests/test_stageb_feature_export_enablement_v1.py`
- Result: `6 passed`
- Remote HEAD matched intended local/pushed commit

### Notes
- Sample artifact evidence in this step was fixture-only/synthetic (not yet real-run-derived); real-run integration was deferred to P3.1d.2.

---

## 2026-03-01 — P3.1d.2 completed (real-run Stage B feature-export integration)

### Scope
Integrate the P3.1d.1 feature-export writer into the real Stage B/SeqFormer runner path and produce real-run-derived sidecar artifacts.

### Completed
- Exposed/propagated query embeddings through the real runner path
- Implemented deterministic query-to-track binding to emit explicit `track_id` + embedding payloads for sidecar export
- Persisted `track_id` / embedding / metadata in evaluator output and invoked the feature-export writer from real runner execution
- Produced real-run-derived `feature_export_v1` sidecar artifacts on the canonical remote runner environment

### Validation
- Canonical remote execution/validation completed on `gpu4090d` with correct branch/commit matching
- Branch: `codex/p3-1d2-stageb-feature-export-real-run-integration-v1`
- Passing intended commit observed in logs: `c194ae6...`
- Real-run artifact evidence confirmed (not fixture-only)

### Notes
- This removed the upstream embedding/source blocker identified in P3.1c.1-pre and made the real bridge-input builder task executable.

---

## 2026-03-01 — P3.1c.1 completed (real Stage B + sidecar -> normalized bridge-input builder + handoff)

### Scope
Implement the builder that joins real Stage B predictions with real feature-export sidecar artifacts and constructs normalized bridge-input consumable by the P3.1b adapter, then validate end-to-end handoff through export/validator on a real sample.

### Completed
- Implemented real Stage B + sidecar -> normalized bridge-input builder (with CLI/invocation path)
- Added deterministic join/mapping logic (including explicit mismatch handling and runtime-status token-domain preservation)
- Added/updated targeted tests for join behavior, runtime-status domain, and embedding/metadata propagation
- Validated downstream handoff chain on a real run18 sample:
  - builder -> P3.1b adapter -> export builder -> export validator
- Achieved successful validator result (`Validation OK`) on real-sample handoff output

### Validation
- Canonical remote validation/execution passed on `gpu4090d` with branch/commit matching
- Targeted tests (builder + relevant bridge/export tests) passed remotely: `16 passed`
- Real run18 input path used for handoff validation (sidecar under `runs/wsovvis_seqformer/18/d2/inference/feature_export_v1`)

### Observed real-sample notes
- Small-sample handoff validation used a deterministic subset (`--sample-video-limit` path)
- Example run summary observed in logs included `runtime_status_counts={success:10, failed:0}` and nonzero dropped-track filtering (`dropped_tracks=4`)

### Notes
- This closes the engineering bridge chain from real Stage B outputs to export artifact validation and unlocks full-split hardening/QA plus Stage C consumer work.

---

## 2026-02-28 — Workflow/spec cold-start hardening (docs)

### Scope
Improve document-only cold-start reliability for future sessions using `codex/common/*`, `codex/specs/*`, and workflow docs.

### Completed
- Added `codex/START_HERE.md` as the workflow entry/index with reading order and precedence rules
- Clarified docs-only bundle limitations (workflow understanding vs executable operations)
- Strengthened workflow README with canonical example values and explicit scope/precedence pointers
- Added inheritance statements to reusable specs to reduce ambiguity and duplicated interpretation
- Updated minimal impl prompt template to reference `codex/START_HERE.md`

### Validation
- Human review (doc-only refinement)

### Notes
- Goal: improve cold-start reproducibility and reduce prompt ambiguity in new sessions.

---

## Current Stage

### Current project status
- ✅ P1 completed (protocol v2 baseline: `uniform` + `long_tail`)
- ✅ P3-Prep completed (schema v1 export contract)
- ✅ P3 core completed (producer + validator + tests)
- ✅ P3.1a completed (Bridge Spec + r1 patch, doc-only)
- ✅ P3.1b completed (normalized bridge-input adapter + CLI/tests + canonical remote validation)
- ✅ P3.1c.0 completed (real Stage B inventory + field binding, run15)
- ✅ P3.1c.1-pre completed (gap resolution, doc-first)
- ✅ P3.1d.0 completed (feature-export enablement contract v1, doc-only)
- ✅ P3.1d.1 completed (feature-export writer/CLI/tests, fixture-validated)
- ✅ P3.1d.2 completed (real-run feature-export integration)
- ✅ P3.1c.1 completed (real Stage B + sidecar -> normalized bridge-input builder + end-to-end handoff validation)
- ▶️ Next recommended step: **P3.1c.2 (full-split hardening + QA for the real bridge chain)**

### Why P3.1c.2 next
The real bridge chain is now functionally complete on a small real sample (including downstream export/validator handoff). The next risk is production robustness and observability on larger/full-split runs: mismatch diagnostics, dropped-track reason accounting, and repeatable QA summaries before Stage C experiments consume bridge exports at scale.

### Follow-on after P3.1c.2
- Stage C loader / consumer data-plane implementation for export artifact v1
- Stage C attribution MVP (recommended order: MIL baseline -> EM baseline -> OT/Sinkhorn mainline)

