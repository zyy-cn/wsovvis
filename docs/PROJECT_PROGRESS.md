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

## Current Stage

### Current project status
- ✅ P1 completed (protocol v2 baseline: `uniform` + `long_tail`)
- ✅ P3-Prep completed (schema v1 export contract)
- ✅ P3 core completed (producer + validator + tests)
- ▶️ Next recommended step: **P3.1a (Bridge Spec)** — define real Stage B output → exporter input bridge contract

### Why P3.1a next
The exporter core is now stable. The next risk is integration ambiguity between actual Stage B outputs and the exporter’s task-local input representation. A bridge spec reduces integration refactors before implementing a real adapter/export bridge.


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
