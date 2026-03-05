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

## 2026-03-01 — P3.1c.2 completed (bridge full-split hardening + QA)

### Scope
Harden the real Stage B + sidecar -> normalized bridge-input builder QA/reporting path and validate the bridge-to-export handoff on larger real-run subsets.

### Completed
- Hardened QA summary schema/counter structure for deterministic machine-readable reporting
- Added richer join counters (prediction/sidecar totals, matched, missing counterparts, extras, duplicate-key diagnostics)
- Added reason-coded dropped-track counters for malformed/non-finite/invalid tracks
- Added runtime/video summaries (runtime status counts, per-video result counters, zero-track/unprocessed diagnostics)
- Preserved bridge-handoff compatibility and completed real-run validation chains on run18 subsets

### Validation
- Canonical remote validation/execution passed on `gpu4090d` with branch/commit matching
- run18 sample chain (`--sample-video-limit 10`) succeeded end-to-end
- run18 larger chain (`--sample-video-limit 200`) succeeded end-to-end
- Downstream export validator reported `Validation OK` for both sample and larger runs

### Key references
- Canonical validation branch: `codex/p3-1c2-bridge-fullsplit-hardening-qa`
- Passing intended commit observed in logs: `4a85ccfe...`

### Notes
- This milestone closes the P3.1c bridge-hardening phase and makes Stage C consumer/loader offline data-plane implementation the next recommended execution phase.

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

## 2026-03-01 — Stage C0 completed (consumer/loader offline data-plane)

### Scope
Implement and validate the Stage C0 offline consumer/loader over Stage B export artifact v1 with deterministic loading/indexing behavior and consumer-facing access APIs.

### Completed
- Implemented export artifact v1 consumer/loader path for manifest + shard consumption.
- Added deterministic manifest/shard loading behavior with stable iteration/indexing checks.
- Preserved strict scope boundaries (no attribution algorithm/training/Stage D additions).
- Preserved workflow branch/commit match discipline for canonical validation evidence.

### Validation
- Canonical remote validation passed on `gpu4090d` for Stage C0 loader tests.
- Branch used: `codex/stagec0-consumer-loader-offline-dataplane-v1`
- Canonical result: `10/10` targeted Stage C0 test suite PASS.
- Remote HEAD matched intended local/pushed commit (commit-consistent PASS evidence).

### Key references
- Stage C0 validation recovery PASS output: `codex/2026030105_progress_sync_and_stagec0_consumer_loader_taskpack_meta_tier1_2/10_output.txt`
- Intended/verified commit in PASS evidence: `9a45f554...`

### Notes
- Stage C0 is complete as the offline data-plane foundation for Stage C attribution work.
- Next recommended step is Stage C1 attribution baseline planning/implementation, starting with an offline MIL baseline over Stage C0 loader outputs.

---

## 2026-03-01 — Stage C1 completed (MIL-first offline baseline attribution)

### Scope
Implement and validate the first Stage C offline attribution baseline (MIL-first only) on top of Stage C0 loader APIs, including deterministic score output artifacts and baseline diagnostics.

### Completed
- Implemented Stage C1 offline MIL-first scoring path over Stage C0 split/track traversal.
- Added Stage C1 CLI/pipeline entrypoint for deterministic offline scoring runs.
- Added focused Stage C1 tests for nominal path, determinism, ordering/identity guarantees, and C1-boundary failure cases.
- Produced and validated Stage C1 baseline artifact set:
  - `track_scores.jsonl`
  - `per_video_summary.json`
  - `run_summary.json`
- Preserved strict non-goals for this stage:
  - no EM/OT/Sinkhorn
  - no training/loss integration
  - no Stage D loop/orchestration logic

### Validation
- Canonical remote validation passed on `gpu4090d` for Stage C1 targeted tests.
- Branch used: `codex/stagec1-milfirst-offline-baseline-v1`
- Canonical result: `6 passed`
- Intended commit and remote verified HEAD matched:
  - `6644e91eeb354aba640c6bb8108ba926f20d849a`

### Key references
- Stage C1 PASS output:
  - `codex/2026030105_progress_sync_and_stagec0_consumer_loader_taskpack_meta_tier1_2/16_output.txt`
- Stage C1 planning/task-pack continuity:
  - `codex/2026030105_progress_sync_and_stagec0_consumer_loader_taskpack_meta_tier1_2/14_output.txt`
  - `codex/2026030107_stagec1_attribution_baseline_offline_milfirst_tier2/`

### Notes
- Stage C1 baseline is complete as the first offline attribution milestone.
- Historical next-step note (at C1 completion time): Stage C1.r1 real-artifact smoke and diagnostics hardening.
- Current state has advanced beyond this note (Stage C1.r1, Stage C2 baseline, and Stage C3 closure are complete; see entries below).

---

## 2026-03-01 — Stage C2 completed (labelset_proto_v1 offline baseline + real-artifact smoke + determinism)

### Scope
Implement and validate Stage C2 labelset/prototype-conditioned offline attribution baseline (`labelset_proto_v1`) as an additive scorer backend on top of the existing Stage C entrypoint, preserving Stage C1 default behavior.

### Completed
- Added scorer backend dispatch with default-preserving `mil_v1` path and new `labelset_proto_v1` path.
- Added prototype manifest + NPZ loading/validation and per-video labelset conditioning flow.
- Preserved existing artifact contracts and Stage C1 compatibility while adding diagnostics fields additively.
- Completed Stage C2.r1b canonical real-artifact smoke rerun and deterministic double-run equality checks after canonical-runner symlink preflight/relink.

### Validation
- Canonical remote focused pytest passed (`16 passed`) with branch/commit equality:
  - branch: `codex/stagec2-r1-labelset-proto-baseline-impl-v1`
  - commit/remote HEAD: `e01da646e0e46ae5e8cff4491eea57587a3882ee`
- Real-artifact smoke on canonical runner passed using rebuilt run18 sample export split with deterministic artifact equality for:
  - `track_scores.jsonl`
  - `per_video_summary.json`
  - `run_summary.json`

### Key references
- Stage C2.r1 implementation evidence: `codex/2026030109_stagec2_r0_labelset_proto_baseline_plan_tier2_3/04_output.txt`
- Stage C2.r1b repair-and-redo closure evidence: `codex/2026030109_stagec2_r0_labelset_proto_baseline_plan_tier2_3/05_output.txt`
- Stage C2 task-pack closure summary: `codex/2026030109_stagec2_r0_labelset_proto_baseline_plan_tier2_3/07_output.txt`

### Notes
- Stage C2 is complete for offline labelset/prototype baseline scope.
- OT/Sinkhorn/EM attribution expansion beyond current baselines is future Stage C work and should be tracked under C4 planning/milestones (not as unfinished "Stage C2").

---

## 2026-03-01 — Stage C3 completed (global decoder backends + comparison closure)

### Scope
Implement and validate Stage C3 global decoder backend expansion and close the decoder-comparison milestone with canonical evidence and reproducible sidecar artifacts.

### Completed
- Implemented decoder backend stack in C3 scope:
  - pre-existing baseline path: `independent`
  - new backend: `coverage_greedy_v1` (C3.r1 + C3.r1b hardening)
  - new backend: `otlite_v1` (C3.r2a + r2b/r2c calibration)
- Preserved default decoder behavior as `independent` (no production default switch in C3).
- Completed comparison protocol implementation and execution milestones:
  - C3.r3a baseline protocol execution (`small=20`, `medium=75`)
  - C3.r3b evidence expansion (`large=150`, fallback cross-run slot)
  - C3.r3c1 true cross-run closure with non-run18 source (`run19`, tail cohort size `75`)
- Produced comparison sidecar bundles under schema `stagec3.decoder_comparison.v1` for r3a/r3b/r3c1.

### Validation
- Canonical remote validation discipline preserved on `gpu4090d` with branch/commit equality checks for C3 implementation/evidence runs.
- Decoder test bundles passed canonically during C3 implementation phases (including Stage C1/C2/C3 decoder-targeted suites).
- C3.r3c1 provided final required evidence set:
  - large-tier execution evidence (`run18`, `n=150`)
  - true cross-run evidence (`run19`, `cross-run-run19-tail75`, `n=75`)
  - determinism evidence for all required decoders on required tiers
  - comparison sidecars regenerated from canonical protocol logs.

### Recommendation state (post-C3)
- Engineering recommendation: keep default `independent` (no default change).
- Research recommendation: use `otlite_v1` for global-consistency experiments when comparison protocol context is matched.
- Evidence basis now includes large tier + true non-run18 cross-run + determinism + canonical remote validation discipline.

### Key references
- C3 closure evidence output: `codex/2026030110_stagec3-global-decoder-v1/12_output.txt`
- C3 blocked->resolved transition:
  - blocked state: `codex/2026030110_stagec3-global-decoder-v1/11_output.txt`
  - resolved state: `codex/2026030110_stagec3-global-decoder-v1/12_output.txt`
- Final comparison sidecars (r3c1):
  - `codex/2026030110_stagec3-global-decoder-v1/comparison/results/stagec3_r3c1_labelset_proto_run18_large150_crossrunrun19tail75_v1/comparison_manifest.json`
  - `codex/2026030110_stagec3-global-decoder-v1/comparison/results/stagec3_r3c1_labelset_proto_run18_large150_crossrunrun19tail75_v1/comparison_metrics.json`
  - `codex/2026030110_stagec3-global-decoder-v1/comparison/results/stagec3_r3c1_labelset_proto_run18_large150_crossrunrun19tail75_v1/comparison_report.md`

### Notes
- C3 milestone closure is complete for decoder comparison scope without changing Stage B contracts or decoder defaults.
- Optional follow-up: add one or more additional non-run18 cohorts (for example run20+) before considering any default-change proposal.

---

## 2026-03-03 — Stage C4 completed (offline attribution expansion: `em_v1` + `sinkhorn_v1` + C4.3 gates)

### Scope
Close the Stage C4 offline attribution expansion milestone with additive-only scorer evolution, default-OFF compatibility for C4.3 extensions, parity hard-gate coverage, and determinism checks.

### Completed
- C4.1 completed: `em_v1` offline scorer backend integrated in Stage C entrypoint as additive backend expansion.
- C4.2 completed: minimal `sinkhorn_v1` offline backend integrated with Stage C contracts preserved.
- C4.3 planning/spec-lock completed and executed in two bounded implementation slices:
  - C4.3-A: `__bg__` special-column path and additive diagnostics.
  - C4.3-B: `__unk_fg__` gating path with row-level/effective-active semantics.
- Preserved non-negotiable compatibility constraints across C4:
  - additive-only schema evolution (no field removals/renames),
  - default-OFF behavior for C4.3 flags,
  - C4.2 parity hard-gate in tests for disabled-C4.3 path.

### Validation
- Canonical remote validation discipline preserved on `gpu4090d` with branch/commit match checks.
- Final C4.3 targeted closure run passed for parity/bg/unk-fg gating tests:
  - `tests/test_stagec4_sinkhorn_scorer_v1.py -k "test_sinkhorn_c42_parity_hard_gate_snapshot or test_sinkhorn_c43_bg_path_schema_and_source_tagging or test_sinkhorn_c43_unk_fg_gating_schema_and_behavior"`
  - result: `3 passed, 5 deselected`.
- Determinism closure check also passed (including unk-fg-enabled path):
  - `tests/test_stagec4_sinkhorn_scorer_v1.py -k "test_sinkhorn_backend_deterministic_double_run"`
  - result: `1 passed, 7 deselected`.

### Key references
- C4 planning/spec-lock baseline: `codex/2026030302_stagec4-c43-coverage-unkfg/01_output.txt`
- C4.3-A implementation baseline: `codex/2026030302_stagec4-c43-coverage-unkfg/02_output.txt`
- C4.3-A parity hard-gate repair + canonical pass: `codex/2026030302_stagec4-c43-coverage-unkfg/05_output.txt`
- C4.3-B implementation + first canonical failure evidence: `codex/2026030302_stagec4-c43-coverage-unkfg/06_output.txt`
- C4.3-B repair + canonical closure pass: `codex/2026030302_stagec4-c43-coverage-unkfg/07_output.txt`

### Notes
- C4 closure covers offline scorer-path expansion only.
- Out of scope and still future work:
  - C4.3-C or broader coverage/slack redesign,
  - training-loop / Stage D integration,
  - default policy changes for scorer/decoder behavior without new evidence gates.

---

## 2026-03-03 — Stage D1-D12 closure completed (unified track docs-sync closure)

### Scope
Close Stage D with a documentation/continuity pass after D1-D12 completion, and lock canonical validation discipline and handoff clarity for future sessions.

### Completed
- Consolidated Stage D1-D12 as complete in project status and closure docs.
- Explicitly marked D12 complete:
  - quick-check wiring (`tools/run_stage_d10_quick_checks.sh`)
  - helper/runbook reinforcement (`docs/STAGE_D_SMOKE_HELPER_QUICKCHECK.md`)
- Codified Stage D canonical constraints for ongoing work:
  - default-OFF compatibility
  - skip-closed behavior
  - additive-only integration style
  - conda-first canonical remote validation (`wsovvis`)
  - dual `PYTHONPATH` with `${PYTHONPATH:-}`
  - D10 helper + D12 quick-check wrapper as standard tooling checks
- Added closure/handoff artifacts for future-session continuity:
  - `docs/STAGE_D_CLOSURE_MEMO_D1_D12.md`
  - `docs/SESSION_HANDOFF_STAGE_D_CLOSURE.md`

### Validation
- Docs-only closure task; no training behavior changes.
- Full GPU smoke not required for this closure step.
- Local docs consistency checks are the preferred validation mode for this milestone update.

### Notes
- Stage D closure here is documentation/discipline closure for D1-D12, not a new feature milestone.
- Non-goals remain unchanged:
  - no new training/loss/objective semantics
  - no Stage C/D schema changes
  - no Stage D13+ feature implementation in this step
- Candidate next directions (not implemented in this closure):
  - research-facing nonzero supervision semantics under explicit new gates
  - longer smoke/training validation depth once next milestone is approved

---

## 2026-03-03 — N4 completed (nonzero quick-check wiring / runbook continuity lock)

### Scope
Record the Stage D nonzero-semantics N4 continuity point (tooling/docs-only), preserving existing training semantics and helper behavior.

### Completed
- Logged N4 completion as docs/tooling continuity only (no training/loss semantics change).
- Locked canonical quick-check command forms for ON-mode zero vs ON-mode nonzero:
  - zero-mode quick check (ON-mode zero):
    - `tools/run_stage_d10_quick_checks.sh`
    - `python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run --on-mode zero`
  - nonzero-mode quick check (ON-mode nonzero, explicit weight):
    - `tools/run_stage_d10_quick_checks.sh --on-mode nonzero --on-weight 0.25`
    - `python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run --on-mode nonzero --on-weight 0.25`
- Reinforced continuity that N3/N4 validation discipline remains canonical:
  - conda-first `wsovvis` activation
  - `python -m pytest --version` preflight
  - dual `PYTHONPATH` with `${PYTHONPATH:-}`
  - canonical runner path `/home/zyy/code/wsovvis_runner`

### Validation
- Docs-only continuity update (no executable behavior change required for this step).

### Notes
- Non-goals preserved:
  - no helper behavior changes
  - no new tests
  - no schema/training semantics changes

---

## 2026-03-03 — N7 completed (pilot-mode quick-check hardening + continuity sync)

### Scope
Harden and operationalize N6 pilot mode in existing Stage D quick-check tooling/tests/docs only, without changing training semantics.

### Completed
- Extended quick-check wrapper + helper ON-mode wiring to include:
  - `--on-mode zero` (compat sentinel)
  - `--on-mode nonzero` (constant nonzero semantic lock)
  - `--on-mode pilot` (requests `gradient_coupled_pilot_v1`)
- Added additive helper option `--pilot-scale` to pass `nonzero_semantics.gradient_coupled_scale` in pilot mode.
- Added/extended GPU-free dry-run assertions in `tests/test_stage_d9_smoke_helper_v1.py` for:
  - backward-compatible nonzero constant mode markers
  - pilot-mode markers and command overrides
- Updated runbook command matrix to explicitly separate:
  - zero mode
  - constant nonzero mode
  - gradient-coupled pilot mode

### Validation
- Canonical remote quick-check smoke executed once in pilot mode via:
  - `tools/run_stage_d10_quick_checks.sh --on-mode pilot --on-weight 0.25 --pilot-scale 1e-6`
- Canonical discipline preserved:
  - conda-first `wsovvis`
  - `python -m pytest --version` before pytest
  - dual `${PYTHONPATH:-}` in both env setup and command context
  - canonical runner path `/home/zyy/code/wsovvis_runner`

### Notes
- Non-goals preserved:
  - no new loss/objective semantics beyond N6 `gradient_coupled_pilot_v1`
  - no Stage C/D schema contract changes
  - no long training/performance experiments

---

## 2026-03-03 — N10 completed (layered fast gate composition and continuity lock)

### Scope
Add a lightweight layered Stage D fast gate that runs helper coverage first, then optionally runs pilot-capable quick-check smoke, without changing Stage D semantics.

### Completed
- Added `tools/run_stage_d10_layered_fast_gate.sh`:
  - step 1 (always): `tools/run_stage_d9_helper_tests_quick.sh`
  - step 2 (optional via `--with-pilot-smoke`): `tools/run_stage_d10_quick_checks.sh` with pass-through pilot-mode options
  - fail-fast CLI guardrails for invalid/partial pilot arguments
  - concise stage markers for helper coverage and optional pilot quick-check execution
- Added GPU-free command-wiring tests:
  - `tests/test_stage_d10_layered_fast_gate_v1.py`
  - validates sequencing, argument forwarding, and fail-fast argument checks
- Updated docs/readme continuity pointers to lock recommended validation order:
  - helper coverage fast gate first
  - then layered/quick-check smoke as needed

### Validation
- Local targeted pytest:
  - `python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py tests/test_stage_d10_layered_fast_gate_v1.py`
- Canonical remote lightweight verification:
  - conda-first `wsovvis`
  - `python -m pytest --version` preflight
  - layered fast gate run with helper-only and helper+pilot smoke forms

### Notes
- Non-goals preserved:
  - no Stage C/D schema changes
  - no new loss/objective/training semantics
  - no refactor of N1-N9 interfaces

---

## 2026-03-03 — N12 completed (CI-stage canonical replay wrapper for N11 sequence)

### Scope
Package the N11 canonical sequence into a reusable replay entry for CI-stage or operator replay use, with minimal wiring tests/docs continuity only.

### Completed
- Added `tools/run_stage_d11_canonical_replay.sh`:
  - step 1: runs `tools/run_stage_d10_layered_fast_gate.sh`
  - step 2: runs real helper smoke `tools/run_stage_d9_smoke_helper.py --on-mode pilot --pilot-scale 1e-6`
  - supports deterministic argument pass-through for helper smoke knobs (`--output-root`, `--config-path`, `--stagec-artifact-root`, `--on-weight`, `--pilot-scale`, `--keep-output`)
- Added GPU-free sequencing/passthrough tests:
  - `tests/test_stage_d11_canonical_replay_v1.py`
  - validates stage ordering and wrapper-to-helper argument forwarding
- Updated continuity docs/readme pointers:
  - README tooling list includes the new replay entry
  - Stage D runbook includes the bundled replay command in canonical quick-check commands

### Validation
- Local targeted pytest:
  - `python -m pytest -q tests/test_stage_d11_canonical_replay_v1.py tests/test_stage_d10_layered_fast_gate_v1.py tests/test_stage_d9_smoke_helper_v1.py`
- Canonical remote validation (conda-first) executed via `tools/remote_verify_wsovvis.sh` with:
  - required preflight `python -m pytest --version`
  - dual `${PYTHONPATH:-}` usage in both `--env-cmd` and `--cmd`
  - replay entry invocation `bash tools/run_stage_d11_canonical_replay.sh`

### Notes
- Non-goals preserved:
  - no new loss semantics or pilot modes
  - no Stage C/D schema changes
  - no refactor of N1-N11 interfaces

---

## 2026-03-03 — N13 completed (bundled quick-pipeline wiring + output-path discipline hardening)

### Scope
Implement a tooling/docs/CI wiring acceleration step that adds a branch-local CI mirror entry for N11 replay, plus output-path discipline hardening and continuity sync.

### Completed
- Added branch-local quick pipeline wrapper:
  - `tools/run_stage_d13_ci_quick_pipeline.sh`
  - step 1: `tools/run_stage_d9_helper_tests_quick.sh`
  - step 2: `tools/run_stage_d11_canonical_replay.sh`
- Added GPU-free wrapper test:
  - `tests/test_stage_d13_ci_quick_pipeline_v1.py`
  - validates helper->replay ordering and replay arg pass-through (`--on-weight`, `--pilot-scale`)
- Updated continuity docs:
  - README tooling list includes N13 quick pipeline wrapper
  - Stage D runbook includes N13 command and common failure-mode guidance for misplaced `xx_output.txt`
  - prompt-template/spec guidance now explicitly requires output files under `codex/<task_dir>/xx_output.txt` and forbids repo-root output files

### Validation
- Local checks:
  - shell syntax check: `bash -n tools/run_stage_d13_ci_quick_pipeline.sh`
  - targeted pytest: `python -m pytest -q tests/test_stage_d13_ci_quick_pipeline_v1.py tests/test_stage_d11_canonical_replay_v1.py`
- Canonical remote light validation (conda-first):
  - `python -m pytest --version` preflight
  - dual `${PYTHONPATH:-}` in both `--env-cmd` and `--cmd`
  - quick pipeline invocation: `bash tools/run_stage_d13_ci_quick_pipeline.sh`

### Notes
- Non-goals preserved:
  - no new loss semantics or nonzero modes
  - no Stage C/D schema changes
  - no long training/performance runs

---

## 2026-03-03 — N14 completed (formal-CI-ready prep + Stage D quick gate policy)

### Scope
Tooling/docs/policy hardening only for formal CI adoption readiness of Stage D quick pipeline gates.

### Completed
- Added formal-CI-ready gate-policy doc:
  - `docs/STAGE_D_CI_QUICK_PIPELINE_GATE_POLICY.md`
  - documents explicit 3-tier gate policy:
    - helper-only fast gate
    - N13 quick pipeline (helper + replay)
    - broader checks escalation conditions
- Added CI-ready invocation snippets with deterministic prerequisites:
  - conda-first `wsovvis`
  - `python -m pytest --version` preflight
  - safe `PYTHONPATH` form using `${PYTHONPATH:-}`
- Reinforced output-path discipline with copy/paste clause:
  - required `codex/<task_dir>/xx_output.txt`
  - forbidden repo-root `./xx_output.txt`
- Updated continuity pointers:
  - `README.md`
  - `docs/STAGE_D_SMOKE_HELPER_QUICKCHECK.md`
  - `docs/SESSION_HANDOFF_STAGE_D_CLOSURE.md`

### Validation
- Local GPU-free targeted check:
  - `python -m pytest -q tests/test_stage_d13_ci_quick_pipeline_v1.py`
- No semantic/runtime training checks added; wrappers are reused unchanged.

### Notes
- Non-goals preserved:
  - no training/loss/objective semantic changes
  - no Stage C/D schema changes
  - no long training/performance runs

---

## 2026-03-03 — N15 completed (platform-specific lightweight CI wiring for Stage D quick pipeline)

### Scope
Add a lightweight platform-specific CI wiring artifact for Stage D quick-pipeline gates without introducing any Stage D semantic/runtime behavior changes.

### Completed
- Added repository-local CI-ready template:
  - `docs/runbooks/tools/ci_examples/stage_d_quick_pipeline.github_actions.yml`
  - wires a `stage-d-quick-pipeline` job to `bash tools/run_stage_d13_ci_quick_pipeline.sh`
  - includes deterministic prerequisites (`conda`, `python -m pytest --version`, safe `PYTHONPATH`)
- Updated continuity/docs pointers:
  - `README.md`
  - `docs/STAGE_D_CI_QUICK_PIPELINE_GATE_POLICY.md`
  - `docs/STAGE_D_SMOKE_HELPER_QUICKCHECK.md`
  - `docs/SESSION_HANDOFF_STAGE_D_CLOSURE.md`

### Validation
- Local best-effort checks:
  - `bash -n tools/run_stage_d13_ci_quick_pipeline.sh`
  - `bash tools/run_stage_d13_ci_quick_pipeline.sh --help`
  - `python -m pytest -q tests/test_stage_d13_ci_quick_pipeline_v1.py`
- Remote canonical light validation was not executed in N15 (tooling/docs-only scope, no runtime semantic deltas).

### Notes
- Non-goals preserved:
  - no training/loss/objective semantic changes
  - no Stage C/D schema changes
  - no long training/performance runs

---

## 2026-03-03 — N16 completed (CI-enabled mirror workflow materialization + dispatch smoke prep)

### Scope
Move from N15 CI-ready template state to a real CI-enabled mirror workflow file for Stage D quick pipeline gate, then execute one `workflow_dispatch` smoke when environment access permits.

### Completed
- Materialized live workflow file from the N15 template:
  - `.github/workflows/stage_d_quick_pipeline.yml`
- Preserved Stage D quick gate command and prerequisite discipline:
  - `bash tools/run_stage_d13_ci_quick_pipeline.sh`
  - `python -m pytest --version`
  - robust `PYTHONPATH` export with `${PYTHONPATH:-}`
- Added replay instructions for CI-enabled mirrors:
  - `docs/runbooks/tools/ci_examples/README.md`

### Validation
- Local wiring sanity checks:
  - workflow file path/materialization verified at `.github/workflows/stage_d_quick_pipeline.yml`
  - quick gate wrapper remains syntax-valid (`bash -n tools/run_stage_d13_ci_quick_pipeline.sh`)
- Dispatch smoke:
  - attempted from current environment; requires authenticated GitHub Actions dispatch capability (`gh` auth or `GITHUB_TOKEN`)

### Notes
- Non-goals preserved:
  - no training/loss/objective semantic changes
  - no Stage C/D schema changes
  - no long training/performance runs

---

## 2026-03-04 — N17 completed (docs closure + continuity sync for N16/N16.r3 CI unblock)

### Scope
Docs-only closure and continuity sync after N16.r3 CI-hosted replay compatibility fix verification.

### Completed
- Recorded closure state: N16 `workflow_dispatch` smoke is unblocked and verified.
- Recorded N16.r3 compatibility validation: authenticated redispatch succeeded after CI-hosted missing-asset replay-skip handling.
- Captured redispatch evidence for continuity: GitHub Actions run `22650332186` (`completed/success`, 2026-03-04 UTC).
- Added operator continuity note: export `GITHUB_TOKEN` in the same shell/environment that executes Codex/dispatch commands.
- Documented expected CI-hosted behavior: canonical replay may be skipped gracefully when checkpoint assets are absent; this is expected compatibility behavior, not a Stage D semantic failure.
- Added continuity pointers to:
  - `tools/run_stage_d13_ci_quick_pipeline.sh`
  - `tools/run_stage_d11_canonical_replay.sh`
  - `docs/STAGE_D_CI_QUICK_PIPELINE_GATE_POLICY.md`
  - `docs/runbooks/tools/ci_examples/README.md`

### Validation
- Docs-only sanity checks (file/anchor consistency via grep + markdown inspection).
- No new remote canonical run required because this step introduces no code/workflow behavior changes.

### Notes
- Non-goals preserved:
  - no training/loss/objective semantic changes
  - no Stage C/D schema changes
  - no CI logic/runtime behavior changes in this step

---

## 2026-03-04 — N24 line completed (semantic hardening + canonical replay contract closure)

### Scope
Close the bounded Stage D nonzero-semantics N24 line: semantic sign/direction hardening, canonical bootstrap hardening, and canonical replay diagnostics-contract triage/rerun closure.

### Completed
- N24 completed: applied-path sign/direction boundary hardening landed and remained stable.
- N24.r1 completed: canonical bootstrap hardening for non-interactive shells and robust conda path detection landed, including `/home/zyy/software/miniconda3` resolution.
- N24.r2 completed: canonical replay smoke diagnostics-consistency mismatch was triaged/fixed and one-shot rerun passed.
- Helper diagnostics consistency contract now accepts valid skipped-pilot mappings for:
  - `planned_loss.apply_mode=loss_dict_insert_zero` (loss-dict insertion branch), and
  - `planned_loss.apply_mode=placeholder_zero` (placeholder branch),
  when associated indicator fields are internally consistent.

### Validation
- N24 line closure evidence includes targeted canonical validation plus canonical replay rerun PASS in the N24.r2 record.
- This continuity entry is docs-only and introduces no code-path behavior changes.

### Notes
- N24 line is now closed end-to-end: semantic hardening + canonical bootstrap reliability + replay diagnostics contract consistency.

---

## 2026-03-04 — N26 line completed (dual-valid skipped-pilot matrix + canonical-first verification closure)

### Scope
Close N26 and N26.r1 for skipped-pilot diagnostics/replay stability using canonical-first verification discipline, without expanding Stage D semantics.

### Completed
- N26 closed with dual-valid skipped-pilot matrix coverage preserved for:
  - `planned_loss.apply_mode=loss_dict_insert_zero`
  - `planned_loss.apply_mode=placeholder_zero`
- N26.r1 canonical-first closure completed after local pytest unavailability triage (`No module named pytest`) by moving directly to canonical ladder on `gpu4090d`.
- A replay-test contract bug surfaced during canonical run and was fixed minimally in-scope:
  - escaped `${PWD}` in f-string shell snippet (`${{PWD}}`) so generated replay stub preserves shell variable semantics.
- Canonical replay smoke rerun reached PASS after fix (`D11_CANONICAL_REPLAY=PASS`).

### Validation
- Canonical preflight passed (conda-first, `pytest 9.0.2`).
- Targeted canonical subset closure rerun passed for failing replay-contract tests (`3 passed`).
- Canonical replay smoke passed after minimal test-contract fix.
- Commit/push closure for N26.r1:
  - `af9cb66a367277e412f35c64ffee32f77ce2aca0`
  - branch: `staged-nonzero-semantics`

### Notes
- N26/N26.r1 closure was contract/tooling hardening only; no Stage C/D schema change and no loss-semantic expansion.

---

## 2026-03-04 — N27 line completed (replay/CLI minimal-fieldset stability + fresh-runner bootstrap closure)

### Scope
Close N27 and N27.r1 by hardening replay/CLI skipped-pilot output assertions and resolving fresh-runner bootstrap blockers for canonical replay.

### Completed
- N27 closed with minimal-fieldset stability assertions for `D10_PILOT_*` replay/CLI output contract:
  - added compact field parsing/assertion strategy to reduce brittle full-line coupling,
  - preserved contradiction-path fail-fast guarantees (no false PASS markers).
- N27 canonical targeted tests passed (`3 passed`) on canonical runner.
- N27.r1 closed by repairing canonical-runner wsovvis_live symlink bootstrap in `/home/zyy/code/wsovvis_runner`:
  - `third_party/CutLER`, `third_party/dinov2`, `runs`, `weights`, `data`
- Canonical replay rerun passed after bootstrap fix (`D11_CANONICAL_REPLAY=PASS`).
- Runbook continuity updated to document the exact linkage pattern and first-check commands.

### Validation
- Canonical preflight passed on fresh runner path with conda-first setup.
- Canonical replay/CLI targeted test selector passed (`3 passed`) on commit:
  - `76c6188e24619a08cf77602ab7b155d735177107`
- Canonical replay smoke PASS recorded after bootstrap repair on commit:
  - `a0a313b6ecd40d65adc016677000acfbf93775d3`

### Notes
- N27/N27.r1 closure remained within tooling/bootstrap and replay contract hardening boundaries.

---

## 2026-03-04 — N28 completed (canonical runner bootstrap link check/fix helper + docs integration)

### Scope
Add a focused helper to standardize canonical-runner wsovvis_live linkage triage/fix and integrate it into Stage D quickcheck documentation.

### Completed
- Added helper:
  - `tools/check_canonical_runner_bootstrap_links.py`
- Helper covers managed paths:
  - `third_party/CutLER`
  - `third_party/dinov2`
  - `runs`
  - `weights`
  - `data`
- Implemented `--check` and `--fix` modes with safe handling:
  - repairs missing/broken/wrong symlinks,
  - reports and skips non-symlink conflicts (`SKIPPED`) instead of destructive replacement.
- Integrated concise usage/status guidance into:
  - `docs/STAGE_D_SMOKE_HELPER_QUICKCHECK.md`

### Validation
- Lightweight validation only (docs/tooling scope):
  - script compile/help/check checks,
  - controlled temp-fixture fix-mode transitions (`WRONG_TARGET/MISSING/BROKEN_SYMLINK -> FIXED`),
  - non-symlink conflict preserved as `SKIPPED`.
- Canonical runner path was unavailable in the execution environment used for the helper implementation step; this did not block helper closure.

### Notes
- N28 is closed as tooling/docs continuity hardening; no Stage D training semantic change.
- Closure commit/head:
  - `fe94247aff03853f1c8ddd85f023a7957084a367` on `staged-nonzero-semantics` (also `origin/staged-nonzero-semantics` at closure time).

---

## 2026-03-05 — N29.r1 completed (real-runner canonical replay verification closure + docs continuity sync)

### Scope
Close the remaining N29 gap with one real-runner canonical replay proof using integrated bootstrap preflight flags, then perform minimal continuity docs sync.

### Completed
- Confirmed N29 integrated bootstrap preflight options are present in canonical replay wrapper:
  - `tools/run_stage_d11_canonical_replay.sh`
  - flags: `--bootstrap-link-check`, `--bootstrap-link-fix`, `--bootstrap-runner-root`
- Executed one real canonical replay smoke on `gpu4090d` with conda-first runtime and integrated preflight enabled.
- Replay used bootstrap fix + enforced re-check in-wrapper and reached pass:
  - `D11_CANONICAL_REPLAY=PASS`
- Maintained fixed validation ladder and hard-stop behavior (no `SKIPPED` continuation).

### Validation
- Real runner path: `/home/zyy/code/wsovvis_runner` (branch `staged-nonzero-semantics`, runtime commit `f36c93d336ed09d7e2cf515a3516b76f7656282a`).
- Conda-first preflight passed:
  - `source ~/software/miniconda3/etc/profile.d/conda.sh`
  - `conda activate wsovvis`
  - `python -m pytest --version` -> `pytest 9.0.2`
- Canonical replay command:
  - `bash tools/run_stage_d11_canonical_replay.sh --bootstrap-link-fix --bootstrap-runner-root /home/zyy/code/wsovvis_runner`
- Bootstrap stages observed:
  - `bootstrap_link_preflight_fix` -> `FIXED` on managed links
  - `bootstrap_link_preflight_recheck` -> `OK` on managed links
- Replay smoke stages passed:
  - `n10_layered_fast_gate PASS`
  - `pilot_helper_smoke PASS`
  - `D11_CANONICAL_REPLAY=PASS`

### Notes
- N29.r1 closure remained tooling/verification/docs only; no Stage D semantic expansion.

---

## 2026-03-05 — N29.r1 tooling/verification/docs-only closure refresh (real-runner integrated replay check path)

### Scope
Execute the N29 integrated bootstrap-preflight replay flow on the real canonical runner with fixed ladder discipline and stop after one successful replay smoke.

### Completed
- Local lightweight checks passed:
  - `bash -n tools/run_stage_d11_canonical_replay.sh`
  - `bash tools/run_stage_d11_canonical_replay.sh --help`
- Real-runner conda-first preflight captured:
  - runner: `/home/zyy/code/wsovvis_runner`
  - branch/head: `staged-nonzero-semantics` / `f36c93d336ed09d7e2cf515a3516b76f7656282a`
  - `python -m pytest --version` -> `pytest 9.0.2`
- Integrated bootstrap check path executed on real runner:
  - `bash tools/run_stage_d11_canonical_replay.sh --bootstrap-link-check --bootstrap-runner-root /home/zyy/code/wsovvis_runner`
  - preflight check status remained `OK` on managed links.
- Exactly one successful canonical replay smoke recorded in this closure run:
  - command run with conda-first + explicit runtime PYTHONPATH:
    - `export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}`
    - `bash tools/run_stage_d11_canonical_replay.sh --bootstrap-link-check --bootstrap-runner-root /home/zyy/code/wsovvis_runner`
  - result markers:
    - `D11_CANONICAL_REPLAY_STAGE=bootstrap_link_preflight_check PASS`
    - `D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate PASS`
    - `D11_CANONICAL_REPLAY_STAGE=pilot_helper_smoke PASS`
    - `D11_CANONICAL_REPLAY=PASS`

### Validation notes
- Canonical validation policy update: use only `/home/zyy/code/wsovvis_runner`; do not use `wsovvis_runner_*` side runners. If wrapper/version mismatch appears, sync/reset the canonical runner in-place before rerun.
- No repo tooling code changes were required for N29 flow logic in this closure refresh.

### Next-step lock
- Next prompt is Stage C mainline C0 (minimal semantic vertical slice interface freeze), unless a new blocker appears.

---

## Validation evidence highlights (through Stage D closure + N29.r1)

### Canonical remote validation discipline (preserved)
- Canonical remote host/path usage remained consistent in PASS evidence:
  - host alias: `gpu4090d`
  - repo dir: `/home/zyy/code/wsovvis_runner`
- Branch/commit match discipline was explicitly enforced in Stage C0 and Stage C1 PASS logs.
- Stage C0 failure context captured the known pitfall:
  - remote verify can pass/fail on a branch/HEAD that does not include local unpushed edits (`08_output.txt` context), requiring commit+push+remote-HEAD equality checks before claiming PASS.

### Stage C0 and Stage C1 canonical test outcomes
- Stage C0 canonical remote PASS:
  - `10 passed` (`tests/test_stagec_loader_v1.py`)
  - evidence: `.../10_output.txt`
- Stage C1 canonical remote PASS:
  - `6 passed` (`tests/test_stagec1_attribution_mil_v1.py`)
  - evidence: `.../16_output.txt`
- Stage C2 canonical remote PASS:
  - `16 passed` (`tests/test_stagec1_attribution_mil_v1.py` + `tests/test_stagec2_labelset_proto_baseline_v1.py`)
  - evidence: `codex/2026030109_stagec2_r0_labelset_proto_baseline_plan_tier2_3/05_output.txt`

### Stage C3 decoder comparison closure evidence
- Decoder backends validated in C3 scope:
  - baseline: `independent`
  - alternatives: `coverage_greedy_v1`, `otlite_v1`
- C3.r3 comparison sidecar schema remained stable:
  - `schema_version = stagec3.decoder_comparison.v1`
- Final closure run (C3.r3c1) captured required tiers and determinism requirements:
  - `required_tiers = [large, cross-run-run19-tail75]`
  - `determinism_required_tiers = [large, cross-run-run19-tail75]`
- Final recommendation state in sidecar metrics:
  - engineering: `keep_default_independent`
  - research: `otlite_v1_for_global_consistency_experiments`
- True cross-run closure status:
  - prior blocker (no non-run18 source) recorded in C3.r3c
  - resolved in C3.r3c1 via run19 source availability and protocol completion

### Stage C4 attribution expansion closure evidence
- C4.1/C4.2 implementation line completed as additive backend expansion under existing Stage C artifact contracts.
- C4.3 closure included both feature slices:
  - C4.3-A (`__bg__`) path and schema/source tagging.
  - C4.3-B (`__unk_fg__`) row-level gating semantics.
- C4.2 parity hard gate remained required and passed in closure run:
  - `test_sinkhorn_c42_parity_hard_gate_snapshot`.
- C4.3 targeted closure run passed on canonical remote:
  - `3 passed, 5 deselected` for parity + bg + unk-fg selector.
- Determinism closure run passed on canonical remote:
  - `1 passed, 7 deselected` for deterministic double-run selector including unk-fg-enabled mode.
- Final C4.3-B repair was test-semantics alignment (not algorithmic redesign):
  - row-level `sinkhorn_active_special_columns` may be `["__bg__"]` or `["__bg__", "__unk_fg__"]` depending on gating effectiveness.

### P3.1c.1 and P3.1c.2 real-run handoff evidence (bridge -> export -> validator)
- P3.1c.1 real run18 sample handoff chain passed:
  - builder -> adapter/export -> validator
  - validator reported `Validation OK`
  - targeted tests passed remotely: `16 passed`
- P3.1c.2 hardening QA passed for run18 sample and larger subsets:
  - sample (`--sample-video-limit 10`) and larger (`--sample-video-limit 200`) both reached validator `Validation OK`
  - QA summary counters were hardened with deterministic join/drop/runtime/video diagnostics
  - representative evidence: sample matched tracks `100` with dropped `4`; larger matched tracks `2000` with dropped `72` (reason-coded in QA summary)

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
- ✅ P3.1c.2 completed (bridge full-split hardening + QA)
- ✅ Stage C0 completed (consumer/loader offline data-plane; canonical remote PASS `10/10`)
- ✅ Stage C1 completed (offline MIL-first attribution baseline; canonical remote PASS `6 passed`)
- ✅ Stage C1.r1 completed (real-artifact smoke + diagnostics hardening + determinism evidence)
- ✅ Stage C2 completed (labelset_proto_v1 offline baseline + canonical remote pytest + real-artifact smoke + determinism)
- ✅ Stage C3 completed (global decoder baseline + alternatives + comparison closure; default remains `independent`)
- ✅ Stage C4 completed (offline attribution expansion line: `em_v1`, `sinkhorn_v1`, C4.3 `bg` + `unk-fg` gating, parity + determinism closure)
- ✅ Stage D1-D12 completed (unified integration line through helper/runbook/quick-check reinforcement)
- ✅ Stage D12 completed (quick-check wiring + runbook/docs reinforcement)
- ▶️ Current state: **Stage D closure complete (docs-sync + continuity lock)**
- ▶️ Policy-synced near-term execution order (2026-03-05):
  - first: **N29.r1** canonical replay with integrated bootstrap preflight as final infra closure;
  - then: return to **Stage C substantive semantic mainline** (`C0 -> C1 -> C2 -> C3 -> C4` minimal vertical slices).
- ▶️ Exception rule: allow at most **1-2 extra prompts** only for unexpected blockers/errors, then return to the mainline sequence above.

### Stage D closure constraints (must remain true)
- default-OFF compatibility remains mandatory.
- skip-closed behavior remains mandatory.
- additive-only integration style remains mandatory.
- canonical remote validation remains conda-first on `gpu4090d` / `/home/zyy/code/wsovvis_runner` with `wsovvis` env.
- dual `PYTHONPATH` form must preserve `${PYTHONPATH:-}`.
- D10 helper + D12 quick-check wrapper are the standard validation tools for Stage D tooling checks.
