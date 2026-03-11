# WSOVVIS Codebase Map

This file maps the current clean code repository to the new automation control plane.

## 1. Top-level code domains
### Protocol / dataset tooling
- `tools/build_wsovvis_labelset_protocol.py`
- `tests/test_build_wsovvis_labelset_protocol.py`
- `wsovvis/data/`

### Stage B export / bridge / loader foundation
Authoritative contract anchors for G0/G2:
- export core: `wsovvis/track_feature_export/v1_core.py`
- bridge: `wsovvis/track_feature_export/bridge_from_stageb_v1.py`
- consumer / loader-facing interface: `wsovvis/track_feature_export/stagec_loader_v1.py`

Authoritative Stage B contract tests:
- `tests/test_stageb_track_feature_export_v1.py`
- `tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py`
- `tests/test_stagec_loader_v1.py`

Supporting Stage B preparation / integration utilities:
- `wsovvis/track_feature_export/feature_export_enablement_v1.py`
- `wsovvis/track_feature_export/real_run_feature_export.py`
- `wsovvis/track_feature_export/real_stageb_sidecar_bridge_input_builder_v1.py`
- `tools/build_stageb_track_feature_export_v1.py`
- `tools/build_stageb_track_feature_export_from_stageb_bridge_v1.py`
- `tools/build_stageb_bridge_input_from_real_stageb_sidecar_v1.py`
- `tools/build_stageb_feature_export_enablement_v1.py`
- `tools/validate_stageb_track_feature_export_v1.py`
- `tests/test_stageb_feature_export_enablement_v1.py`
- `tests/test_stageb_feature_export_real_run_integration_v1.py`
- `tests/test_real_stageb_sidecar_bridge_input_builder_v1.py`

Clarification:
- `feature_export_enablement_v1.py` and `real_run_feature_export.py` prepare and export the upstream sidecar-style enablement artifact; they are not the final split-manifest + per-video Stage B export contract consumed by Stage C.
- `real_stageb_sidecar_bridge_input_builder_v1.py` normalizes real-run sidecar outputs into the reduced bridge-input domain consumed by `bridge_from_stageb_v1.py`.
- `tools/build_stageb_track_feature_export_v1.py` wraps `build_track_feature_export_v1`, `tools/build_stageb_track_feature_export_from_stageb_bridge_v1.py` wraps `build_track_feature_export_v1_from_stageb_bridge_input`, and `stagec1_attribution_mil_v1.py` consumes the export through `load_stageb_export_split_v1`.

### Stage C semantic and attribution stack
Primary files:
- `wsovvis/track_feature_export/stagec_clip_text_prototype_cache_v1.py`
- `wsovvis/track_feature_export/stagec_semantic_slice_v1.py`
- `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- `wsovvis/training/stagec_semantic_plumbing_v1.py`
- `wsovvis/training/staged_attribution_plumbing_v1.py`
- `tools/run_stagec1_mil_baseline_offline.py`
- `tools/compare_stagec_unknown_handling_v1.py`
- `tests/test_stagec1_attribution_mil_v1.py`
- `tests/test_stagec2_labelset_proto_baseline_v1.py`
- `tests/test_stagec3_otlite_decoder_v1.py`
- `tests/test_stagec4_em_scorer_v1.py`
- `tests/test_stagec4_sinkhorn_scorer_v1.py`
- `tests/test_stagec_semantic_slice_v1.py`
- `tests/test_staged_attribution_plumbing_v1.py`

Important note:
- `stagec1_attribution_mil_v1.py` contains multiple historical decoders / scorers.
- Under this automation kit, they are **not** all mainline by default.
- The active default must be recorded in `STATUS.md` and judged by the current gate.

### G5 full-video linking and inference closure
Primary runtime files:
- `third_party/VNext/projects/SeqFormer/seqformer/models/clip_output.py`
- `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`

Primary repo-level policy test:
- `tests/test_g5_bounded_policy_v1.py`

Supporting wrapper coverage only:
- `tools/run_stage_d9_smoke_helper.py`
- `tests/test_stage_d9_smoke_helper_v1.py`
- `tests/test_stage_d11_canonical_replay_v1.py`
- `tests/test_stage_d13_ci_quick_pipeline_v1.py`

Clarification:
- `clip_output.py` owns cross-window state accumulation, geometry-gated query matching, and quality-weighted logit aggregation across clips.
- `seqformer.py` owns the runtime switch between whole-video inference and clip-matching inference, and `inference_clip` is the authoritative handoff point for query embeddings, raw class logits, and clip-quality scores into `Clips`.
- `tests/test_g5_bounded_policy_v1.py` is the dedicated repo-level bounded-policy test for `Videos.update` and `Videos.get_result`; canonical runtime smoke remains the authoritative end-to-end closure check.

### Metrics and reporting
- `wsovvis/metrics/ws_metrics.py`
- `wsovvis/metrics/ws_metrics_reporting_v1.py`
- `wsovvis/metrics/ws_metrics_stage_d_adapter_v1.py`
- `tools/ws_metrics_demo.py`
- `tools/ws_metrics_summary_demo_v1.py`
- `tests/test_ws_metrics_tooling.py`
- `tests/test_ws_metrics_reporting_v1.py`
- `tests/test_stage_d_ws_metrics_adapter_v1.py`

Gap note:
- The mainline acceptance expects `HPR` and `UAR`.
- If these are missing in code, this is an evidence gap, not a reason to widen scope.

### Stage D and refinement-related utilities
Authoritative bounded G6 path:
- refinement loop: `tools/run_stage_d0_self_training_loop.py`
- refinement ws-metrics adapter: `wsovvis/metrics/ws_metrics_stage_d_adapter_v1.py`
- refinement reporting snapshot: `tools/stage_d_reporting_snapshot.py`
- authoritative G6 tests:
  `tests/test_stage_d0_self_training_loop_v1.py`
  `tests/test_stage_d_reporting_snapshot_v1.py`
  `tests/test_stage_d_ws_metrics_adapter_v1.py`

Supporting Stage D wrappers and CI helpers only:
- `tools/run_stage_d9_smoke_helper.py`
- `tools/run_stage_d10_quick_checks.sh`
- `tools/run_stage_d10_layered_fast_gate.sh`
- `tools/run_stage_d11_canonical_replay.sh`
- `tools/run_stage_d13_ci_quick_pipeline.sh`
- `tests/test_stage_d9_smoke_helper_v1.py`
- `tests/test_stage_d10_layered_fast_gate_v1.py`
- `tests/test_stage_d11_canonical_replay_v1.py`
- `tests/test_stage_d13_ci_quick_pipeline_v1.py`

Policy:
- These files are available, but multi-round Stage D behavior is not mainline by default.
- Use only bounded single-round refinement semantics unless `STATUS.md` explicitly activates more.

### Supervisor and terminal automation
Primary control-plane file:
- `tools/run_mainline_loop.py`

Primary repo-level supervisor/terminal test:
- `tests/test_run_mainline_loop_v1.py`

Authoritative bounded terminal regression anchors:
- G4 bounded attribution:
  `tests/test_stagec2_labelset_proto_baseline_v1.py::test_decoder_independent_matches_scorer_predictions`
  `tests/test_stagec4_sinkhorn_scorer_v1.py::test_sinkhorn_c43_unk_fg_gating_schema_and_behavior`
  `tests/test_ws_metrics_reporting_v1.py::test_ws_metrics_summary_v1_exposes_hpr_and_uar_when_hidden_positive_inputs_exist`
- G5 bounded linking / quality-weighted classification:
  `tests/test_g5_bounded_policy_v1.py`
- G6 bounded refinement:
  `tests/test_stage_d0_self_training_loop_v1.py::test_d0_round0_round1_with_minimal_refine_and_stagec_seed`
  `tests/test_stage_d0_self_training_loop_v1.py::test_d0_emit_ws_metrics_preserves_hidden_positive_fields_for_hpr_uar`
  `tests/test_stage_d_reporting_snapshot_v1.py::test_stage_d_snapshot_happy_path_detects_round_paths`
  `tests/test_stage_d_ws_metrics_adapter_v1.py::test_stage_d_ws_metrics_adapter_exposes_hpr_and_uar_when_optional_fields_exist`

Policy:
- after the accepted terminal gate is reached, `tools/run_mainline_loop.py` must stop by default and write a terminal summary instead of a new coding-step prompt
- bounded terminal revalidation is allowed only to preserve the accepted G4 / G5 / G6 defaults under canonical validation semantics

## 2. Mainline entrypoint policy
This automation kit treats the repo as code-first but gate-controlled.
If multiple code paths exist, the active one must be declared in `STATUS.md`; it must not be chosen ad hoc from historical code variety.
