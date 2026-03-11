# WS-OVVIS Codebase Map

This file maps the repository to the automation control plane.
It distinguishes between current reusable anchors and planned v9 entrypoints that are not yet present in the working tree.

## Top-level code domains

### 1. Outline and control-plane authority
Current authoritative design source:
- `docs/outline/WS_OVVIS_outline_v9.tex`

Active control-plane support files after deployment:
- `AGENTS.md`
- `docs/mainline/*`
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `tools/run_mainline_loop.py`

### 2. Protocol and weak-label tooling (`G1`)
Current reusable anchors:
- `tools/build_wsovvis_labelset_protocol.py`
- `tests/test_build_wsovvis_labelset_protocol.py`
- `wsovvis/metrics/ws_metrics.py`
- `wsovvis/metrics/ws_metrics_reporting_v1.py`
- `tools/ws_metrics_demo.py`
- `tools/ws_metrics_summary_demo_v1.py`

Expected evidence role:
- clip-level `Y'(v)` statistics
- full-vs-observed label worked examples
- `SCR`, `HPR`, `UAR`, and `AURC` reporting support

### 3. Class-agnostic basis and local-tracklet export (`G2`)
Current reusable anchors:
- `train_seqformer_pseudo.py`
- `configs/seqformer_pseudo_sacred.yaml`
- `scripts/run_videocutler_4gpu_png_only.sh`
- `scripts/rerun_failed_videos.sh`
- `tools/convert_videocutler_png_to_json.py`
- `tools/build_stageb_feature_export_enablement_v1.py`
- `tools/build_stageb_bridge_input_from_real_stageb_sidecar_v1.py`
- `tools/build_stageb_track_feature_export_v1.py`
- `tools/build_stageb_track_feature_export_from_stageb_bridge_v1.py`
- `wsovvis/track_feature_export/v1_core.py`
- `wsovvis/track_feature_export/feature_export_enablement_v1.py`
- `wsovvis/track_feature_export/real_stageb_sidecar_bridge_input_builder_v1.py`
- `wsovvis/track_feature_export/bridge_from_stageb_v1.py`
- `wsovvis/track_feature_export/real_run_feature_export.py`
- `wsovvis/track_feature_export/stagec_loader_v1.py`

Primary tests:
- `tests/test_stageb_feature_export_enablement_v1.py`
- `tests/test_real_stageb_sidecar_bridge_input_builder_v1.py`
- `tests/test_stageb_track_feature_export_v1.py`
- `tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py`
- `tests/test_stageb_feature_export_real_run_integration_v1.py`
- `tests/test_stagec_loader_v1.py`

### 4. Clip-level global track bank (`G3`)
Current active mainline entrypoints and reusable structural references:
- `tools/build_global_track_bank_v9.py`
- `wsovvis/tracking/global_track_bank_v9.py`
- `tests/test_global_track_bank_v9.py`
- `wsovvis/track_feature_export/bridge_from_stageb_v1.py`
- `wsovvis/track_feature_export/stagec_loader_v1.py`
- `tests/test_stagec_loader_v1.py`
- `tests/test_g5_bounded_policy_v1.py` (legacy reference only for bounded linking policy)
- `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py` (legacy/reference only; not the active v9 mainline by default)

Expected evidence role:
- score matrices
- merge logs
- clip-level global-track coverage figures

### 5. DINO-only semantic carrier (`G4`)
Current active mainline entrypoints and reusable anchors:
- `tools/extract_track_dino_features_v9.py`
- `wsovvis/features/track_dino_feature_v9.py`
- `tests/test_track_dino_feature_v9.py`
- `wsovvis/modeling/backbone/dinov2_backbone.py`

Expected evidence role:
- crop and pooling provenance
- semantic-cache coverage
- objectness diagnostics

### 6. Prototype bank and text map (`G5`)
Current reusable anchors or scaffolds:
- `wsovvis/track_feature_export/stagec_clip_text_prototype_cache_v1.py`
- `tests/test_stagec2_labelset_proto_baseline_v1.py`

Planned mainline entrypoints:
- `tools/init_prototype_bank_v9.py`
- `tools/train_text_map_v9.py`
- `wsovvis/semantics/prototype_bank_v9.py`
- `wsovvis/semantics/text_map_v9.py`

Expected evidence role:
- prototype coverage
- text-map alignment
- class-level worked examples

### 7. Core attribution (`G6`)
Current baseline or scaffold anchors:
- `tools/run_stagec1_mil_baseline_offline.py`
- `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- `wsovvis/track_feature_export/stagec_semantic_slice_v1.py`
- `wsovvis/training/stagec_semantic_plumbing_v1.py`
- `wsovvis/training/staged_attribution_plumbing_v1.py`
- `tools/compare_stagec_unknown_handling_v1.py`
- `tools/run_stagec_c5_micro_training.py`

Primary tests:
- `tests/test_stagec1_attribution_mil_v1.py`
- `tests/test_stagec_semantic_c1_v1.py`
- `tests/test_stagec_semantic_c2_sinkhorn_v1.py`
- `tests/test_stagec_semantic_c3_minimal_v1.py`
- `tests/test_stagec_semantic_c4_observability_v1.py`
- `tests/test_stagec_semantic_c9_em_minimal_v1.py`
- `tests/test_stagec_semantic_c9_mil_minimal_v1.py`
- `tests/test_stagec_semantic_slice_v1.py`
- `tests/test_stagec3_otlite_decoder_v1.py`
- `tests/test_stagec4_em_scorer_v1.py`
- `tests/test_stagec4_sinkhorn_scorer_v1.py`
- `tests/test_staged_attribution_plumbing_v1.py`
- `tests/test_stagec_c5_micro_training_real_data_v1.py`

Planned mainline entrypoints:
- `tools/train_openworld_core_v9.py`
- `wsovvis/attribution/openworld_core_v9.py`

### 8. Bag-free inference and terminal evaluation (`G7`)
Current reusable reporting anchors:
- `wsovvis/metrics/ws_metrics.py`
- `wsovvis/metrics/ws_metrics_reporting_v1.py`
- `tests/test_ws_metrics_reporting_v1.py`
- `tests/test_ws_metrics_tooling.py`

Planned mainline entrypoints:
- `tools/run_bagfree_eval_v9.py`
- `wsovvis/inference/bagfree_inference_v9.py`

Expected evidence role:
- full-vocabulary prediction dumps
- AP / HPR / UAR / robustness summaries
- qualitative visualizations

### 9. Legacy default-off continuation paths
These remain in the repository as legacy, baseline, or optional reference assets until explicitly promoted:
- `tools/run_stage_d0_self_training_loop.py`
- `tools/run_stage_d9_smoke_helper.py`
- `tools/run_stage_d10_layered_fast_gate.sh`
- `tools/run_stage_d11_canonical_replay.sh`
- `tools/run_stage_d13_ci_quick_pipeline.sh`
- `wsovvis/metrics/ws_metrics_stage_d_adapter_v1.py`
- `tools/stage_d_reporting_snapshot.py`

Primary tests:
- `tests/test_stage_d0_self_training_loop_v1.py`
- `tests/test_stage_d9_smoke_helper_v1.py`
- `tests/test_stage_d10_layered_fast_gate_v1.py`
- `tests/test_stage_d11_canonical_replay_v1.py`
- `tests/test_stage_d13_ci_quick_pipeline_v1.py`
- `tests/test_stage_d_reporting_snapshot_v1.py`
- `tests/test_stage_d_ws_metrics_adapter_v1.py`

These assets are **not** the active v9 mainline unless `STATUS.md` explicitly says otherwise.

## Mainline entrypoint policy
If multiple code paths exist, the active one must be declared in `STATUS.md`; it must not be chosen ad hoc from historical code variety.

## Evidence-producing entrypoint policy
For each active gate, the authoritative scripts, tests, or artifact producers that generate the required evidence pack should be identifiable from this map or explicitly recorded in `STATUS.md`.
