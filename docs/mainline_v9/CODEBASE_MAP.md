# WSOVVIS V9 Codebase Map

This file maps the current repository to the v9 automation control plane.
It also identifies the expected evidence-bearing entrypoints for each gate.

## 1. Top-level code domains
### Protocol / dataset tooling
Current reusable anchors:
- `tools/build_wsovvis_labelset_protocol.py`
- `tests/test_build_wsovvis_labelset_protocol.py`
- `wsovvis/data/`

Planned v9 role:
- authoritative clip-level protocol generation for `Y'(v)`
- source for G1 protocol statistics and worked examples

### Class-agnostic basis / Stage B foundation
Current reusable anchors:
- `train_seqformer_pseudo.py`
- `configs/seqformer_pseudo_sacred.yaml`
- `tools/convert_videocutler_png_to_json.py`
- `scripts/run_videocutler_4gpu_png_only.sh`
- `scripts/rerun_failed_videos.sh`

Current export/bridge anchors worth reusing structurally:
- `wsovvis/track_feature_export/v1_core.py`
- `wsovvis/track_feature_export/bridge_from_stageb_v1.py`
- `wsovvis/track_feature_export/stagec_loader_v1.py`

Planned v9 entrypoints:
- `tools/build_stageb_local_tracklets_v9.py`
- `wsovvis/track_feature_export/stageb_local_tracklets_v9.py`

Expected evidence role:
- G2 local-tracklet metrics, visualizations, and worked-example dumps

### Clip-level global track bank (new v9 mainline module)
Planned mainline entrypoints:
- `tools/build_global_track_bank_v9.py`
- `wsovvis/tracking/global_track_bank_v9.py`

Legacy/reference only:
- G5 linking behavior in `third_party/VNext/projects/SeqFormer/seqformer/models/clip_output.py`
- `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`
- `tests/test_g5_bounded_policy_v1.py`

Expected evidence role:
- G3 score matrices, merge logs, global-track coverage figures

### DINO-only track semantics (new v9 mainline module)
Current reusable integration experience:
- `wsovvis/modeling/backbone/dinov2_backbone.py` (if present in local branches / prior runs; verify before promotion)

Planned mainline entrypoints:
- `tools/extract_track_dino_features_v9.py`
- `wsovvis/features/track_dino_feature_v9.py`

Expected evidence role:
- G4 crop/pooling provenance, semantic-cache statistics, objectness diagnostics

### Prototype bank and text map (new v9 mainline module)
Planned mainline entrypoints:
- `tools/init_prototype_bank_v9.py`
- `tools/train_text_map_v9.py`
- `wsovvis/semantics/prototype_bank_v9.py`
- `wsovvis/semantics/text_map_v9.py`

Legacy/reference only:
- `wsovvis/track_feature_export/stagec_clip_text_prototype_cache_v1.py`

Expected evidence role:
- G5 prototype coverage, alignment metrics, class-level worked examples

### Core attribution (new v9 mainline module)
Planned mainline entrypoints:
- `tools/train_openworld_core_v9.py`
- `wsovvis/attribution/openworld_core_v9.py`

Legacy/reference / baseline only:
- `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- `wsovvis/track_feature_export/stagec_semantic_slice_v1.py`
- `wsovvis/training/stagec_semantic_plumbing_v1.py`
- `wsovvis/training/staged_attribution_plumbing_v1.py`
- `tools/run_stagec1_mil_baseline_offline.py`
- `tools/compare_stagec_unknown_handling_v1.py`

Expected evidence role:
- G6 cost matrices, allocation metrics, closed-vs-open comparisons, clip-level worked examples

### Bag-free inference (new v9 mainline module)
Planned mainline entrypoints:
- `tools/run_bagfree_eval_v9.py`
- `wsovvis/inference/bagfree_inference_v9.py`

Expected evidence role:
- G7 prediction dumps, evaluation summaries, qualitative visualizations

### Optional refinement (default-off)
Planned optional module:
- `wsovvis/refinement/refine_once_v9.py`

Legacy/reference only:
- `tools/run_stage_d0_self_training_loop.py`
- `wsovvis/metrics/ws_metrics_stage_d_adapter_v1.py`
- `tools/stage_d_reporting_snapshot.py`

Expected evidence role when explicitly activated:
- structure-repair before/after figures and worked examples

### Metrics and reporting
Reusable anchors:
- `wsovvis/metrics/ws_metrics.py`
- `wsovvis/metrics/ws_metrics_reporting_v1.py`
- `tests/test_ws_metrics_reporting_v1.py`
- `tests/test_ws_metrics_tooling.py`

Gap note:
- v9 mainline acceptance still depends on `HPR` and `UAR`.
- missing hidden-positive metrics are an evidence gap, not a reason to widen scope.

## 2. Mainline entrypoint policy
The v9 control plane treats the repo as code-first but gate-controlled.
If multiple code paths exist, the active one must be declared in `STATUS.md`; it must not be chosen ad hoc from historical code variety.

## 3. Legacy policy
The following remain in the repository as legacy / baseline / scaffold assets until explicitly retired:
- old Stage C mixed-representation semantic path
- old bounded open-world comparator path
- old Stage D marker-based scaffold path

They must not be treated as the active v9 mainline unless the docs explicitly say so.
