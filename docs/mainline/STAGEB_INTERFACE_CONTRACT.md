# WS-OVVIS Stage-B / Bank Export Interface Contract

This contract documents the repository interface between the basis-generator side and the bank-backed semantic pipeline.

## Purpose
WS-OVVIS treats exported query trajectories and semantic-carrier artifacts as durable handoff objects. Any gate that consumes stage-B / bank-export artifacts must use a documented schema rather than ad hoc tensors or notebook-only state.

## Minimum object contract
A durable export should preserve, directly or through a manifest:
- `video_id`
- stable `query_id` / track identifier
- frame indices or temporal span
- masks and/or boxes sufficient to reconstruct the structural object
- confidence or quality score
- source model/config/version metadata
- path to semantic feature arrays when applicable

## Authoritative repository entrypoints
- `tools/build_stageb_feature_export_enablement_v1.py`
- `tools/build_stageb_track_feature_export_v1.py`
- `tools/build_stageb_track_feature_export_from_stageb_bridge_v1.py`
- `tools/build_stageb_bridge_input_from_real_stageb_sidecar_v1.py`
- `tools/validate_stageb_track_feature_export_v1.py`
- `tests/test_stageb_feature_export_enablement_v1.py`
- `tests/test_stageb_track_feature_export_v1.py`
- `tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py`
- `tests/test_real_stageb_sidecar_bridge_input_builder_v1.py`

## Control-plane rule
If a gate depends on bank-backed export behavior, the active gate evidence must point to the exact manifest and output root being judged. A gate may not pass on verbal claims that export “should work later.”
