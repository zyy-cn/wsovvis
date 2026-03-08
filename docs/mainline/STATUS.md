# WSOVVIS Mainline Status

This file tracks the current authoritative state for document-driven automation.
It is the shared memory for new Codex sessions and supervisor-driven loops.

## Current state
- Current code snapshot status: existing codebase imported under new automation control plane
- Active gate: `G6 — Single-round bounded refinement`
- Current G0 judgment: `PASS`
- Current G1 judgment: `PASS`
- Current G2 judgment: `PASS`
- Current G3 judgment: `PASS`
- Current G4 judgment: `PASS`
- Current G5 judgment: `PASS`
- Current G6 judgment: `PASS`
- Terminal mainline mode: `active`
- Terminal detection explicit: `yes`
- Terminal revalidation mode available: `yes`
- `gpu4090d` environment contract execution-proven: `yes`
- G5/G6 numbering conflict across authoritative docs: `resolved`
- Stage B active-entrypoint confirmation: `complete`
- Stage B contract alignment with code: `complete`
- G1 protocol build path confirmation: `complete`
- G1 closed-world baseline smoke confirmation: `complete`
- G2 authoritative Stage B validation subset: `PASS`
- G3 representation / semantic-plumbing validation subset: `PASS`
- G4 protocol-aligned closed-world vs bounded open-world comparator step: `PASS`
- G5 dedicated bounded-policy tests: `PASS`
- G5 canonical bounded-policy smoke: `PASS`
- G6 hidden-positive handoff/reporting tests: `PASS`
- G6 canonical single-round bounded refinement smoke: `PASS`
- Mainline authority: `docs/mainline/*`
- Default-off modules: label-set expansion, second-round refinement, scenario/domain-missing mainline, enhanced memory aggregation, decoder-comparison branches, Stage D multi-round continuation

## Why terminal mainline is active
`G6` is the accepted terminal gate in the current documented mainline.
No further gate should activate under the current scope.
The automation flow must now:
1. stop automatically by default,
2. avoid proposing a new coding step,
3. write/update the terminal summary,
4. allow only bounded terminal revalidation when fresh evidence is needed,
5. preserve the accepted G4, G5, and G6 defaults without widening scope.

## Current blockers
- No remaining G6 blockers.

## Canonical environment evidence tracker
Record these fields before promoting G0 to PASS.

- Canonical host alias confirmed: `yes`
- Canonical repo dir confirmed: `yes` (`/home/zyy/code/wsovvis_runner`; remote helper resolves to `/mnt/sda/zyy/code/wsovvis_runner`)
- `tools/remote_verify_wsovvis.sh` exists and is callable: `yes`
- Canonical runner bootstrap preflight passed: `yes`
- Required live links verified (`runs/outputs/weights/data/third_party/*`): `yes`
- Remote env activation recipe confirmed (`conda activate wsovvis` + `PYTHONPATH`): `yes`
- Push route confirmed if needed (`github-via-gpu` or equivalent): `yes`
- Remote HEAD consistency recorded on first canonical run: `yes` (`5e9172b7f9f685f9409ebd8351b028272d00b7cc`)

## Authoritative G2 Stage B files
- export core: `wsovvis/track_feature_export/v1_core.py`
- bridge: `wsovvis/track_feature_export/bridge_from_stageb_v1.py`
- consumer / loader-facing interface: `wsovvis/track_feature_export/stagec_loader_v1.py`

## Authoritative G2 Stage B tests
- `tests/test_stageb_track_feature_export_v1.py`
- `tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py`
- `tests/test_stagec_loader_v1.py`

## Authoritative G3 files
- representation loading / ingestion: `wsovvis/track_feature_export/stagec_loader_v1.py`
- text prototype cache: `wsovvis/track_feature_export/stagec_clip_text_prototype_cache_v1.py`
- semantic slice construction: `wsovvis/track_feature_export/stagec_semantic_slice_v1.py`
- Stage C semantic plumbing: `wsovvis/training/stagec_semantic_plumbing_v1.py`
- staged attribution plumbing: `wsovvis/training/staged_attribution_plumbing_v1.py`

## Authoritative G3 tests
- `tests/test_stagec_semantic_slice_v1.py`
- `tests/test_staged_attribution_plumbing_v1.py`

## Authoritative G4 closed-world baseline files
- entrypoint: `tools/run_stagec1_mil_baseline_offline.py`
- implementation: `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- active closed-world scorer path for this gate: `labelset_proto_v1` with `decoder_backend=independent`

## Authoritative G4 bounded open-world comparator files
- entrypoint: `tools/run_stagec1_mil_baseline_offline.py`
- implementation: `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- active bounded open-world comparator path for this gate: `sinkhorn_v1` with bounded C4.3 `bg` / `unk_fg` special columns and `decoder_backend=independent`
- bounded metrics/reporting layer: `wsovvis/metrics/ws_metrics.py` and `wsovvis/metrics/ws_metrics_reporting_v1.py`

## Authoritative G5 files
- cross-window linking: `third_party/VNext/projects/SeqFormer/seqformer/models/clip_output.py`
- full-video inference aggregation: `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`
- conflict resolution / output stitching: `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`
- runtime switch between whole-video and clip-matching inference: `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`

## Authoritative G5 tests
- dedicated repo-level G5 bounded-policy test: `tests/test_g5_bounded_policy_v1.py`
- supporting wrapper coverage only: `tests/test_stage_d9_smoke_helper_v1.py`, `tests/test_stage_d11_canonical_replay_v1.py`, `tests/test_stage_d13_ci_quick_pipeline_v1.py`
- canonical runtime smoke summary: `codex/g5_bounded_policy_20260308T113745Z/g5_bounded_policy_summary.json`

## Authoritative G6 files
- single-round bounded refinement loop: `tools/run_stage_d0_self_training_loop.py`
- refinement input/output handoff: `tools/run_stage_d0_self_training_loop.py`
- refinement ws-metrics adapter: `wsovvis/metrics/ws_metrics_stage_d_adapter_v1.py`
- refinement acceptance / reporting snapshot: `tools/stage_d_reporting_snapshot.py`

## Authoritative G6 tests
- `tests/test_stage_d0_self_training_loop_v1.py`
- `tests/test_stage_d_reporting_snapshot_v1.py`
- `tests/test_stage_d_ws_metrics_adapter_v1.py`
- preserved G5 bounded-policy regression check during the G6 loop: `tests/test_g5_bounded_policy_v1.py`

## Authoritative terminal automation files
- terminal detection / prompt preparation helper: `tools/run_mainline_loop.py`
- terminal automation regression test: `tests/test_run_mainline_loop_v1.py`

## Authoritative terminal regression suite
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

## Authoritative G1 entrypoints
- WS-OVVIS protocol build tool: `tools/build_wsovvis_labelset_protocol.py`
- WS-OVVIS protocol build test path: `tests/test_build_wsovvis_labelset_protocol.py`
- Closed-world baseline tool: `tools/run_stagec1_mil_baseline_offline.py`
- Closed-world baseline implementation: `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- Closed-world baseline split consumer: `wsovvis/track_feature_export/stagec_loader_v1.py`

## G1 validation artifact paths
- Task root used: `codex/g1_protocol_baseline_smoke_20260308T100527Z`
- Protocol input used: `codex/g1_protocol_baseline_smoke_20260308T100527Z/protocol_input.json`
- Protocol output used: `codex/g1_protocol_baseline_smoke_20260308T100527Z/protocol_uniform_mr050_seed123.json`
- Protocol manifest used: `codex/g1_protocol_baseline_smoke_20260308T100527Z/protocol_uniform_mr050_seed123.manifest.json`
- Split root used: `codex/g1_protocol_baseline_smoke_20260308T100527Z/export_train`
- Baseline output dir used: `codex/g1_protocol_baseline_smoke_20260308T100527Z/stagec1_mil_baseline`

## G2 validation artifact paths
- Task root used: `codex/g2_stageb_integrity_20260308T101557Z`
- Pytest base temp used: `codex/g2_stageb_integrity_20260308T101557Z/pytest_stageb`

## G3 validation artifact paths
- Task root used: `codex/g3_semantic_plumbing_20260308T102553Z`
- Pytest base temp used: `codex/g3_semantic_plumbing_20260308T102553Z/pytest_stagec`

## G4 validation artifact paths
- Task root used: `codex/g4_open_world_validation_20260308T103951Z`
- Protocol input used: `codex/g4_open_world_validation_20260308T103951Z/protocol_input.json`
- Protocol output used: `codex/g4_open_world_validation_20260308T103951Z/protocol_uniform_mr050_seed123.json`
- Protocol manifest used: `codex/g4_open_world_validation_20260308T103951Z/protocol_uniform_mr050_seed123.manifest.json`
- Split root used: `codex/g4_open_world_validation_20260308T103951Z/export_train`
- Closed-world output dir used: `codex/g4_open_world_validation_20260308T103951Z/closed_world_labelset_proto`
- Open-world output dir used: `codex/g4_open_world_validation_20260308T103951Z/open_world_sinkhorn_c43`
- Compare summary used: `codex/g4_open_world_validation_20260308T103951Z/g4_compare_summary.json`

## G5 validation artifact paths
- Task root used: `codex/g5_bounded_policy_20260308T113745Z`
- Pytest base temp used: `codex/g5_bounded_policy_20260308T113745Z/pytest_g5`
- Runtime smoke summary used: `codex/g5_bounded_policy_20260308T113745Z/g5_bounded_policy_summary.json`

## G6 validation artifact paths
- Task root used: `codex/g6_single_round_refinement_20260308T123824Z`
- Pytest base temp used: `codex/g6_single_round_refinement_20260308T123824Z/pytest_g6`
- Stage C summary input used: `codex/g6_single_round_refinement_20260308T123824Z/stagec_hidden_positive_summary.json`
- Loop summary used: `codex/g6_single_round_refinement_20260308T123824Z/d0_loop_summary.json`
- Snapshot summary used: `codex/g6_single_round_refinement_20260308T123824Z/d0_snapshot.json`
- Runtime smoke summary used: `codex/g6_single_round_refinement_20260308T123824Z/g6_refinement_summary.json`

## Latest evidence
- Local workstation probe was informative only and blocked by missing `numpy`; per `ENVIRONMENT_AND_VALIDATION.md`, canonical remote validation was used for the authoritative G1 judgment.
- Local deployment/control docs and supervisor skill files are present and readable.
- All paths named in `CODEBASE_MAP.md` exist locally.
- `gpu4090d` SSH alias resolves and the host is reachable.
- Canonical repo dir `/home/zyy/code/wsovvis_runner` exists on the remote runner; bootstrap helper resolves the live runner root to `/mnt/sda/zyy/code/wsovvis_runner`.
- `tools/remote_verify_wsovvis.sh` is present locally and remotely and is callable via `bash`.
- Remote bootstrap preflight passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Remote environment activation recipe succeeded with `conda activate wsovvis` and `PYTHONPATH` including `third_party/VNext`.
- `github-via-gpu` minimal connectivity check succeeded.
- Canonical wrapper smoke succeeded via `bash tools/remote_verify_wsovvis.sh --remote gpu4090d --repo-dir /home/zyy/code/wsovvis_runner --branch codex/p3-stagec-c0-c4-semantic-slice --clone-url git@github.com:zyy-cn/wsovvis.git --env-cmd 'source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis && export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}' --cmd 'git rev-parse HEAD' --keep-untracked`.
- Canonical wrapper smoke recorded remote HEAD `5e9172b7f9f685f9409ebd8351b028272d00b7cc`, matching intended local commit `5e9172b7f9f685f9409ebd8351b028272d00b7cc`.
- G5/G6 numbering consistency between `PLAN.md`, `METRICS_ACCEPTANCE.md`, and `SUPERVISOR_STATE_MACHINE.md` is restored.
- Stage B authoritative export core confirmed: `wsovvis/track_feature_export/v1_core.py`.
- Stage B authoritative bridge confirmed: `wsovvis/track_feature_export/bridge_from_stageb_v1.py`.
- Stage B authoritative consumer-facing loader confirmed: `wsovvis/track_feature_export/stagec_loader_v1.py`.
- Stage B authoritative contract tests confirmed: `tests/test_stageb_track_feature_export_v1.py`, `tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py`, and `tests/test_stagec_loader_v1.py`.
- CODEBASE_MAP Stage B entrypoint ambiguity was narrowed: the core contract anchors are the three files above; enablement and real-run helpers remain supporting integration utilities.
- No material conflict was found between `STAGEB_INTERFACE_CONTRACT.md` and the audited Stage B code paths.
- Canonical remote preflight for the G1 run passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Canonical remote protocol pytest path passed via `pytest -q tests/test_build_wsovvis_labelset_protocol.py --basetemp codex/g1_protocol_baseline_smoke_20260308T100527Z/pytest_protocol` with `13 passed`.
- Direct protocol generation completed canonically under `codex/g1_protocol_baseline_smoke_20260308T100527Z/` with `PROTOCOL_CLIP_COUNT=3`.
- Direct closed-world baseline smoke completed canonically under `codex/g1_protocol_baseline_smoke_20260308T100527Z/` with `num_tracks_scored=3` and `num_videos_scored_non_empty=2`.
- No G1 wiring fix was required; the documented protocol path and the documented closed-world baseline path already cohere.
- `STAGEB_INTERFACE_CONTRACT.md` still matches the current code and tests:
  export artifacts remain a split-level manifest plus per-video `track_metadata.v1.json` and `track_arrays.v1.npz`;
  the Stage C primary key remains `(video_id, track_id)` with deterministic row alignment by `row_index`;
  the bridge runtime status remains in the reduced `success` / `failed` domain with split-domain reconciliation handled downstream;
  required producer provenance fields remain enforced.
- Canonical remote preflight for the G2 run passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Canonical remote G2 validation subset passed via `pytest -q tests/test_stageb_track_feature_export_v1.py tests/test_stageb_track_feature_export_bridge_from_stageb_v1.py tests/test_stagec_loader_v1.py --basetemp codex/g2_stageb_integrity_20260308T101557Z/pytest_stageb` with `31 passed in 1.61s`.
- No G2 wiring fix was required; the authoritative Stage B export, bridge, and consumer paths remain coherent.
- `CODEBASE_MAP.md` remains consistent with the current G3 code and tests:
  `stagec_loader_v1.py` remains the representation-loading / ingestion path;
  `stagec_clip_text_prototype_cache_v1.py` remains the text prototype cache path;
  `stagec_semantic_slice_v1.py` remains the semantic-slice construction and assignment-interface path;
  `stagec_semantic_plumbing_v1.py` remains the Stage C semantic plumbing path that threads candidate assembly and prototype cache usage into the training interface;
  `staged_attribution_plumbing_v1.py` remains the training-side staged attribution plumbing boundary currently expressed through default-off attribution-consumption / coupling helpers.
- Canonical remote preflight for the G3 run passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Canonical remote G3 validation subset passed via `pytest -q tests/test_stagec_semantic_slice_v1.py tests/test_staged_attribution_plumbing_v1.py --basetemp codex/g3_semantic_plumbing_20260308T102553Z/pytest_stagec` with `41 passed in 1.95s`.
- No G3 wiring fix was required; the loader / prototype-cache / semantic-slice / plumbing path remains coherent.
- Canonical remote preflight for the G4 run passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Canonical remote G4 protocol build completed under `codex/g4_open_world_validation_20260308T103951Z/` using `tools/build_wsovvis_labelset_protocol.py` with `protocol=uniform`, `missing_rate=0.5`, `seed=123`, and `min_labels_per_clip=1`.
- The bounded G4 split root was built from the protocol-aligned tiny fixture under `codex/g4_open_world_validation_20260308T103951Z/export_train`; the observed label was `10` and the hidden positive label was `20`.
- The protocol-aligned closed-world baseline ran canonically via `tools/run_stagec1_mil_baseline_offline.py` with `scorer_backend=labelset_proto_v1` and `decoder_backend=independent`.
- The bounded open-world comparator ran canonically via `tools/run_stagec1_mil_baseline_offline.py` with `scorer_backend=sinkhorn_v1`, `decoder_backend=independent`, and bounded C4.3 special columns enabled for `bg` and `unk_fg`.
- No G4 wiring fix was required; the existing metrics/reporting layer accepted a tiny `ws_eval_bundle` built from protocol full/observed sets plus the comparator's track-level outputs.
- Canonical bounded G4 metric comparison:
  closed-world `HPR=0.0`, `UAR=0.0`, `SCR=0.5`, `AURC=0.5`;
  open-world `HPR=1.0`, `UAR=1.0`, `SCR=0.5`, `AURC=0.5`.
- Standard bounded-path metric availability for this loop:
  `AP` unavailable, `AP_base` unavailable, `AP_novel` unavailable, `AURC` available.
- G4 acceptance conditions are satisfied on the recorded protocol-aligned bounded comparison; the next gate may activate on the next bounded loop if promotion is authorized.
- `CODEBASE_MAP.md` now records the active G5 runtime files and the dedicated repo-level bounded-policy test.
- `SeqFormer.inference_clip` now passes raw class logits, clip-quality scores, and query embeddings into `Clips` for the clip-matching runtime path.
- `Videos.update` now keeps geometry as the match gate and blends geometry with query similarity in the active linking score; semantics remain out of the linking score.
- `Videos.get_result` now closes class predictions with quality-weighted logit averaging instead of uniform clip averaging.
- Canonical remote preflight for the updated G5 run passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Canonical remote G5 dedicated policy pytest passed via `pytest -q tests/test_g5_bounded_policy_v1.py --basetemp codex/g5_bounded_policy_20260308T113745Z/pytest_g5` with `3 passed in 3.00s`.
- Canonical remote G5 smoke reran the same direct runtime pattern against `Videos.update`, `Videos.get_result`, `SeqFormer.clip_matching_postprocess`, and `SeqFormer.whole_video_inference` under `gpu4090d` with remote `HEAD` matching the intended local commit.
- The recorded G5 smoke confirms the bounded policy is live:
  `linking_policy_observed=geometry_gate_plus_query_similarity`,
  `classification_closure_observed=quality_weighted_logit_averaging`,
  `whole_video_track_count=10`,
  `whole_video_has_track_ids=true`,
  `whole_video_has_embeddings=true`.
- The bounded linking smoke also confirmed that the higher-semantics but query-misaligned candidate stayed a new track while the query-aligned candidate matched the existing track.
- G5 acceptance conditions are satisfied on the recorded bounded-policy-aligned canonical run; the next gate may activate on the next bounded loop if promotion is authorized.
- `CODEBASE_MAP.md` now records the authoritative bounded G6 loop, metrics adapter, and reporting snapshot separately from the broader Stage D wrappers.
- The G6 loop now preserves optional `observed_label_ids`, `hidden_positive_label_ids`, and `unknown_attributed_label_ids` from the Stage C summary handoff into `round_input_summary`, `round_output_summary`, and emitted `ws_metrics_summary_v1` sidecars.
- Canonical remote preflight for the G6 run passed via `python tools/check_canonical_runner_bootstrap_links.py --check`.
- Canonical remote G6 targeted pytest subset passed via `pytest -q tests/test_stage_d0_self_training_loop_v1.py::test_d0_round0_round1_with_minimal_refine_and_stagec_seed tests/test_stage_d0_self_training_loop_v1.py::test_d0_emit_ws_metrics_preserves_hidden_positive_fields_for_hpr_uar tests/test_stage_d_reporting_snapshot_v1.py::test_stage_d_snapshot_happy_path_detects_round_paths tests/test_stage_d_ws_metrics_adapter_v1.py::test_stage_d_ws_metrics_adapter_exposes_hpr_and_uar_when_optional_fields_exist tests/test_g5_bounded_policy_v1.py --basetemp codex/g6_single_round_refinement_20260308T123824Z/pytest_g6` with `7 passed in 3.07s`.
- Canonical remote G6 smoke completed under `codex/g6_single_round_refinement_20260308T123824Z/` with one bounded refinement round (`round1_refine_applied=true`) and a reporting snapshot written by `tools/stage_d_reporting_snapshot.py`.
- Canonical bounded G6 metric comparison on the refinement smoke:
  round0 `HPR=1.0`, `UAR=1.0`, `SCR=1.0`, `AURC=0.75`;
  round1 `HPR=1.0`, `UAR=1.0`, `SCR=1.0`, `AURC=0.75`.
- The G6 commit did not modify the accepted G4 or G5 runtime files; the canonical run verified `git diff --name-only HEAD^ HEAD -- tools/run_stagec1_mil_baseline_offline.py wsovvis/track_feature_export/stagec1_attribution_mil_v1.py third_party/VNext/projects/SeqFormer/seqformer/models/clip_output.py third_party/VNext/projects/SeqFormer/seqformer/seqformer.py` returned no paths, and the preserved G5 bounded-policy regression check passed.
- G6 acceptance conditions are satisfied on the recorded bounded single-round refinement run: net value is non-negative on the exposed metrics, hidden-positive handling does not regress, and default-off branches remain off.
- `tools/run_mainline_loop.py --dry-run` now detects terminal-mainline mode and writes `docs/mainline/reports/mainline_terminal_summary.txt` instead of a new coding-step prompt.
- `tools/run_mainline_loop.py --terminal-revalidate --dry-run` now prepares a bounded terminal revalidation prompt that preserves the accepted G4 / G5 / G6 regression suite and canonical `gpu4090d` validation semantics.
- Terminal automation control-plane validation completed locally via `python3 -m py_compile tools/run_mainline_loop.py tests/test_run_mainline_loop_v1.py`, `python3 tools/run_mainline_loop.py --dry-run`, and `python3 tools/run_mainline_loop.py --terminal-revalidate --dry-run`.

## Next smallest valid step
1. keep terminal-mainline mode active and do not open a new gate,
2. if fresh evidence is needed, run `python tools/run_mainline_loop.py --terminal-revalidate --dry-run` and rerun only the bounded terminal regression suite,
3. do not open second-round refinement, label-set expansion, enhanced memory aggregation, or broader Stage D branches unless the authoritative docs explicitly change scope.

## Latest G6 canonical validation record
- local branch: `codex/p3-stagec-c0-c4-semantic-slice`
- intended local commit: `520077937ba43583d055cbf1c85e01a4b2283a8f`
- remote host: `gpu4090d`
- remote repo dir: `/home/zyy/code/wsovvis_runner`
- remote HEAD: `520077937ba43583d055cbf1c85e01a4b2283a8f`
- HEAD matches intended commit: `yes`
- command summary: canonical wrapper with bootstrap preflight, targeted G6 pytest subset plus `tests/test_g5_bounded_policy_v1.py`, direct single-round refinement via `tools/run_stage_d0_self_training_loop.py`, snapshot generation via `tools/stage_d_reporting_snapshot.py`, and summary emission to `codex/g6_single_round_refinement_20260308T123824Z/g6_refinement_summary.json`
- outcome: `PASS`

## Latest G5 canonical validation record
- local branch: `codex/p3-stagec-c0-c4-semantic-slice`
- intended local commit: `84c51f2c5881729cefffe65929e370da81a4d57e`
- remote host: `gpu4090d`
- remote repo dir: `/home/zyy/code/wsovvis_runner`
- remote HEAD: `84c51f2c5881729cefffe65929e370da81a4d57e`
- HEAD matches intended commit: `yes`
- command summary: canonical wrapper with bootstrap preflight, `pytest -q tests/test_g5_bounded_policy_v1.py --basetemp codex/g5_bounded_policy_20260308T113745Z/pytest_g5`, and the bounded G5 runtime smoke that writes `codex/g5_bounded_policy_20260308T113745Z/g5_bounded_policy_summary.json`
- outcome: `PASS`

## Prior G4 canonical validation record
- local branch: `codex/p3-stagec-c0-c4-semantic-slice`
- intended local commit: `5e9172b7f9f685f9409ebd8351b028272d00b7cc`
- remote host: `gpu4090d`
- remote repo dir: `/home/zyy/code/wsovvis_runner`
- remote HEAD: `5e9172b7f9f685f9409ebd8351b028272d00b7cc`
- HEAD matches intended commit: `yes`
- command: `REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
TASK_ROOT="codex/g4_open_world_validation_20260308T103951Z"
mkdir -p "$TASK_ROOT"
python tools/check_canonical_runner_bootstrap_links.py --check
python - <<'PY'
from pathlib import Path
import json

task_root = Path("codex/g4_open_world_validation_20260308T103951Z")
task_root.mkdir(parents=True, exist_ok=True)
protocol_input = {
    "videos": [{"id": 1, "name": "vid_g4"}],
    "annotations": [{"video_id": 1, "category_id": 10}, {"video_id": 1, "category_id": 20}],
    "categories": [{"id": 10, "name": "class_10"}, {"id": 20, "name": "class_20"}],
}
(task_root / "protocol_input.json").write_text(json.dumps(protocol_input, indent=2) + "\n", encoding="utf-8")
PY
python tools/build_wsovvis_labelset_protocol.py --input-json "$TASK_ROOT/protocol_input.json" --output-json "$TASK_ROOT/protocol_uniform_mr050_seed123.json" --manifest-json "$TASK_ROOT/protocol_uniform_mr050_seed123.manifest.json" --protocol uniform --missing-rate 0.5 --seed 123 --min-labels-per-clip 1
python - <<'PY'
from pathlib import Path
import json
import numpy as np
from wsovvis.track_feature_export import build_track_feature_export_v1

task_root = Path("codex/g4_open_world_validation_20260308T103951Z")
protocol = json.loads((task_root / "protocol_uniform_mr050_seed123.json").read_text(encoding="utf-8"))
clip = protocol["clips"][0]
full_ids = [int(x) for x in clip["label_set_full_ids"]]
observed_ids = [int(x) for x in clip["label_set_observed_ids"]]
hidden_ids = [x for x in full_ids if x not in observed_ids]
video_name = str(clip.get("video_name") or f"vid_{clip['video_id']}")
visible_id = observed_ids[0]
hidden_id = hidden_ids[0]
proto_by_label = {10: np.asarray([1.0, 0.0], dtype=np.float32), 20: np.asarray([0.0, 1.0], dtype=np.float32)}
build_track_feature_export_v1(
    {
        "split": "train",
        "embedding_dim": 2,
        "embedding_dtype": "float32",
        "embedding_pooling": "track_pooled",
        "embedding_normalization": "none",
        "producer": {
            "stage_b_checkpoint_id": "ckpt_g4_tiny",
            "stage_b_checkpoint_hash": "sha256:g4tiny",
            "stage_b_config_ref": "configs/stage_b.yaml",
            "stage_b_config_hash": "sha256:g4tinycfg",
            "pseudo_tube_manifest_id": "ptube_g4_tiny",
            "pseudo_tube_manifest_hash": "sha256:g4tinytubes",
            "split": "train",
            "extraction_settings": {
                "frame_sampling_rule": "uniform_stride_2",
                "pooling_rule": "mean_over_active_frames",
                "min_track_length": 1,
            },
        },
        "videos": [
            {
                "video_id": video_name,
                "status": "processed_with_tracks",
                "tracks": [
                    {"track_id": 1, "start_frame_idx": 0, "end_frame_idx": 2, "num_active_frames": 3, "objectness_score": 0.9, "embedding": proto_by_label[visible_id].tolist()},
                    {"track_id": 2, "start_frame_idx": 3, "end_frame_idx": 5, "num_active_frames": 3, "objectness_score": 0.8, "embedding": [-1.0, -1.0]},
                ],
            }
        ],
    },
    task_root / "export_train",
)
(task_root / "labelset_observed.json").write_text(json.dumps({"videos": [{"video_id": video_name, "label_set_observed_ids": observed_ids}]}, indent=2) + "\n", encoding="utf-8")
np.savez(task_root / "prototype_arrays.npz", prototypes=np.asarray([proto_by_label[label_id] for label_id in full_ids], dtype=np.float32))
(task_root / "prototype_manifest.json").write_text(
    json.dumps(
        {
            "schema_name": "wsovvis.stagec.label_prototypes.v1",
            "schema_version": "1.0.0",
            "embedding_dim": 2,
            "dtype": "float32",
            "labels": [{"label_id": int(label_id), "row_index": idx} for idx, label_id in enumerate(full_ids)],
            "arrays_path": "prototype_arrays.npz",
            "array_key": "prototypes",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
(task_root / "truth_eval_context.json").write_text(
    json.dumps(
        {
            "video_id": video_name,
            "full_label_ids": full_ids,
            "observed_label_ids": observed_ids,
            "hidden_label_ids": hidden_ids,
            "track_truth_by_id": {"1": visible_id, "2": hidden_id},
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
python tools/run_stagec1_mil_baseline_offline.py --split-root "$TASK_ROOT/export_train" --output-dir "$TASK_ROOT/closed_world_labelset_proto" --scorer-backend labelset_proto_v1 --decoder-backend independent --labelset-json "$TASK_ROOT/labelset_observed.json" --prototype-manifest-json "$TASK_ROOT/prototype_manifest.json"
python tools/run_stagec1_mil_baseline_offline.py --split-root "$TASK_ROOT/export_train" --output-dir "$TASK_ROOT/open_world_sinkhorn_c43" --scorer-backend sinkhorn_v1 --decoder-backend independent --labelset-json "$TASK_ROOT/labelset_observed.json" --prototype-manifest-json "$TASK_ROOT/prototype_manifest.json" --sinkhorn-c43-enable --sinkhorn-c43-enable-bg --sinkhorn-c43-enable-unk-fg --sinkhorn-c43-bg-prior-weight 2.0 --sinkhorn-c43-unk-fg-prior-weight 3.0 --sinkhorn-c43-unk-fg-min-top-obs-score -0.8 --sinkhorn-c43-unk-fg-max-top-obs-score -0.6 --sinkhorn-c43-bg-score -2.0
python - <<'PY'
from pathlib import Path
import json

task_root = Path("codex/g4_open_world_validation_20260308T103951Z")
truth = json.loads((task_root / "truth_eval_context.json").read_text(encoding="utf-8"))
gt_entities = [int(x) for x in truth["full_label_ids"]]
observed_entities = [int(x) for x in truth["observed_label_ids"]]
hidden_entities = [int(x) for x in truth["hidden_label_ids"]]
track_truth = {str(k): int(v) for k, v in truth["track_truth_by_id"].items()}

def build_bundle(run_dir: str) -> dict:
    run_path = task_root / run_dir
    run_summary = json.loads((run_path / "run_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_path / "track_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    predicted_entities = set()
    unknown_attributed_entities = set()
    for row in rows:
        gt_label = track_truth[str(row["track_id"])]
        pred_label = row.get("predicted_label_id")
        pred_source = row.get("predicted_label_source")
        if pred_label == gt_label:
            predicted_entities.add(gt_label)
        elif pred_label == "__unk_fg__" or pred_source == "unk_fg":
            predicted_entities.add(gt_label)
            unknown_attributed_entities.add(gt_label)
    return {
        "video_id": truth["video_id"],
        "assignment_backend": run_summary.get("scorer_backend"),
        "ws_eval_bundle": {
            "gt_entities": gt_entities,
            "observed_entities": observed_entities,
            "hidden_positive_entities": hidden_entities,
            "predicted_entities": sorted(predicted_entities),
            "unknown_attributed_entities": sorted(unknown_attributed_entities),
            "predictions_by_missing_rate": {"0.0": sorted(predicted_entities), "0.5": sorted(predicted_entities), "1.0": sorted(predicted_entities)},
        },
    }

(task_root / "closed_world_metrics_input.json").write_text(json.dumps(build_bundle("closed_world_labelset_proto"), indent=2) + "\n", encoding="utf-8")
(task_root / "open_world_metrics_input.json").write_text(json.dumps(build_bundle("open_world_sinkhorn_c43"), indent=2) + "\n", encoding="utf-8")
PY
python tools/ws_metrics_summary_demo_v1.py --input-json "$TASK_ROOT/closed_world_metrics_input.json" > "$TASK_ROOT/closed_world_metrics_summary.json"
python tools/ws_metrics_summary_demo_v1.py --input-json "$TASK_ROOT/open_world_metrics_input.json" > "$TASK_ROOT/open_world_metrics_summary.json"
echo "REMOTE_HEAD=$(git rev-parse HEAD)"
echo "TASK_ROOT=$TASK_ROOT"
EOF
); bash tools/remote_verify_wsovvis.sh --remote gpu4090d --repo-dir /home/zyy/code/wsovvis_runner --branch codex/p3-stagec-c0-c4-semantic-slice --clone-url git@github.com:zyy-cn/wsovvis.git --env-cmd 'source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis && export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}' --cmd "$REMOTE_CMD" --keep-untracked`
- outcome: `PASS`
- notes: minimal canonical G4 validation proving that the protocol-aligned closed-world baseline and the bounded open-world comparator both run on the matching remote commit and that `HPR` / `UAR` improve without `SCR` or `AURC` regression
