# WSOVVIS Worked Example Verification

Timestamp (UTC): 2026-03-08T16:04:38Z
Active gate: `G6 — Single-round bounded refinement` in terminal-mainline revalidation mode
Local branch: `codex/p3-stagec-c0-c4-semantic-slice`
Intended local commit: `23748b3c348f0745bdee82df43096be95c6b4dc7`
Canonical remote host: `gpu4090d`
Canonical remote repo dir: `/home/zyy/code/wsovvis_runner`
Remote HEAD: `23748b3c348f0745bdee82df43096be95c6b4dc7`
Remote HEAD matches intended local commit: `yes`
Canonical task root: `codex/worked_example_verification_20260308T155833Z`

## Why This Worked Example

This report is tied to the outline's mainline claim and scope:

- Outline section 2.0: prove that under incomplete positive label sets, closed-world supervision collapses hidden positives to background while bounded open-world attribution can preserve them.
- Outline section 2.5: use open-world set-to-track attribution with explicit `bg` and `unk-fg`.
- Outline section 2.7: use geometry-gated plus query-resolved linking and quality-weighted logit averaging for full-video closure.
- Outline section 2.6: keep only single-round bounded refinement and keep label-set expansion and second-round refinement off.

The selected case is the smallest protocol-aligned example that exposes the hidden-positive phenomenon directly:

- source record: one synthetic video record, `video_id=1`, `video_name=video_worked_example_hp`
- full label set `Y*={10,20}`
- observed label set `Y'={10}`
- hidden positive set `{20}`

Limitation up front:

- This is a protocol-aligned synthetic worked example, not a raw LV-VIS clip replay.
- The current executable tiny path is video-level rather than fixed temporal-window protocol; the protocol tool emits one per-video `clip` record here.

## Canonical Execution

Measured:

- Canonical wrapper used: `tools/remote_verify_wsovvis.sh`
- Bootstrap preflight passed: `python tools/check_canonical_runner_bootstrap_links.py --check`
- Canonical execution ran inside `/home/zyy/code/wsovvis_runner`
- Remote HEAD matched intended local commit

Exact wrapper commands used:

```bash
bash tools/remote_verify_wsovvis.sh \
  --remote gpu4090d \
  --repo-dir /home/zyy/code/wsovvis_runner \
  --branch codex/p3-stagec-c0-c4-semantic-slice \
  --clone-url git@github.com:zyy-cn/wsovvis.git \
  --env-cmd 'source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis && export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}' \
  --cmd "$REMOTE_CMD" \
  --keep-untracked
```

```bash
bash tools/remote_verify_wsovvis.sh \
  --remote gpu4090d \
  --repo-dir /home/zyy/code/wsovvis_runner \
  --branch codex/p3-stagec-c0-c4-semantic-slice \
  --clone-url git@github.com:zyy-cn/wsovvis.git \
  --env-cmd 'source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis && export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}' \
  --cmd "$REMOTE_CMD" \
  --keep-untracked
```

The first wrapper command executed the full worked example and wrote remote artifacts under `codex/worked_example_verification_20260308T155833Z/`. The second wrapper command retrieved a compact artifact bundle for local reporting.

## Stage 0 — Raw Protocol Input

Measured:

- raw input video record:
  `video_id=1`, `name=video_worked_example_hp`, `frame_count=6`, `height=2`, `width=2`
- protocol builder parameters:
  `protocol=uniform`, `missing_rate=0.5`, `seed=123`, `min_labels_per_clip=1`
- protocol output record:
  `label_set_full_ids=[10,20]`
  `label_set_observed_ids=[10]`
  `num_full=2`
  `num_observed=1`

Inferred from `tools/build_wsovvis_labelset_protocol.py`:

- the tool first builds the full positive set from all annotations for the video
- under `uniform` missing, it samples the observed set from the full set while respecting the keep floor
- in this tiny executable path, the protocol operates on one video-level record rather than a real sliding LV-VIS window

## Stage 1 — Stage B Class-Agnostic Instance Basis

Authoritative files used:

- export core: `wsovvis/track_feature_export/v1_core.py`
- loader: `wsovvis/track_feature_export/stagec_loader_v1.py`

Measured artifact paths:

- split root: `codex/worked_example_verification_20260308T155833Z/export_train`
- metadata shard: `videos/video_worked_example_hp/track_metadata.v1.json`
- array shard: `videos/video_worked_example_hp/track_arrays.v1.npz`

Measured Stage B basis:

| row_index | track_id | start | end | num_active_frames | objectness_score | embedding |
|---|---:|---:|---:|---:|---:|---|
| 0 | 1 | 0 | 2 | 3 | 0.9 | `[1.0, 0.0]` |
| 1 | 2 | 3 | 5 | 3 | 0.8 | `[-1.0, -1.0]` |

Measured tensor shape:

- embeddings: `[N_track, D] = [2, 2]`
- `N_track=2`
- `D=2`

Interpretation:

- row 0 is the observed-positive-aligned track basis
- row 1 is the hidden-positive candidate basis used to test background-vs-unknown handling

## Stage 2 — Stage C Semantic Batch Construction

Authoritative files used:

- loader: `wsovvis/track_feature_export/stagec_loader_v1.py`
- text prototype cache: `wsovvis/track_feature_export/stagec_clip_text_prototype_cache_v1.py`
- semantic slice construction: `wsovvis/track_feature_export/stagec_semantic_slice_v1.py`
- semantic plumbing: `wsovvis/training/stagec_semantic_plumbing_v1.py`
- staged attribution plumbing: `wsovvis/training/staged_attribution_plumbing_v1.py`

Measured semantic batch:

- candidate labels: `[10, "__bg__", "__unk_fg__"]`
- `track_features shape = [2, 2]`
- `prototype_features shape = [3, 2]`
- `candidate_matrix shape = [2, 3]`
- `valid_track_mask shape = [2]`
- `valid_column_mask shape = [3]`

Measured excerpts:

- track features:
  row 0: `[1.0, 0.0]`
  row 1: `[-1.0, -1.0]`
- prototype features from the cache helper:
  row 0 (`10`): `[-0.131977, -0.991253]`
  row 1 (`__bg__`): `[0.324055, -0.946038]`
- candidate matrix:

```text
[[1.0, 1.0, 1.0],
 [1.0, 1.0, 1.0]]
```

- minimal C2 sinkhorn assignment excerpt from the semantic-slice helper:

```text
[[0.000018, 0.251263, 0.748719],
 [0.500062, 0.498776, 0.001162]]
```

Important limitation:

- This Stage C semantic batch uses the current deterministic pseudo-prototype cache backend, not a live CLIP text encoder call.
- The semantic batch is therefore a structural demonstration of loader/candidate/prototype plumbing, not the exact scoring object used by the G4 comparator below.

## Stage 3 — Closed-World vs Bounded Open-World Comparator

Authoritative entrypoints:

- closed-world baseline entrypoint: `tools/run_stagec1_mil_baseline_offline.py`
- closed-world implementation: `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- bounded open-world comparator entrypoint: `tools/run_stagec1_mil_baseline_offline.py`
- bounded open-world comparator implementation: `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`

Measured closed-world result (`labelset_proto_v1`, `decoder_backend=independent`):

| track_id | predicted_label_id | score |
|---:|---:|---:|
| 1 | 10 | 1.0 |
| 2 | 10 | -0.707107 |

Measured bounded open-world result (`sinkhorn_v1`, bounded C4.3 `bg` + `unk_fg`, `decoder_backend=independent`):

| track_id | predicted_label_id | predicted_label_source | score | bg posterior | unk_fg posterior | top observed score |
|---:|---|---|---:|---:|---:|---:|
| 1 | `__bg__` | `bg` | 0.63959 | 0.63959 | 0.0 | 0.36041 |
| 2 | `__unk_fg__` | `unk_fg` | 0.930123 | 0.069877 | 0.930123 | 0.0 |

Measured sinkhorn run summary:

- active special columns: `["__bg__", "__unk_fg__"]`
- policy version: `sinkhorn_v1.r2`
- `c43_num_tracks_bg=1`
- `c43_num_tracks_unk_fg=1`
- `c43_unk_fg_eligible_track_count_total=1`

Inferred but not directly emitted:

- effective open-world posterior shape is `[2, 3]`
- columns are `[10, "__bg__", "__unk_fg__"]`
- the full posterior matrix is not directly written to the current offline artifact set; only row-level top score and special-column posteriors are emitted

Interpretation:

- the closed-world baseline collapses the hidden-positive track (track 2) onto observed label `10`
- the bounded open-world comparator does not recover the hidden class name `20`; instead it preserves the hidden-positive track by routing it into `__unk_fg__`
- that is exactly the hidden-positive rescue behavior the outline prioritizes

## Stage 4 — G4 Evidence and Metric Computation

Authoritative metric files:

- `wsovvis/metrics/ws_metrics.py`
- `wsovvis/metrics/ws_metrics_reporting_v1.py`

Measured metric inputs:

- closed-world `predicted_entities=[10]`
- open-world `predicted_entities=[20]`
- hidden-positive entities in both cases: `[20]`
- open-world `unknown_attributed_entities=[20]`

Metric formulas inferred from code:

- `SCR = |Y* ∩ Y_hat| / |Y*|`
- `HPR = SCR(hidden_positive_entities, predicted_entities)`
- `UAR = SCR(hidden_positive_entities, unknown_attributed_entities)`
- `AURC` is the normalized trapezoidal integral over the missing-rate recall curve

Measured metrics:

| variant | HPR | UAR | SCR | AURC | AP | AP_base | AP_novel |
|---|---:|---:|---:|---:|---|---|---|
| closed-world | 0.0 | 0.0 | 0.5 | 0.5 | not directly available in current path | not directly available in current path | not directly available in current path |
| bounded open-world | 1.0 | 1.0 | 0.5 | 0.5 | not directly available in current path | not directly available in current path | not directly available in current path |

Meaning relative to the core claim:

- directly supported: the bounded open-world path preserves the hidden positive while the closed-world path does not
- indirectly supported: `SCR` and `AURC` do not improve, but they also do not regress in this tiny case
- limitation: the open-world path rescues only the hidden positive in this example; it does not simultaneously retain the observed positive in `predicted_entities`

## Stage 5 — G5 Full-Video Linking / Inference Closure

Authoritative G5 files:

- `third_party/VNext/projects/SeqFormer/seqformer/models/clip_output.py`
- `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`

Measured bounded-policy replay on the same label semantics (`10`, `20`):

- linking policy observed: `geometry_gate_plus_query_similarity`
- classification closure observed: `quality_weighted_logit_averaging`
- after clip 1: `num_inst=1`
- after clip 2: `num_inst=2`
- final per-track class probabilities:

```text
track 0: [0.759599, 0.240401]
track 1: [0.05, 0.95]
```

- saved query embeddings:

```text
track 0: [1.0, 0.0]
track 1: [-1.0, 0.0]
```

Measured/inferred interpretation:

- one second-clip candidate was linked to the existing track because its query embedding aligned with the first clip
- the higher-semantic but query-misaligned candidate became a new global track
- the final class closure used quality-weighted logit averaging, not simple mean pooling and not memory aggregation

Important limitation:

- the current G5 bounded-policy path operates on runtime clip outputs, not a literal Stage C offline artifact handoff
- this stage therefore demonstrates the accepted G5 policy on the same label semantics, not a fully fused Stage C -> G5 executable chain

## Stage 6 — G6 Single-Round Bounded Refinement

Authoritative G6 files:

- loop: `tools/run_stage_d0_self_training_loop.py`
- snapshot: `tools/stage_d_reporting_snapshot.py`
- metrics adapter: `wsovvis/metrics/ws_metrics_stage_d_adapter_v1.py`

Measured Stage C seed handed into G6:

- `selected_positive_label_ids=[10,20]`
- `observed_label_ids=[10]`
- `hidden_positive_label_ids=[20]`
- `unknown_attributed_label_ids=[20]`
- `final.candidate_label_ids=[20]`

Measured bounded loop outcome:

- rounds executed: `2` (`round0` seed + `round1` single refine pass)
- round policy: `minimal_curriculum_v1`
- round 0 output candidate labels: `[20]`
- round 1 output candidate labels: `[20, 909001]`
- round 1 added label ids: `[909001]`
- single-round refinement confirmed: `yes`
- second-round refinement active: `no`
- label-set expansion active: `no`

Measured ws-metrics sidecars:

| round | HPR | UAR | SCR | AURC |
|---|---:|---:|---:|---:|
| 0 | 1.0 | 1.0 | 0.5 | 0.375 |
| 1 | 1.0 | 1.0 | 0.5 | 0.375 |

Why Stage 6 `AURC` is `0.375` instead of the Stage 4 `0.5`:

- measured from the current adapter implementation, the Stage D path rebuilds `predictions_by_missing_rate` from the current `candidate_label_ids`
- round 0 has one candidate label, so the synthetic missing-rate curve becomes:
  `0.0 -> [20]`, `0.5 -> [20]`, `1.0 -> []`
- that yields SCR values `0.5, 0.5, 0.0` and therefore `AURC=0.375`

Clarification about `909001`:

- `909001` is the deterministic marker label added by the current bounded smoke refinement path
- it is not the mainline label-set expansion module described as default-off in the outline and mainline docs
- this is why the report classifies G6 as partly scaffolded even though the bounded single-round loop executed correctly

## Does the worked example support the current outline’s core claim?

1. What exact part of the claim is directly supported?

- Directly supported: under incomplete positive supervision (`Y'={10}` inside `Y*={10,20}`), the bounded open-world comparator preserved the hidden positive `20` via `__unk_fg__` while the closed-world baseline did not.
- Direct evidence: `HPR` improved from `0.0` to `1.0`, `UAR` improved from `0.0` to `1.0`, and `SCR` / `AURC` did not worsen in this tiny case.
- G6 also directly supported that the current bounded single-round loop preserves the hidden-positive fields and does not regress `HPR` / `UAR`.

2. What exact part is only indirectly supported?

- Indirectly supported: the accepted G5 bounded policy is executable and consistent with the outline (`geometry + query` linking, quality-weighted logit closure), but the current executable path does not directly pass Stage C offline outputs into G5 runtime linking.
- Indirectly supported: the pipeline remains coherent across Stage B, Stage C, G4, G5, and G6 on one shared label semantics case, even though not every handoff is a literal tensor-to-tensor runtime bridge.

3. What, if anything, remains only scaffolded or partially implemented in the current executable path?

- real LV-VIS worked-example replay is not used here; the case is synthetic but protocol-aligned
- Stage C semantic prototypes are deterministic cache vectors rather than a live CLIP text encoder path
- AP / `AP_base` / `AP_novel` are not directly available in this bounded worked-example path
- the G6 smoke path still uses a deterministic refine-marker label (`909001`), which is bounded scaffolding rather than a full quality-aware pseudo-tube regeneration pipeline

