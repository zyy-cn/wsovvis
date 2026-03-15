# WS-OVVIS Mainline Plan

## 1. Purpose
This file is the source of truth for the current project automation mainline.

Main claim to prove:

> Under clip-level incomplete positive evidence Y'(v), prove that a frozen video-native class-agnostic basis, materialized query-trajectory and semantic-carrier banks, seen visual prototypes, a class-level text map, and coverage-aware open-world attribution are sufficient for bag-free text-conditioned open-vocabulary video instance inference without using the observed label bag at test time.

The automation system must prioritize proving this claim over expanding the system.

## 2. Mainline phases and gates
### G0 — Authority switch and canonical environment inheritance
Goal: install the privatized control plane against the actual repository, verify the canonical wrapper/bootstrap contract, and record the first evidence-backed repository state.  
Authoritative code paths: `tools/run_mainline_loop.py`, `tools/check_canonical_runner_bootstrap_links.py`, `tools/remote_verify_wsovvis.sh`, `tests/test_run_mainline_loop_v1.py`.  
Exit intent: the repo can run one bounded supervisor loop and the canonical environment facts are explicitly recorded.

### G1 — Weak-label protocol contract for `Y'(v)`
Goal: prove the clip-level incomplete-positive-evidence protocol is deterministic, reviewable, and aligned with the outline’s training assumption.  
Authoritative code paths: `tools/build_wsovvis_labelset_protocol.py`, `tests/test_build_wsovvis_labelset_protocol.py`.  
Exit intent: reproducible protocol artifacts exist with a worked example and manifest.

### G2 — Frozen video-native basis generator integration and raw-structure evidence
Goal: prove the DINOv2-VideoMask2Former path can train/export stable non-empty query trajectories and collect the structure-only S1 evidence bundle at original thresholds.  
Authoritative code paths: `train_seqformer_pseudo.py`, `configs/seqformer_pseudo_s1_canonical.yaml`, `tools/run_s1_basis_superiority_eval.py`, `tests/test_s1_basis_superiority_v1.py`.  
Exit intent: the raw basis path is integrated, exportable, and scientifically reviewable without activating refinement branches.

### G3 — Query-Trajectory Bank and Semantic Carrier Bank materialization
Goal: prove query trajectories and trajectory-level DINO carriers are durable, replayable repository artifacts rather than transient tensors.  
Authoritative code paths: `tools/build_global_track_bank_v9.py`, `tools/extract_track_dino_features_v9.py`, `tests/test_global_track_bank_v9.py`, `tests/test_track_dino_feature_v9.py`.  
Exit intent: later semantic steps can be replayed from bank artifacts without rerunning the front-end basis generator.

### G4 — Prototype bank and class-level text map closure
Goal: prove seen-class visual prototypes and the constrained class-level text map can be initialized, serialized, and reloaded from bank-backed inputs.  
Authoritative code paths: `tools/init_prototype_bank_v9.py`, `tools/train_text_map_v9.py`, `tests/test_prototype_bank_v9.py`, `tests/test_text_map_v9.py`.  
Exit intent: prototype and text-map states exist with manifests and minimal retrieval/alignment evidence.

### G5 — Coverage-aware open-world attribution closure
Goal: prove the bounded `Y'(v) + bg + unk` attribution policy operates on the bank-backed semantic carriers and emits the hidden-positive diagnostics required by the outline.  
Authoritative code paths: `tools/train_openworld_core_v9.py`, `tests/test_openworld_core_v9.py`, `tests/test_g5_bounded_policy_v1.py`, `tests/test_ws_metrics_reporting_v1.py`, `tests/test_stage_d_ws_metrics_adapter_v1.py`.  
Exit intent: attribution is operational, explicit unknown-foreground handling exists, and ws-metric reporting is durable.

### G6 — Bag-free inference and evaluation closure
Goal: prove test-time inference runs without the observed label bag and produces the bounded evaluation bundle for the benchmark-scoped open-vocabulary claim.  
Authoritative code paths: `tools/run_bagfree_eval_v9.py`, `tests/test_bagfree_inference_v9.py`.  
Exit intent: the bag-free path is explicitly proven in artifacts, not merely assumed.

### G7 — Canonical replay, evidence freeze, and terminal closure
Goal: perform the canonical replay on the intended commit, freeze the final evidence bundle, and switch the mainline into bounded terminal revalidation mode.  
Authoritative code paths: `tools/remote_verify_wsovvis.sh`, `tools/run_mainline_loop.py`, `tools/stage_d_reporting_snapshot.py`, `tests/test_stage_d11_canonical_replay_v1.py`, `tests/test_stage_d_reporting_snapshot_v1.py`, `tests/test_stage_d13_ci_quick_pipeline_v1.py`.  
Exit intent: the accepted mainline is frozen with evidence-backed canonical PASS and no further algorithm development is opened by default.

Each gate must define both:
- its acceptance contract in `METRICS_ACCEPTANCE.md`, and
- its minimum evidence pack in `EVIDENCE_REQUIREMENTS.md`.

## 3. Default-off research branches
- Threshold-scan or low-confidence teacher-surface branches may open only after `G2` documents a raw-threshold failure against the S1 non-inferiority evidence bundle.
- Stronger semantic defenses such as duplicate-aware prototype refresh or uncertain-foreground filtering may open only after `G5` shows a specific attribution failure they directly target.
- Any unrestricted text, prompt-ensemble, or cross-dataset expansion branch stays outside the mainline unless the user explicitly opens new scope after `G7`.

## 4. Exploration policy
Exploration is allowed only when:
- the current gate explicitly requires it, and
- `STATUS.md` records the uncertainty, the smallest proposed experiment, and the fallback path.

If a result is inconclusive, prefer narrowing the current design rather than opening another branch.
