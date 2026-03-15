# WS-OVVIS Metrics and Acceptance

This file defines gate-level acceptance for the mainline.

## 1. Mainline metric priority
Primary metrics:
- raw-basis structure metrics at original thresholds: `mean_best_iou`, `recall@0.5`, `fragmentation_per_gt_instance`, `predicted_tracks_total`
- bank durability / replay consistency for query trajectories and semantic carriers
- prototype/text-map alignment indicators sufficient to show constrained class-level mapping is operational
- weak-supervision metrics: `SCR`, missing-rate curve, `AURC`, and when applicable `HPR` / `UAR`
- bag-free evaluation metrics and explicit bag-free path proof for the benchmark-scoped OV inference claim

Secondary / supporting metrics:
- schema completeness and manifest consistency
- non-empty export counts, deterministic IDs, and qualitative worked-example integrity
- remote commit-consistency and wrapper/bootstrap health for canonical PASS

## 2. Gate acceptance policy
### G0
Pass only when the adapted control plane is deployed, `python tools/run_mainline_loop.py --dry-run` succeeds, wrapper/checker presence is recorded, and the canonical environment facts needed for future replay are explicitly captured. Missing remote replay evidence keeps `G0` at `INCONCLUSIVE` unless the gate was scoped as local-only bootstrap documentation.

### G1
Pass only when the weak-label protocol builder produces reproducible output + manifest artifacts with parameters, counts, and one worked example, and `tests/test_build_wsovvis_labelset_protocol.py` is green in the chosen validation mode.

### G2
Pass only when the DINOv2 basis path is integrated, non-empty query-trajectory export is demonstrated, and the S1 structure-only comparator bundle exists at original thresholds. A raw path that exports but lacks the comparator evidence is `INCONCLUSIVE`, not `PASS`.

### G3
Pass only when Query-Trajectory Bank and Semantic Carrier Bank artifacts are materialized with stable IDs, manifests, replayable paths, and an explicit demonstration that later semantic steps can consume them without rerunning the basis generator.

### G4
Pass only when prototype-bank and text-map states are serialized with manifests, reload cleanly, and the constrained class-level mapping shows reviewable alignment evidence from bank-backed inputs.

### G5
Pass only when coverage-aware attribution runs on the documented input contract, emits bounded unknown-foreground / background behavior, and produces the ws-metric bundle needed to judge hidden-positive handling.

### G6
Pass only when bag-free inference is explicitly proven by artifacts showing the observed label bag is not used at test time, and the evaluation bundle contains the expected main result and weak-supervision diagnostics.

### G7
Pass only when canonical remote replay is executed on the intended commit, final reports are frozen, `STATUS.md` marks terminal mode active, and the terminal summary records the bounded revalidation rule.

## 3. Relationship to evidence requirements
A gate may pass only if:
1. its acceptance contract is satisfied, and
2. its required evidence pack from `EVIDENCE_REQUIREMENTS.md` is complete and reviewable.

## 4. PASS / FAIL / INCONCLUSIVE semantics
- `PASS`: gate contract is satisfied, evidence is complete, and the next gate may become active.
- `FAIL`: gate is not satisfied or evidence directly contradicts the gate claim; the fallback path must be used.
- `INCONCLUSIVE`: evidence is insufficient or incomplete; do not widen scope, first close the evidence gap.
- `BLOCKED`: environment or validation preconditions prevent canonical evaluation.
