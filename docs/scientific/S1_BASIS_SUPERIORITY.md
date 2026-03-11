# S1 — Basis Superiority Gate

## Scientific question
Does the SeqFormer-refined instance basis significantly improve over raw VideoCutLER pseudo tubes as a class-agnostic video-structure basis, without relying on downstream semantic modules?

## Purpose
`S1` proves that the learned/refined instance basis is scientifically better than the raw pseudo-tube seed, rather than merely exportable or reusable by later engineering gates.

## Scope
`S1` is structure-only.

### Allowed
- raw VideoCutLER pseudo tubes
- SeqFormer-refined local-tracklet basis
- canonical train/val split
- class-agnostic GT-based structure evaluation

### Forbidden
- global-track stitching as the primary comparator
- semantic cache / `z_tau` / `o_tau`
- prototype or text-map modules
- attribution or open-world logic
- bag-free inference
- undeclared default-off scientific shortcuts

## Mandatory comparators
- `C1`: raw pseudo tube
- `C2`: SeqFormer-refined local-tracklet basis

### Supplementary comparator
- raw pseudo tube vs refined global tracks may be reported only as supplementary/appendix evidence, not as the primary `S1` proof surface

## Data charter
- train is used to produce the refined basis
- val is used for formal scientific comparator evaluation
- test is forbidden
- held-out-seen / pseudo-novel are not part of the primary `S1` validation surface

## Training charter
- historical SeqFormer runs may be used for pilot comparator setup only
- formal `S1 PASS` requires at least one fresh canonical SeqFormer training run under frozen configuration
- no expensive training may begin before explicit human approval

## Primary metrics
`S1` must include:
1. structure-quality primary metrics:
   - `mean_best_iou`
   - `recall_at_0.5`
2. fragmentation-quality primary metric:
   - `fragmentation_per_gt_instance`

Optional additional primary/supporting metric:
- `recall_at_0.7`

## Diagnostic metrics
At minimum:
- `short_track_ratio`
- `broken_track_ratio`
- `average_track_duration`
- `temporal_inconsistency`
- `unmatched_gt_ratio`
- `over_segmentation_tendency`
- `under_coverage_tendency`

## Required qualitative evidence
- one paired worked example showing raw pseudo tube vs refined basis vs GT
- one failure-case example

## PASS rule
`S1` may PASS only if:
- the comparator is run under the frozen canonical protocol
- the refined basis improves at least one structure-quality primary metric over raw pseudo tubes
- the refined basis also improves the fragmentation-quality primary metric
- the improvement is not achieved by catastrophic degradation of the other primary metric(s)
- the paired worked example supports the claimed improvement
- the failure-case analysis does not contradict the claimed basis improvement
- the result includes at least one fresh canonical training-based comparator run

## INCONCLUSIVE
`S1` is INCONCLUSIVE if:
- only pilot/historical evidence exists without a fresh canonical retraining result
- one primary metric improves but the other remains ambiguous
- comparator fairness remains unresolved
- qualitative evidence is incomplete
- evaluator stability is not yet sufficient

## FAIL
`S1` is FAIL if:
- the refined basis does not outperform raw pseudo tubes on the primary structure criterion
- fragmentation does not improve or clearly worsens
- the apparent gain depends on unfair comparator conditions
- worked examples show that the gain is not a real basis improvement

## BLOCKED
`S1` is BLOCKED if:
- a unified canonical comparator evaluation surface cannot be constructed
- fresh canonical training cannot be executed under approved budget
- required GT alignment/evaluation tooling is missing

## Required outputs
- `phase_gate_latest`
- `evidence_latest`
- `acceptance_latest`
- comparator metrics table
- paired worked example
- failure-case example
- canonical training/evaluation record

## Stop rule
Do not activate `S2` until `S1` is evidence-backed PASS.
