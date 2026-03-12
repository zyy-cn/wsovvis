# S1 — Basis Utility Gate

> Compatibility note: this file retains the legacy `S1_BASIS_SUPERIORITY.md` path for overlay stability, but the active gate content is the refined **Basis Utility Gate**.

## Scientific question
Does the refined instance basis provide scientifically meaningful utility over raw VideoCutLER pseudo tubes as a video-native basis, either by direct structural superiority or by non-inferior structure together with improved downstream consumability under the bounded refinement setting?

## Why this gate changed
The earlier strict superiority-only wording proved too brittle: it could reject a refined basis that is not uniformly stronger on every direct structure metric yet is materially more consumable by later structure-preserving stages. The refined gate therefore distinguishes:
- direct basis superiority
- non-inferior basis utility with downstream-consumption value

## Scope
S1 remains structure-first and may not use later semantic or attribution modules as proof shortcuts.

Allowed:
- raw VideoCutLER pseudo tubes
- refined local-tracklet basis
- bounded structure-side diagnostics
- narrowly declared downstream-consumability evidence that does not require semantic correctness claims

Forbidden:
- using later semantic success as the sole proof of S1
- prototype/text-map, attribution, or bag-free inference as replacement evidence
- undeclared default-off modules

## Mandatory comparators
- C1: raw pseudo tube
- C2: refined local-tracklet basis

Supplementary only:
- refined global-track or other later structure stages may appear only as utility-supporting evidence if the gate spec explicitly references them; they may not replace the primary C1/C2 comparator.

## Data charter
- train for producing the refined basis
- val for formal scientific comparison
- test forbidden
- held-out-seen / pseudo-novel not part of primary S1 validation

## Evidence routes
### Route A — Direct superiority
A refined basis may PASS via direct superiority if it improves the primary structure criteria without unacceptable tradeoffs.

### Route B — Utility without direct dominance
A refined basis may PASS via utility if:
- it is not clearly worse in the aggregate structure sense under the frozen pass rule,
- and it yields materially better bounded downstream consumability / stitchability / structural usability evidence,
- and the utility claim is supported by explicit comparator evidence rather than narrative argument.

## Primary metrics
At minimum:
- `mean_best_iou`
- `recall_at_0.5`
- `fragmentation_per_gt_instance`

Optional supporting metric:
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
- any declared utility-side diagnostic needed to support Route B

## Required qualitative evidence
- one paired worked example: raw pseudo tube vs refined basis vs GT
- one failure-case example
- if Route B is claimed, one explicit utility-side worked example

## PASS rule
S1 may PASS only if either Route A or Route B is satisfied under the frozen canonical protocol.

### Route A PASS
- refined basis improves at least one primary structure metric over raw
- refined basis does not catastrophically degrade the others
- fragmentation is not clearly worse in a way that invalidates utility
- worked examples support the improvement claim
- at least one fresh canonical training-based comparator result exists

### Route B PASS
- refined basis is not clearly worse in aggregate structural usefulness
- direct superiority may be mixed or absent
- bounded downstream-consumability evidence shows a concrete utility gain that the raw basis does not provide
- utility evidence is explicit, comparator-based, and human-reviewable
- at least one fresh canonical training-based comparator result exists

## INCONCLUSIVE
S1 is INCONCLUSIVE if:
- comparator fairness remains unresolved
- direct structure metrics are mixed and utility evidence is incomplete
- only pilot/historical evidence exists without a fresh canonical retraining result
- the utility claim is plausible but not yet sufficiently evidenced

## FAIL
S1 is FAIL if:
- the refined basis is clearly worse on the primary structure surface and no credible bounded utility path rescues it
- fragmentation inflation or over-segmentation invalidates the claimed basis utility
- worked examples contradict the utility claim
- the apparent gain depends on unfair comparator conditions

## BLOCKED
S1 is BLOCKED if:
- a unified canonical comparator evaluation surface cannot be constructed
- the required fresh canonical training cannot be executed under approved budget
- GT alignment/evaluation tooling is missing

## Required outputs
- `phase_gate_latest`
- `evidence_latest`
- `acceptance_latest`
- comparator metrics table
- paired worked example
- failure-case example
- canonical training/evaluation record
- explicit route declaration (A or B)

## Stop rule
Do not activate refined `S2` until refined `S1` is evidence-backed PASS.
