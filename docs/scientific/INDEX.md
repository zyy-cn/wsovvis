# WS-OVVIS Scientific Validation Overlay (v2 / v10-refined)

This directory is the **scientific validation overlay** for the repository.
It sits on top of the engineering control plane under `docs/mainline/*`.

## Relationship to engineering gates
- `docs/mainline/*` governs engineering readiness, canonical execution, artifact integrity, and bounded loop control.
- `docs/scientific/*` governs scientific question definition, mandatory comparators, metric hierarchies, evidence packs, formal-vs-diagnostic mode semantics, recovery subgates, pass rules, and human scientific sign-off.
- Engineering PASS is necessary but not sufficient for Scientific PASS.

## Source-of-truth note
The refined overlay is derived from:
- `docs/outline/WS_OVVIS_outline_v10_gate_refined_for_codex.tex`
- `大纲增量修改总结-V10.md`

When the two disagree during migration, `V10_RECONCILIATION_MEMO.md` records the authoritative reconciliation choices for the scientific overlay docs.

## Read order when a scientific gate is active
1. `V10_RECONCILIATION_MEMO.md`
2. `P0_EXPERIMENTAL_CHARTER.md`
3. `STATUS.md`
4. the active scientific gate spec
5. `reports/README.md`

## Scientific overlay inventory
Formal progression gates:
- `P0_EXPERIMENTAL_CHARTER.md`
- `S1_BASIS_SUPERIORITY.md`  (content migrated to Basis Utility while retaining path stability)
- `S2_GLOBAL_TRACK_VALUE.md`
- `S3A_SEMANTIC_CARRIER_VALIDITY.md`
- `S3B_OBJECTNESS_VALIDITY.md`
- `S4_PROTOTYPE_TEXT_BRIDGE_TRANSFER.md`
- `S5A_ATTRIBUTION_DECOMPOSITION.md`
- `S5B_HIDDEN_POSITIVE_RECOVERY.md`
- `S5C_MISSINGNESS_ROBUSTNESS.md`
- `S6_BAGFREE_OVVIS_FINAL.md`

Compatibility / legacy wrappers:
- `S3_SEMANTIC_CARRIER_VALIDITY.md`
- `S5_HIDDEN_POSITIVE_RECOVERY.md`

## Progression semantics
- Formal scientific sign-off remains ordered.
- Diagnostic-only pilots may run under explicit bounded gate specs without activating later formal gates.
- A formal scientific FAIL blocks later formal progression.
- A formal scientific FAIL does not automatically block bounded diagnosis or recovery planning.
- Recovery subgates are legal only when explicitly declared by the active charter or human-approved migration plan.

## Stop rules
- Do not activate `S1` until the refined `P0` is evidence-backed PASS.
- Do not activate later formal scientific gates unless the immediately previous formal scientific gate has passed under the refined charter.
- Do not treat diagnostic-only evidence as formal PASS evidence unless the active gate spec explicitly allows it.
