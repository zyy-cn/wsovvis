# Scientific Overlay v2 Migration Plan

## Goal
Switch the scientific control logic from the earlier strict overlay to the refined v10-based overlay without rewriting engineering history.

## Migration phases
1. Freeze old overlay progression and preserve historical reports.
2. Reconcile v10 tex and the summary intent via `V10_RECONCILIATION_MEMO.md`.
3. Rewrite the scientific overlay docs to the refined inventory and semantics.
4. Re-run `P0` under the refined charter.
5. Re-activate `S1` under the refined gate content only after refined `P0 PASS`.

## Migration-closeout rules
- The refined v10 overlay docs under `docs/scientific/*` are the only active scientific authority during migration closeout.
- Old strict-overlay `S1 FAIL` and `S1R` outputs remain historical evidence only and must stay archived.
- `docs/scientific/reports/*_latest` must point to the active refined gate packet; during migration closeout that gate is `P0`.
- `S1` activation is a separate bounded loop after `P0` closure; it is not part of the migration-closeout loop itself.

## Immediate rule
Do not continue old `S1R` after deploying this overlay package.
