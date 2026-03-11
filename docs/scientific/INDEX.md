# WS-OVVIS Scientific Validation Overlay

This directory is the **scientific validation overlay** for the repository.
It sits on top of the engineering control plane under `docs/mainline/*`.

## Relationship to engineering gates
- `docs/mainline/*` governs engineering readiness, canonical execution, artifact integrity, and bounded loop control.
- `docs/scientific/*` governs scientific question definition, mandatory comparators, metric hierarchies, evidence packs, pass rules, and human scientific sign-off.
- Engineering PASS is necessary but not sufficient for Scientific PASS.

## Read order when a scientific gate is active
1. `P0_EXPERIMENTAL_CHARTER.md`
2. `STATUS.md`
3. the active scientific gate spec (`S1...S6`)
4. `reports/README.md`

## Scientific gate inventory
- `P0_EXPERIMENTAL_CHARTER.md`
- `S1_BASIS_SUPERIORITY.md`
- `S2_GLOBAL_TRACK_VALUE.md`
- `S3_SEMANTIC_CARRIER_VALIDITY.md`
- `S4_PROTOTYPE_TEXT_BRIDGE_TRANSFER.md`
- `S5_HIDDEN_POSITIVE_RECOVERY.md`
- `S6_BAGFREE_OVVIS_FINAL.md`

## Current activation rule
Do not activate `S1` until `P0` is evidence-backed PASS.
Do not activate later scientific gates unless the immediately previous scientific gate is evidence-backed PASS.
