# V10 Reconciliation Memo for Scientific Overlay v2

## Purpose
This memo reconciles the scientific overlay implied by:
- `docs/outline/WS_OVVIS_outline_v10_gate_refined_for_codex.tex`
- `大纲增量修改总结-V10.md`

The goal is to produce a single deployable scientific overlay for Codex execution.

## Key reconciliation decisions

### 1. Summary intent takes precedence when the v10 tex has not fully caught up
The summary document describes the intended refined overlay more explicitly than the current v10 tex in several places. During migration, the summary's gate-refinement intent is authoritative unless it directly contradicts the high-level method design.

### 2. S1 is migrated from strict Basis Superiority to Basis Utility
The file path `S1_BASIS_SUPERIORITY.md` is retained for compatibility, but the active content is the refined `S1 — Basis Utility Gate`.

### 3. S3 is split into S3a and S3b
- `S3A_SEMANTIC_CARRIER_VALIDITY.md`
- `S3B_OBJECTNESS_VALIDITY.md`
The legacy `S3_SEMANTIC_CARRIER_VALIDITY.md` remains only as a compatibility wrapper.

### 4. S5 is split into S5a / S5b / S5c
- `S5A_ATTRIBUTION_DECOMPOSITION.md`
- `S5B_HIDDEN_POSITIVE_RECOVERY.md`
- `S5C_MISSINGNESS_ROBUSTNESS.md`
The legacy `S5_HIDDEN_POSITIVE_RECOVERY.md` remains only as a compatibility wrapper.

### 5. Formal progression vs diagnostic-only probing is explicit
The refined overlay distinguishes:
- formal progression evidence
- diagnostic-only evidence
- recovery-gate evidence
A formal FAIL blocks later formal gates, but not necessarily bounded diagnosis or recovery planning.

### 6. Anti-dumping / unknown-legitimacy requirements belong in S5b
The refined overlay treats anti-dumping as a first-class scientific requirement, not as an afterthought.

### 7. Held-out transfer must be primary in S4
Seen alignment alone is not sufficient for `S4 PASS`.

## Migration handling of old results
- Historical `S1 FAIL` / `S1R` materials remain valid historical evidence.
- They do not automatically determine the outcome of the refined `S1`.
- Formal scientific status must be re-established under the refined charter after migration.
