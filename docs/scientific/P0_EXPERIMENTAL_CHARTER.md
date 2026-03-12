# P0 — Experimental Charter Gate (Refined v2)

## Purpose
Freeze the scientific evaluation charter for all subsequent scientific gates so that later results are comparable, bounded, scientifically interpretable, and not reducible to engineering-complete execution only.

## Relationship to engineering gates
- `G0–G7` remain the engineering gate layer.
- `P0 + S1/S2/S3a/S3b/S4/S5a/S5b/S5c/S6` form the scientific validation overlay.
- Engineering PASS is necessary but not sufficient for Scientific PASS.

## Formal scientific inventory
Formal progression gates are:
- `S1 — Basis Utility`
- `S2 — Global Track Value`
- `S3a — Semantic Carrier Validity`
- `S3b — Objectness Validity`
- `S4 — Prototype/Text Bridge Transfer`
- `S5a — Attribution Decomposition`
- `S5b — Hidden Positive Recovery`
- `S5c — Missingness Robustness`
- `S6 — Bag-free OVVIS Final`

Diagnostic-only and recovery subgates may exist, but must be explicitly declared and may not silently substitute for formal progression gates.

## Dataset and split charter
- Primary dataset: LV-VIS
- Train/val/test follow the project split
- `S1–S5c` formal validation is restricted to train+val unless a gate spec explicitly says otherwise
- `S6` may use train+val for bounded pilot/ablation/debug, but final formal PASS requires benchmark-scoped final evaluation
- held-out-seen / pseudo-novel are allowed under:
  - `S4`: mandatory
  - `S6`: supplementary / pre-final diagnostic only
  - `S3a/S3b`: optional supplementary sanity checks
  - `S1/S2/S5a/S5b/S5c`: not part of the primary validation surface by default

## Weak-supervision charter
- `Y'(v)` is the only active weak-supervision object
- allowed protocol family:
  - `Uniform Missing`
  - `Long-tail Missing`
- main-table operating point: `p = 0.4`
- other `p` values are supplementary / robustness / ablation only unless explicitly elevated by charter revision

## Comparator charter
Each formal scientific gate must have mandatory comparators frozen before activation:
- `S1`: raw pseudo tube vs utility-oriented refined basis comparator(s)
- `S2`: local tracklets vs global tracks
- `S3a`: semantic carrier necessity comparator
- `S3b`: objectness-validity comparator
- `S4`: seen alignment vs held-out transfer
- `S5a`: decomposition comparator
- `S5b`: aligned closed-world comparator vs bounded open-world core
- `S5c`: missingness / protocol robustness comparator set
- `S6`: bag-constrained vs bag-free full-vocabulary inference

Diagnostic-only probes may define temporary comparators, but they may not be used as formal PASS evidence unless the active gate spec explicitly permits that.

## Metric charter
Each formal scientific gate must explicitly define:
- primary metrics
- diagnostic metrics
- supporting statistics
- mandatory worked example(s)
- mandatory visualization(s)

## Evidence charter
Each formal scientific gate must produce at least:
- `phase_gate_latest`
- `evidence_latest`
- `acceptance_latest`
- metrics summary
- comparator summary
- at least one worked example
- at least one core visualization
- canonical run record when required

Diagnostic-only and recovery gates must also write reports, but those reports must be marked as non-formal unless explicitly promoted by charter.

## Pass-rule charter
Before formal activation, each scientific gate must freeze:
- `PASS`
- `INCONCLUSIVE`
- `FAIL`
- `BLOCKED`
semantics, including:
- what counts as improvement
- what counts as non-inferior utility
- what remains diagnostic-only
- what negative result is sufficient for FAIL
- what conditions are purely blocked by environment/budget/data

## Default-off / forbidden-shortcut charter
The following remain forbidden unless explicitly authorized by a later scientific gate spec:
- retrieval
- warm-up BCE
- temporal consistency
- unknown fallback
- one-round refinement
- multi-round continuation
- prototype EM
- momentum refresh
- prompt/synonym expansion
- any undeclared shortcut that changes comparator fairness or scientific interpretation

## Training-budget charter
- Each scientific gate begins with a bounded pilot unless formal activation already requires a completed full run.
- No expensive training may begin before explicit human confirmation.
- Comparators should use matched or closely matched budget whenever possible.
- Early scientific gates may begin with single-seed runs.
- Stronger repetition may be required later, but must be declared in the gate spec.
- Codex may not reduce comparator fairness in order to make a cheaper gate appear complete.

## Canonical-validation charter
- Formal Scientific PASS should rely on canonical evidence by default.
- Local runs are informative/debug/pilot only unless explicitly elevated by the charter.
- Main evidence used for formal Scientific PASS must come from matched canonical runs.
- Diagnostic-only probes may inform recovery decisions but do not substitute for formal PASS evidence by default.

## Sign-off charter
- Codex implements, runs, evaluates, and reports.
- The assistant may summarize scientific judgment language.
- Final formal Scientific PASS is human-signed and may not be auto-declared by Codex beyond the gate contract.

## Change-control charter
After refined `P0 PASS`, the charter may not be changed ad hoc.
Any revision must state:
- why
- affected gates
- backward impact
- whether previous results remain comparable

## PASS condition
`P0` may PASS only if all required frozen fields above are fully specified under the refined inventory.

## INCONCLUSIVE condition
`P0` is INCONCLUSIVE if any comparator, metric, pass rule, split/protocol rule, shortcut restriction, budget rule, or canonical rule remains unresolved.

## BLOCKED condition
`P0` is BLOCKED if required benchmark/split/baseline information is unavailable or the environment cannot support the frozen scientific protocol.

## Stop rule
Do not activate refined `S1` until refined `P0` is evidence-backed PASS.
