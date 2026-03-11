# P0 — Experimental Charter Gate

## Purpose
Freeze the scientific evaluation charter for all subsequent scientific gates (`S1–S6`) so that later results are comparable, bounded, and scientifically interpretable rather than merely engineering-complete.

## Relationship to engineering gates
- `G0–G7` remain the engineering gate layer.
- `P0 + S1–S6` form the scientific validation overlay.
- Engineering PASS is necessary but not sufficient for Scientific PASS.

## Scientific gate inventory
The scientific overlay consists of:
- `S1 — Basis Superiority`
- `S2 — Global Track Value`
- `S3 — Semantic Carrier Validity`
- `S4 — Prototype/Text Bridge Transfer`
- `S5 — Hidden Positive Recovery`
- `S6 — Bag-free OVVIS Final`

These names and this order may not be changed after `P0 PASS` without explicit charter revision approval.

## Dataset and split charter
- The primary dataset is fixed to **LV-VIS**.
- `train/val/test` follow the current project split.
- `S1–S5` scientific validation is restricted to `train+val`.
- `S6` may use `train+val` for bounded pilot / ablation / debug, but final Scientific PASS requires official final evaluation on the benchmark-scoped final split.
- Held-out-seen / pseudo-novel splits are allowed.
  - `S4`: mandatory
  - `S6`: supplementary / pre-final diagnostic only
  - `S3`: optional supplementary sanity check
  - `S1 / S2 / S5`: not part of the primary validation surface by default

## Weak-supervision protocol charter
- `Y'(v)` is the only active weak-supervision object.
- Allowed protocol family:
  - `Uniform Missing`
  - `Long-tail Missing`
- The main-table operating point is fixed at `p = 0.4`.
- Other `p` values are supplementary / ablation only unless explicit charter revision is approved.

## Comparator charter
Each scientific gate must have mandatory comparators:
- `S1`: raw pseudo tube vs refined basis
- `S2`: local tracklets vs global tracks
- `S3`: semantic-carrier necessity comparator
- `S4`: seen alignment vs held-out transfer
- `S5`: aligned closed-world comparator vs bounded core open-world attribution
- `S6`: bag-constrained vs bag-free full-vocabulary inference

Codex may not silently replace comparators.

## Metric charter
Each scientific gate must explicitly define:
- primary metrics
- diagnostic metrics
- supporting statistics
- mandatory worked example
- mandatory visualization

## Evidence charter
Each scientific gate must produce at least:
- `phase_gate_latest`
- `evidence_latest`
- `acceptance_latest`
- metrics summary
- comparator summary
- at least one worked example
- at least one core visualization
- canonical run record when required

## Pass-rule charter
Before formal activation, each scientific gate must freeze:
- `PASS`
- `INCONCLUSIVE`
- `FAIL`
- `BLOCKED`
semantics, including:
- what counts as improvement
- what remains merely inconclusive
- what negative result is sufficient for `FAIL`
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
- Each scientific gate begins with a bounded pilot.
- No expensive training may begin before explicit human confirmation.
- Comparators should use matched or closely matched budget whenever possible.
- Early scientific gates may begin with single-seed runs.
- Stronger repetition may be required later, but must be declared in the gate spec.
- Codex may not reduce comparator fairness in order to make a cheaper gate appear complete.

## Canonical-validation charter
- Scientific PASS should rely on canonical evidence by default.
- Local runs are informative/debug/pilot only unless explicitly elevated by the charter.
- Main evidence used for formal Scientific PASS must come from matched canonical runs.

## Sign-off charter
- Codex implements, runs, evaluates, and reports.
- The assistant may summarize scientific judgment language.
- Final Scientific PASS is human-signed and may not be auto-declared by Codex beyond the gate contract.

## Change-control charter
After `P0 PASS`, the charter may not be changed ad hoc.
Any revision must state:
- why
- affected gates
- backward impact
- whether previous results remain comparable

## P0 PASS condition
`P0` may PASS only if all required frozen fields above are fully specified.

## P0 INCONCLUSIVE condition
`P0` is INCONCLUSIVE if any comparator, metric, pass rule, split/protocol rule, shortcut restriction, budget rule, or canonical rule remains unresolved.

## P0 BLOCKED condition
`P0` is BLOCKED if required benchmark/split/baseline information is unavailable or the environment cannot support the frozen scientific protocol.

## Stop rule
Do not activate `S1` until `P0` is evidence-backed PASS.
