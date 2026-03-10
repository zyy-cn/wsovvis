# WSOVVIS V9 Evidence Requirements

A gate must not be marked `PASS` unless both of the following are true:
1. the acceptance contract in `METRICS_ACCEPTANCE.md` is satisfied; and
2. the gate's evidence pack is complete, reviewable, and consistent with the code, logs, and artifacts.

If the acceptance contract is satisfied but the evidence pack is incomplete, the correct judgment is `INCONCLUSIVE`.

## 1. Evidence-pack structure required for every gate
Each gate evidence pack must contain the following six sections.

### E1. Goal attainment summary
Required content:
- gate name and current judgment
- what changed in this bounded step
- which acceptance items were targeted
- why the current evidence is or is not sufficient for `PASS`

### E2. Quantitative evidence
Required content:
- the primary quantitative metrics for the gate
- dataset / split / sample scope
- comparison baseline or before/after numbers when applicable
- clear indication of which numbers are informative only and which are canonical

### E3. Visualization evidence
Required content:
- one or more figures that directly support the gate claim
- figure captions or short text telling the reviewer what to inspect
- file paths for each figure or rendered table

### E4. Worked example
Required content:
- at least one minimal worked example for the gate
- the chosen clip / window / track / class / batch identifier
- step-by-step intermediate values that can be traced back to code paths and artifacts
- a short explanation of why this example is representative enough to support the gate review

### E5. Step logs and input/output records
Required content:
- exact command or entrypoint used
- config / checkpoint / commit references
- input artifact paths
- output artifact paths
- key structured logs and intermediate dumps needed to reproduce the worked example

### E6. Gate judgment note
Required content:
- the final `PASS` / `FAIL` / `INCONCLUSIVE` recommendation
- the smallest unresolved gap, if any
- why remaining defects do or do not block the current gate

## 2. Gate-specific minimum evidence expectations

### G0
Minimum evidence:
- dry-run or equivalent control-plane readout proving the v9 control plane is the active interpretation source
- a worked example showing how the active gate is resolved from the current docs
- authority-chain summary comparing legacy vs v9 paths

### G1
Minimum evidence:
- actual protocol statistics for `Y'(v)` generation
- at least one clip-level full-vs-observed label worked example
- contract parse/readback proof for the declared artifact forms

### G2
Minimum evidence:
- local-tracklet counts, duration statistics, and filtering-retention statistics
- at least one pseudo-tube to local-tracklet worked example
- visualization of filtering or local-tracklet outputs

### G3
Minimum evidence:
- stitching match statistics and fragmentation-style diagnostics
- at least one local-to-global stitching worked example with score matrix and merge result
- a visualization of global-track coverage on a clip

### G4
Minimum evidence:
- semantic-cache coverage and objectness-distribution statistics
- at least one per-track DINO extraction and aggregation worked example
- crop/pooling visualizations or equivalent feature provenance figures

### G5
Minimum evidence:
- prototype-bank coverage statistics and text-map alignment metrics
- at least one class-level prototype/text-map worked example
- visualization of prototype distances or mapped-text alignment

### G6
Minimum evidence:
- known/background/unknown allocation metrics
- hidden-positive metrics (`HPR`, `UAR`) or an explicit evidence-gap statement
- at least one full clip-level attribution worked example with cost matrix and assignment summary
- comparison against the aligned closed-world baseline

### G7
Minimum evidence:
- canonical AP / HPR / UAR / robustness results
- at least one bag-free inference worked example from global tracks to final predictions
- a small set of qualitative prediction visualizations
- canonical remote validation record with intended commit and remote `HEAD`

## 3. Report locations
When the v9 control plane is active, each gate review should write:
- `docs/mainline_v9/reports/phase_gate_latest.txt`
- `docs/mainline_v9/reports/acceptance_latest.txt`
- `docs/mainline_v9/reports/evidence_latest.txt`
- timestamped archives under `docs/mainline_v9/reports/archive/`

The evidence report must point to figure paths, worked-example logs, and artifact locations rather than only restating conclusions.

## 4. Reviewability rule
Evidence must be reviewable by a human without reverse-engineering the entire repository.
A report that only says a command passed is not sufficient.
