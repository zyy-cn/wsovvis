# WS-OVVIS Evidence Requirements

This file defines the minimum durable evidence required before a gate may be marked `PASS`.

A gate must not be marked `PASS` unless both of the following are true:
1. the acceptance contract in `METRICS_ACCEPTANCE.md` is satisfied, and
2. the gate's evidence pack is complete, reviewable, and consistent with the code, logs, and artifacts.

If the acceptance contract is satisfied but the evidence pack is incomplete, the correct judgment is `INCONCLUSIVE`.

## 1. Universal evidence pack requirements
Every active gate should provide, at minimum:
- quantitative indicators that directly support the gate objective,
- the exact commands run and the artifact paths produced,
- one minimal worked example with inputs, intermediate steps, and outputs,
- at least one human-reviewable artifact or visualization when applicable,
- a concise explanation of why the evidence is sufficient,
- archive copies under `docs/mainline/reports/archive/` for durable recovery.

## 2. Standard durable outputs
Unless a gate explicitly says otherwise, produce:
- `docs/mainline/reports/evidence_latest.txt`
- `docs/mainline/reports/worked_example_verification_latest.md`
- `docs/mainline/reports/worked_example_verification_latest.json`
- timestamped archive copies for each of the above

## 3. Gate-specific minimum evidence expectations

### G0
Minimum evidence:
- dry-run or equivalent control-plane readout proving `docs/mainline/*` is the active interpretation source
- a worked example showing how the active gate is resolved from the docs
- authority-chain summary comparing legacy / draft paths versus the active `docs/mainline/*` path
- canonical environment fact summary, including wrapper and bootstrap-preflight availability

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
- canonical `mAP` / `HPR` / `UAR` / robustness results
- at least one bag-free inference worked example from global tracks to final predictions
- a small set of qualitative prediction visualizations
- canonical remote validation record with intended commit and remote `HEAD`

## 4. Judgment rule
- If acceptance is satisfied but the required evidence pack is incomplete, return `INCONCLUSIVE`, not `PASS`.
- If the evidence contradicts the gate claim, return `FAIL`.
- If canonical validation is required and unavailable, return `BLOCKED` or `INCONCLUSIVE` according to `ENVIRONMENT_AND_VALIDATION.md`.
