# WS-OVVIS Evidence Requirements

This file defines the minimum durable evidence required before a gate may be marked `PASS`.

## 1. Universal evidence pack requirements
Every active gate should provide, at minimum:
- quantitative indicators that directly support the gate objective,
- the exact commands run and the artifact paths produced,
- one minimal worked example with inputs, intermediate steps, and outputs,
- a concise explanation of why the evidence is sufficient,
- archive copies under `docs/mainline/reports/archive/` for durable recovery.

## 2. Standard durable outputs
Unless a gate explicitly says otherwise, produce:
- `docs/mainline/reports/evidence_latest.txt`
- `docs/mainline/reports/worked_example_verification_latest.md`
- `docs/mainline/reports/worked_example_verification_latest.json`
- timestamped archive copies for each of the above

## 3. Gate-specific evidence requirements
### G0 evidence
- wrapper/checker presence and usage captured in `evidence_latest.txt`
- successful `python tools/run_mainline_loop.py --dry-run` output path
- one worked example showing how the first supervisor prompt is generated
- explicit record of inferred canonical environment facts and any unresolved blockers

### G1 evidence
- protocol output JSON, manifest JSON, and command line used
- one worked example from a small input JSON showing `label_set_full_ids` and `label_set_observed_ids`
- at least one relevant test result reference

### G2 evidence
- basis export artifact schema with non-empty query trajectories
- S1 comparator summary at original thresholds
- command log, config path, and output directory layout
- one worked example tracing a predicted trajectory back to source fields

### G3 evidence
- Query-Trajectory Bank manifest and sample entry
- Semantic Carrier Bank manifest and sample carrier entry
- replay demonstration proving downstream consumption without rerunning the basis generator

### G4 evidence
- prototype-bank manifest + arrays path
- text-map state + mapped prototype manifest
- one worked example showing label text, prototype row, and mapped prototype lookup

### G5 evidence
- attribution summary including the bounded label space (`Y'(v)+bg+unk`)
- ws-metrics summary containing `SCR`, missing-rate curve, `AURC`, and `HPR` / `UAR` when applicable
- worked example showing one clip’s observed labels, candidate labels, and unknown-attribution behavior

### G6 evidence
- bag-free evaluation manifest with explicit proof that the observed label bag was not used at test time
- main result table and qualitative outputs or their recorded paths
- worked example for one selected video showing the candidate label set and final predictions

### G7 evidence
- canonical remote command, intended commit, remote `HEAD`, and whether they match
- final acceptance/evidence reports archived
- `mainline_terminal_summary.txt` updated with the bounded revalidation rule

## 4. Judgment rule
- If acceptance is satisfied but the required evidence pack is incomplete, return `INCONCLUSIVE`, not `PASS`.
- If the evidence contradicts the gate claim, return `FAIL`.
- If canonical validation is required and unavailable, return `BLOCKED` or `INCONCLUSIVE` according to `ENVIRONMENT_AND_VALIDATION.md`.
