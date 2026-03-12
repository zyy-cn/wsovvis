# WS-OVVIS Scientific Reports (Refined v2)

This directory stores the durable outputs required by the scientific validation overlay.

## Report classes
### 1. Formal gate reports
These can support formal PASS/FAIL/INCONCLUSIVE/BLOCKED judgments under the active refined charter.

### 2. Diagnostic-only reports
These may inform recovery or charter-revision decisions, but do not substitute for formal PASS evidence unless the active gate explicitly allows it.

### 3. Recovery-gate reports
These support the decision of whether a failed formal gate may be reopened under a bounded recovery plan.

## Core expected files
- `phase_gate_latest.txt`
- `evidence_latest.txt`
- `acceptance_latest.txt`
- `comparator_latest.txt`
- `signoff_latest.txt`

## Policy
- every scientific loop should also write timestamped copies under `archive/`
- scientific reports and engineering reports must remain separate
- engineering readiness does not substitute for scientific acceptance
- diagnostic-only evidence must be labeled as such
- historical evidence from a prior overlay version must not be silently promoted to refined formal PASS evidence
