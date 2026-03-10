# WSOVVIS V9 Mainline Implement Runbook

This file defines how Codex should work inside the repository once the v9 control plane becomes authoritative.
It preserves the current development mode: document-driven, phase-gated, bounded, and terminal-aware.

## 1. Non-negotiable operating rules
1. Read control docs before touching code.
2. Work only on the active gate.
3. Take the smallest valid next step.
4. Do not enable default-off modules unless the failure playbook explicitly authorizes that step.
5. Do not mark a gate `PASS` unless both the acceptance contract and the required evidence pack are complete.
6. Stop after one bounded loop and write status/report artifacts.

## 2. Required read order for a v9-controlled loop
1. `AGENTS.md`
2. `docs/mainline_v9/INDEX.md`
3. `docs/mainline_v9/EXECUTION_SOURCE.md`
4. `docs/mainline_v9/PLAN.md`
5. `docs/mainline_v9/IMPLEMENT.md`
6. `docs/mainline_v9/STATUS.md`
7. `docs/mainline_v9/METRICS_ACCEPTANCE.md`
8. `docs/mainline_v9/EVIDENCE_REQUIREMENTS.md`
9. `docs/mainline_v9/FAILURE_PLAYBOOK.md`
10. `docs/mainline_v9/ENVIRONMENT_AND_VALIDATION.md`
11. `docs/mainline_v9/CODEBASE_MAP.md`
12. `docs/mainline_v9/STAGEB_INTERFACE_CONTRACT.md`

## 3. Loop skeleton
For each bounded loop:
1. determine the active gate from `STATUS.md`
2. identify the blocking acceptance or evidence gap
3. identify the smallest valid step
4. touch only the files needed for that step
5. run the minimum informative validation
6. run canonical validation if the gate needs canonical evidence for a `PASS`
7. evaluate acceptance status
8. evaluate evidence status
9. update `STATUS.md`
10. write gate / acceptance / evidence reports
11. stop

## 4. Acceptance vs evidence evaluation
Each loop must separately determine:
- `Acceptance status`: `satisfied`, `unsatisfied`, or `unknown`
- `Evidence status`: `complete`, `incomplete`, or `not-reviewed`

Judgment mapping:
- `PASS`: acceptance satisfied + evidence complete + canonical requirements met when required
- `INCONCLUSIVE`: acceptance partly or fully satisfied but evidence incomplete, noisy, missing, or not reviewable
- `FAIL`: acceptance contradicted by the evidence
- `BLOCKED`: infrastructure, remote, or environment issues prevent trustworthy evaluation

## 5. Out-of-scope discipline
The following remain default-off unless the docs explicitly activate them:
- prototype EM / momentum refresh
- retrieval
- warm-up BCE
- temporal consistency
- unknown fallback
- one-round quality-aware refinement

Do not use these to rescue a gate before the bounded core path is understood.

## 6. Report outputs required per bounded loop
When v9 is authoritative, each bounded loop should update or create:
- `docs/mainline_v9/reports/phase_gate_latest.txt`
- `docs/mainline_v9/reports/acceptance_latest.txt`
- `docs/mainline_v9/reports/evidence_latest.txt`
- timestamped copies in `docs/mainline_v9/reports/archive/`

The evidence report must include or point to:
- quantitative metrics
- figure paths
- worked-example identifiers
- key input/output artifact paths
- exact commands and configs for the reviewed run

## 7. Entry-point discipline by gate
- G0: docs and supervisor/control-plane entrypoints only
- G1: protocol tooling and contract parsing only
- G2: pseudo-tube / class-agnostic basis / local-tracklet entrypoints only
- G3: global-track-bank entrypoints only
- G4: DINO semantic-cache entrypoints only
- G5: prototype-bank and text-map entrypoints only
- G6: core attribution entrypoints only
- G7: bag-free inference and evaluation entrypoints only

Do not mix later-gate algorithmic work into earlier-gate loops.

## 8. Terminal behavior
Once G7 has an evidence-backed canonical `PASS`, mainline development must stop.
Only bounded terminal revalidation is allowed unless the authoritative docs are updated.
