---
name: mainline-phase-gate-check
description: Determine the active WS-OVVIS mainline gate, blocking acceptance, smallest next valid step, and fallback path using the project-private document-driven control layer.
---

# WS-OVVIS Mainline Phase/Gate Check

Use this skill when you need to decide what the repository should do next under the active mainline.

## Read order
1. `AGENTS.md`
2. `docs/mainline/INDEX.md`
3. `docs/mainline/PLAN.md`
4. `docs/mainline/STATUS.md`
5. `docs/mainline/METRICS_ACCEPTANCE.md`
6. `docs/mainline/EVIDENCE_REQUIREMENTS.md`
7. `docs/mainline/FAILURE_PLAYBOOK.md`
8. `docs/mainline/CODEBASE_MAP.md`
9. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
10. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`

## Your task
Output a concise structured decision with:
1. active phase/gate
2. why that gate is active
3. blocking acceptance conditions
4. explicitly out-of-scope modules
5. smallest next valid coding or evidence task
6. fallback path if that task does not pass
7. exact files likely to be touched
8. exact commands likely to be run
9. the minimum required evidence pack for the gate

## Hard rules
- Do not widen scope.
- Do not enable default-off modules.
- If evidence is missing, mark the gate `INCONCLUSIVE` rather than guessing.
- Prefer smaller implementation or evidence-closing steps over larger redesigns.
- During `G0`, prioritize authority-switch and environment inheritance verification over algorithmic work.
- Write the result to `docs/mainline/reports/phase_gate_latest.txt` and a timestamped archive copy.
- If the gate decision depends on a specific evidence pack, reference the required evidence outputs explicitly.
