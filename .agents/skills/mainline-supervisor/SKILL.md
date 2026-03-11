---
name: mainline-supervisor
description: Supervises one bounded WS-OVVIS mainline loop. Determines the active gate, proposes or executes the smallest next valid step, triggers acceptance evaluation, and writes results back to reports and status files.
---

# WS-OVVIS Mainline Supervisor

Use this skill when the repository is operating in document-driven automation mode and you need to advance the mainline by one controlled step.

## Required reading order
1. `AGENTS.md`
2. `START_AUTOMATION.md`
3. `docs/mainline/INDEX.md`
4. `docs/mainline/EXECUTION_SOURCE.md`
5. `docs/mainline/PLAN.md`
6. `docs/mainline/IMPLEMENT.md`
7. `docs/mainline/STATUS.md`
8. `docs/mainline/METRICS_ACCEPTANCE.md`
9. `docs/mainline/EVIDENCE_REQUIREMENTS.md`
10. `docs/mainline/FAILURE_PLAYBOOK.md`
11. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
12. `docs/mainline/CODEBASE_MAP.md`
13. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
14. `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
15. `docs/mainline/SUPERVISOR_DEPLOYMENT.md`
16. `docs/runbooks/mainline_phase_gate_runbook.md`

## Purpose
This skill is a **bounded supervisor**, not an unrestricted autonomous researcher.
Its job is to:
- determine the active gate
- identify the blocking acceptance condition
- produce or execute the smallest next valid step
- avoid scope expansion
- route failures into the documented fallback path
- write back the updated state
- leave durable evidence artifacts that justify the current gate judgment

## Hard rules
- Never widen scope beyond the current gate.
- Never enable default-off modules unless the documents explicitly promote them.
- Never introduce new algorithmic mechanisms during gate plumbing unless a targeted validation failure proves a minimal scoped fix is required.
- Prefer canonical remote validation when local execution is blocked or non-canonical.
- During `G0`, prioritize authority-switch and environment inheritance verification over algorithmic coding.
- When terminal mode is active, stop by default and allow only bounded terminal revalidation.
- `G7` is the terminal gate for this control plane.

## Output requirements
For each invocation, produce:
1. `active phase/gate`
2. `blocking acceptance condition`
3. `smallest next valid step`
4. `out-of-scope modules`
5. `fallback path if blocked or failed`
6. update `docs/mainline/STATUS.md` if state materially changed
7. write the full result to the appropriate report files under `docs/mainline/reports/`
8. ensure any required evidence/worked-example outputs are present before treating a gate as passed
