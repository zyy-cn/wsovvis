---
name: wsovvis-mainline-supervisor
description: Supervises one bounded mainline loop for the WS-OVVIS repository. It determines the current active gate from the mainline docs, proposes or executes the smallest next valid step, triggers acceptance evaluation, and writes results back to the report and status files without widening scope.
---

# WS-OVVIS Mainline Supervisor

Use this skill when the repository is operating in document-driven automation mode and you need to advance the current mainline by one controlled step.

## Required reading order
Before doing anything, read these files in this exact order:
1. `AGENTS.md`
2. `START_AUTOMATION.md`
3. `docs/mainline/INDEX.md`
4. `docs/mainline/EXECUTION_SOURCE.md`
5. `docs/mainline/PLAN.md`
6. `docs/mainline/IMPLEMENT.md`
7. `docs/mainline/STATUS.md`
8. `docs/mainline/METRICS_ACCEPTANCE.md`
9. `docs/mainline/FAILURE_PLAYBOOK.md`
10. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
11. `docs/mainline/CODEBASE_MAP.md`
12. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
13. `docs/runbooks/mainline_phase_gate_runbook.md`
14. `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
15. `docs/mainline/SUPERVISOR_DEPLOYMENT.md`

## Purpose
This skill is a **bounded supervisor**, not an unrestricted autonomous researcher.
Its job is to:
- determine the active gate
- identify the blocking acceptance condition
- produce or execute the smallest next valid step
- avoid scope expansion
- route failures into the documented fallback path
- write back the updated state

## Hard rules
- Never widen scope beyond the current gate.
- Never enable default-off modules unless the documents explicitly promote them.
- Never introduce new algorithmic mechanisms during gate plumbing unless a targeted validation failure proves a minimal scoped fix is required.
- Prefer canonical remote validation when local execution is blocked or non-canonical.
- Treat environment mismatch, remote mismatch, missing wrapper evidence, missing bootstrap evidence, and missing push-route evidence as `BLOCKED` or `INCONCLUSIVE`, not as algorithmic failure.
- During G0, prioritize environment inheritance verification over any algorithm coding.

## Output requirements
For each invocation, produce:
1. `active phase/gate`
2. `blocking acceptance condition`
3. `smallest next valid step`
4. `out-of-scope modules`
5. `fallback path if blocked or failed`
6. update `docs/mainline/STATUS.md` if state materially changed
7. write the full result to the appropriate report file under `docs/mainline/reports/`
8. if the accepted terminal gate is already reached, write/update `docs/mainline/reports/mainline_terminal_summary.txt`

## Loop discipline
If asked to execute one loop:
1. run/trigger `wsovvis-phase-gate-check`
2. if implementation is needed, perform only the smallest valid scoped step
3. run/trigger `wsovvis-eval-acceptance`
4. update `STATUS.md`
5. stop and report

Do not recursively continue beyond one bounded loop unless the caller explicitly asks for multiple iterations.

## Terminal-mainline rule
If the documented terminal gate is already accepted:
- stop automatically
- do not generate a new coding step
- write/update the terminal summary
- allow only bounded terminal revalidation under the current documented scope
