# WSOVVIS Mainline Automation Index

This directory is the **authoritative execution source** for document-driven automation.
It is designed to operate without any old workflow documents.

Read order:
1. `EXECUTION_SOURCE.md` — compact execution-source summary
2. `PLAN.md` — the mainline source of truth
3. `IMPLEMENT.md` — execution rules for Codex
4. `STATUS.md` — current active phase/gate and latest evidence
5. `METRICS_ACCEPTANCE.md` — gate-level PASS/FAIL/INCONCLUSIVE rules
6. `FAILURE_PLAYBOOK.md` — fallback order when a gate does not pass
7. `ENVIRONMENT_AND_VALIDATION.md` — environment model, canonical runner recipe, and validation rules
8. `CODEBASE_MAP.md` — code entrypoints and module roles
9. `STAGEB_INTERFACE_CONTRACT.md` — required technical interface inherited from existing code
10. `SUPERVISOR_DEPLOYMENT.md` — supervisor-specific deployment rules
11. `SUPERVISOR_STATE_MACHINE.md` — bounded supervisor state machine
12. `../runbooks/mainline_phase_gate_runbook.md` — concise gate transition recipe

These files replace old workflow control docs. Historical memos, handoffs, task packs, and old workflow READMEs are not authoritative under this kit.
