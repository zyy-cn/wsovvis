---
name: wsovvis-phase-gate-check
description: Determine the active WSOVVIS mainline phase/gate, blocking acceptance, smallest next valid step, and fallback path using the new document-driven automation layer.
---

# WSOVVIS Phase/Gate Check

Use this skill when you need to decide what the repository should do next under the mainline.

## Read order
1. `AGENTS.md`
2. `docs/mainline/INDEX.md`
3. `docs/mainline/PLAN.md`
4. `docs/mainline/STATUS.md`
5. `docs/mainline/METRICS_ACCEPTANCE.md`
6. `docs/mainline/FAILURE_PLAYBOOK.md`
7. `docs/mainline/CODEBASE_MAP.md`
8. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
9. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`

## Your task
Output a concise structured decision with:
1. active phase/gate
2. why that gate is active
3. blocking acceptance conditions
4. explicitly out-of-scope modules
5. smallest next valid coding task
6. fallback path if that task does not pass
7. exact files likely to be touched
8. exact commands likely to be run

## Hard rules
- Do not widen scope.
- Do not enable default-off modules.
- If evidence is missing, mark the gate `INCONCLUSIVE` rather than guessing.
- Prefer smaller implementation steps over larger redesigns.
