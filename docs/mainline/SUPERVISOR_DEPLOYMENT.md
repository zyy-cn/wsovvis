# WS-OVVIS Supervisor Deployment

## Goal
Deploy a conservative supervisor layer on top of the document-driven automation kit.

## Components
- repo skill: `.agents/skills/mainline-supervisor/`
- helper script: `tools/run_mainline_loop.py`
- state machine: `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
- environment authority: `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
- execution runbook: `docs/mainline/IMPLEMENT.md`
- evidence contract: `docs/mainline/EVIDENCE_REQUIREMENTS.md`

## Deployment steps
1. Place all files in the repository root according to their relative paths.
2. Ensure the base automation kit is already deployed.
3. Create reports directories if missing.
4. Confirm the project still contains the authoritative entrypoints named in `CODEBASE_MAP.md`.
5. Treat the first supervisor run as `G0` environment inheritance verification unless the adapted project state proves a later gate is already authoritative.
