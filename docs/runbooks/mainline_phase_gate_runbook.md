# WS-OVVIS Mainline Phase/Gate Runbook

## Standard loop
1. Read `AGENTS.md`.
2. Read `docs/mainline/INDEX.md` and linked files.
3. Identify the active gate from `STATUS.md`.
4. Identify the blocking acceptance from `METRICS_ACCEPTANCE.md`.
5. Identify the required evidence pack from `EVIDENCE_REQUIREMENTS.md`.
6. Propose the smallest valid step.
7. Execute the step.
8. Evaluate acceptance and evidence as `PASS` / `FAIL` / `INCONCLUSIVE` / `BLOCKED`.
9. Update `STATUS.md` and the required evidence reports.

## Gate transition rule
Move to the next gate only after the current gate has an evidence-backed `PASS`.

## Terminal rule
If the accepted terminal gate is reached, stop and enter terminal-mainline mode.
