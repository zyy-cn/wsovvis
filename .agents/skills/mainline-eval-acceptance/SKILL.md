---
name: mainline-eval-acceptance
description: Evaluate the latest WS-OVVIS implementation or experiment result against the project-private acceptance contract and decide PASS, FAIL, INCONCLUSIVE, or BLOCKED.
---

# WS-OVVIS Evaluate Acceptance

Use this skill after a coding or experiment step.

## Read order
1. `AGENTS.md`
2. `docs/mainline/INDEX.md`
3. `docs/mainline/STATUS.md`
4. `docs/mainline/METRICS_ACCEPTANCE.md`
5. `docs/mainline/EVIDENCE_REQUIREMENTS.md`
6. `docs/mainline/FAILURE_PLAYBOOK.md`
7. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
8. any files, logs, test results, metrics, worked examples, or evidence reports produced by the current step

## Your task
Return a concise structured judgment with:
1. active phase/gate being evaluated
2. evidence reviewed
3. acceptance judgment
4. evidence judgment
5. PASS / FAIL / INCONCLUSIVE / BLOCKED
6. which acceptance conditions passed
7. which acceptance conditions failed or are missing
8. which evidence requirements passed or are missing
9. whether the next gate may activate
10. if not, the exact fallback path
11. what should be written back to `STATUS.md`

## Hard rules
- Do not claim PASS when canonical validation is missing but required.
- Do not claim PASS on stale-commit remote validation.
- If key metrics are required by the gate but missing, return `INCONCLUSIVE`.
- If acceptance is satisfied but required evidence is missing, return `INCONCLUSIVE`, not `PASS`.
- Write the result to `docs/mainline/reports/acceptance_latest.txt`, `docs/mainline/reports/evidence_latest.txt`, and timestamped archive copies.
