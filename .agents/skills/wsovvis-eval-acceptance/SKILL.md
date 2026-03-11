---
name: wsovvis-eval-acceptance
description: Evaluate the latest implementation or experiment result against the WSOVVIS mainline acceptance contract and decide PASS, FAIL, or INCONCLUSIVE.
---

# WSOVVIS Evaluate Acceptance

Use this skill after a coding or experiment step.

## Read order
1. `AGENTS.md`
2. `docs/mainline/INDEX.md`
3. `docs/mainline/STATUS.md`
4. `docs/mainline/METRICS_ACCEPTANCE.md`
5. `docs/mainline/FAILURE_PLAYBOOK.md`
6. any files, logs, test results, or metrics produced by the current step

## Your task
Return a concise structured judgment with:
1. active phase/gate being evaluated
2. evidence reviewed
3. PASS / FAIL / INCONCLUSIVE
4. which acceptance conditions passed
5. which acceptance conditions failed or are missing
6. whether the next gate may activate
7. if not, the exact fallback path from `FAILURE_PLAYBOOK.md`
8. what should be written back to `STATUS.md`

## Hard rules
- Do not claim PASS when canonical validation is missing but required.
- Do not claim PASS on stale-commit remote validation.
- If key metrics such as `HPR` / `UAR` are required by the gate but missing, return `INCONCLUSIVE`.
