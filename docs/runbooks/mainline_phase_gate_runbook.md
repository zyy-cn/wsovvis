# WSOVVIS Mainline Phase/Gate Runbook

## Purpose
A concise operational recipe for moving through the mainline gates.

## Standard loop
1. Read `AGENTS.md`.
2. Read `docs/mainline/INDEX.md` and linked files.
3. Identify the active gate from `STATUS.md`.
4. Identify the blocking acceptance from `METRICS_ACCEPTANCE.md`.
5. Propose the smallest valid step.
6. Execute the step.
7. Evaluate PASS / FAIL / INCONCLUSIVE.
8. Update `STATUS.md`.

## Gate transition rule
Move to the next gate only after the current gate has an evidence-backed `PASS`.

## Terminal rule
If the current accepted gate is the documented terminal gate:
- do not activate a new gate
- stop and write/update the terminal summary
- use bounded terminal revalidation only if fresh evidence is required

## Failure rule
If the current gate does not pass:
- use `FAILURE_PLAYBOOK.md`
- do not open default-off branches unless the docs explicitly allow it

## First-run bootstrap
If the repository has existing mature code but no active automation status:
- start at `G0`
- verify environment, codebase map, Stage B contract, and canonical validation semantics
- only then promote to the first evidence-backed active gate
