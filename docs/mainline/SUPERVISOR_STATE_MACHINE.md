# WS-OVVIS Supervisor State Machine

## Purpose
This file defines the bounded state machine for the mainline supervisor.

## States
- `gate-progression`
- `blocked`
- `inconclusive`
- `terminal-mainline`

## Gate order
`G0 -> G1 -> G2 -> G3 -> G4 -> G5 -> G6 -> G7`

## Transition rules
- Start in `gate-progression` at the active gate recorded in `STATUS.md`.
- A transition to the next gate is allowed only after an evidence-backed `PASS`, not after a contract-only candidate pass.
- `INCONCLUSIVE` keeps the workflow in the current gate and restricts the next loop to the smallest evidence-closing or scoped wiring step.
- `BLOCKED` keeps the workflow in the current gate and routes the next loop through the relevant fallback path in `FAILURE_PLAYBOOK.md`.
- `FAIL` keeps the workflow in the current gate and requires the documented fallback path before retry.
- `G7` with an evidence-backed canonical `PASS` transitions to `terminal-mainline`.
- In `terminal-mainline`, do not activate a new gate; only bounded terminal revalidation is allowed.

## Terminal-mainline behavior
When terminal mode is active:
- stop by default
- do not propose a new algorithm-development step
- write or update `docs/mainline/reports/mainline_terminal_summary.txt`
- allow only bounded terminal revalidation using the same canonical validation semantics
