# WS-OVVIS Supervisor State Machine

## Purpose
This file defines the bounded state machine for the mainline supervisor.

## States
- `gate-progression`
- `blocked`
- `inconclusive`
- `terminal-mainline`

## Transition rules
- Start in `gate-progression` at the active gate recorded in `STATUS.md`.
- Move to `blocked` when canonical validation prerequisites fail for environment reasons.
- Move to `inconclusive` when acceptance looks plausible but evidence is incomplete or contradictory.
- Return from `blocked` or `inconclusive` only by taking the smallest documented recovery step for the same gate.
- Advance to the next gate only after an evidence-backed `PASS`.
- Enter `terminal-mainline` only after `G7` is accepted and `STATUS.md` marks terminal mode active.
- In `terminal-mainline`, permit only bounded revalidation, report refresh, and evidence integrity maintenance.


A transition to the next gate is allowed only after an evidence-backed `PASS`, not after a contract-only candidate pass.
