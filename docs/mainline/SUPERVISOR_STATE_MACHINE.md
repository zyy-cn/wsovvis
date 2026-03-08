# WS-OVVIS Mainline Supervisor State Machine

This document defines the bounded supervisor behavior for document-driven automation.

## Purpose
The supervisor advances the repository by **one smallest valid step** at a time.
It is intentionally conservative and must not substitute broad algorithm exploration
for gate-driven progress.

## States
- `G0`: repository bootstrap and environment verification
- `G1`: protocol and baseline alignment
- `G2`: Stage B export / bridge / consumer integrity
- `G3`: mixed representation and Stage C semantic plumbing
- `G4`: open-world attribution validation
- `G5`: full-video linking and inference closure
- `G6`: single-round bounded refinement

## Supervisor loop
For one bounded iteration:
1. Read `STATUS.md` and identify the current gate.
2. Run phase/gate check.
3. If the gate is blocked by missing evidence, collect only the minimal evidence needed.
4. If the gate is blocked by a scoped wiring failure, apply only the minimal scoped fix.
5. Run acceptance evaluation.
6. Update `STATUS.md` and report files.
7. Stop.

## Allowed transitions
- `PASS` -> advance to the next gate
- `INCONCLUSIVE` -> remain in current gate and request the smallest next step
- `BLOCKED` -> remain in current gate and route to the documented fallback path
- `FAIL` -> remain in current gate and apply the documented failure playbook before retrying

## Terminal condition
If the active gate is the accepted terminal gate and it is already `PASS`:
- enter terminal-mainline mode
- stop automatically
- do not generate a new coding step
- write/update `docs/mainline/reports/mainline_terminal_summary.txt`
- allow only bounded terminal revalidation under the current scope

### Terminal revalidation mode
Terminal revalidation may:
- rerun the authoritative bounded G4 / G5 / G6 regression suite
- reuse canonical `gpu4090d` validation semantics
- refresh `STATUS.md`, `phase_gate_latest.txt`, `acceptance_latest.txt`, and `mainline_terminal_summary.txt` if evidence changes

Terminal revalidation may not:
- activate a new gate
- widen scope beyond the accepted mainline
- enable default-off branches

## Prohibited behavior
- skipping gates
- enabling default-off modules without documented promotion
- broad algorithm changes during plumbing/integrity gates
- interpreting environment failure as algorithm failure

## Reporting
Each supervisor iteration must write:
- `docs/mainline/reports/phase_gate_latest.txt`
- `docs/mainline/reports/acceptance_latest.txt`
- timestamped archival copy under `docs/mainline/reports/archive/`
- any required updates to `docs/mainline/STATUS.md`
- when terminal-mainline mode is active:
  `docs/mainline/reports/mainline_terminal_summary.txt`
