# WSOVVIS V9 Mainline Status

This file is the shared memory for the v9 control plane.
It is initialized in draft mode and should be updated once the repository switches authority.

- Current mainline authority: `docs/mainline_v9/*` is drafted; repository authority has not yet switched from `docs/mainline/*`
- Legacy control plane: `docs/mainline/*`
- Active gate: `G0 — V9 control-plane bootstrap and inheritance verification`
- Current gate judgment: `INCONCLUSIVE`
- Acceptance status: `unknown`
- Evidence status: `incomplete`
- Evidence bundle reviewed: `none yet`
- Terminal mainline mode: `inactive`
- Terminal detection explicit: `no`

## Current blocker
The v9 control plane exists only as a draft and is not yet the repository-wide execution source.
`AGENTS.md`, startup instructions, and supervisor tooling still point to `docs/mainline/*`.
The evidence-backed PASS policy is not yet wired into the active control plane.

## Smallest next valid step
Confirm the v9 control-plane document set, then prepare a bounded authority-switch patch that updates:
- `AGENTS.md`
- startup instructions that point to `docs/mainline/*`
- any prompt-generation path that must read `docs/mainline_v9/*`
- any report templates that must emit evidence-pack locations

## Canonical evidence
- none yet for the v9 control plane
- local draft only

## Evidence-pack expectation for the active gate
Before G0 can pass, provide:
- a control-plane readout or dry-run proving the v9 docs are the active interpretation source
- a worked example showing active-gate resolution under the v9 docs
- a short authority-chain comparison showing that legacy docs are no longer the active authority

## Out-of-scope while G0 is active
- algorithmic code changes
- attribution redesign
- refinement work
- enabling any default-off module

## Notes
- Legacy Stage C / G5 / G6 assets remain available as baseline or scaffold material.
- The v9 core path is not yet the active authority until the control-plane switch is explicitly completed.
