# WS-OVVIS Scientific Overlay Status

This file tracks the current authoritative state for the scientific validation overlay.

## Current state
- Active scientific gate: `P0`
- Scientific status: `INCONCLUSIVE`
- Last scientific PASS: `none`
- Next planned scientific gate after P0 PASS: `S1`
- Human scientific sign-off status: `pending`
- Scientific evidence bundle reviewed: `20260311T164159Z bounded P0 charter-verification loop`

## Why P0 is active
The engineering control plane is already active, but the scientific overlay must first freeze the comparator, metric, evidence, canonical, shortcut, and sign-off charter before any scientific gate may formally PASS.
The bounded `20260311T164159Z` P0 loop verified that `docs/scientific/P0_EXPERIMENTAL_CHARTER.md` is present and readable, and that the scientific overlay is referenced by `AGENTS.md`, `docs/mainline/INDEX.md`, `docs/scientific/INDEX.md`, and this status file.
However, the same loop also recorded that `git status --short docs/scientific` reports `?? docs/scientific/`, so the scientific overlay is not yet evidenced as a stable tracked in-repo authority bundle.
Because `P0` is not yet evidence-backed `PASS`, `S1` may not activate.

## Immediate next step
- durably anchor the scientific overlay files as stable in-repo artifacts
- rerun the bounded `P0` evidence loop and refresh the scientific reports under `docs/scientific/reports/`
- do not activate `S1` until `P0` is evidence-backed `PASS`
