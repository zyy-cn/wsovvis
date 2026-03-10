# WSOVVIS V9 Mainline Index

This directory is the **v9 control-plane draft**.
It mirrors the current document-driven automation mode without changing the existing authority yet.

## Status of this control plane
- current role: draft / parallel control plane
- current authority: `docs/mainline/*` remains authoritative until `AGENTS.md` and the supervisor entrypoints are explicitly switched
- migration intent: replace the old mainline objective with the v9 outline objective while preserving the exact same development mode

## Read order for a v9-controlled session
1. `AGENTS.md`
2. `docs/mainline_v9/INDEX.md`
3. `docs/mainline_v9/EXECUTION_SOURCE.md`
4. `docs/mainline_v9/PLAN.md`
5. `docs/mainline_v9/IMPLEMENT.md`
6. `docs/mainline_v9/STATUS.md`
7. `docs/mainline_v9/METRICS_ACCEPTANCE.md`
8. `docs/mainline_v9/EVIDENCE_REQUIREMENTS.md`
9. `docs/mainline_v9/FAILURE_PLAYBOOK.md`
10. `docs/mainline_v9/ENVIRONMENT_AND_VALIDATION.md`
11. `docs/mainline_v9/CODEBASE_MAP.md`
12. `docs/mainline_v9/STAGEB_INTERFACE_CONTRACT.md`
13. `docs/runbooks/mainline_phase_gate_runbook.md`
14. `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
15. `docs/mainline/SUPERVISOR_DEPLOYMENT.md`

## Purpose
This control plane exists to let the repository keep the same:
- document-driven workflow
- bounded supervisor loop
- gate acceptance discipline
- evidence-backed PASS discipline
- terminal stop rule

while changing only the algorithmic end goal to the v9 mainline.

## V9 mainline objective
The v9 mainline exists to prove the following core path:

> Under clip-level incomplete positive evidence `Y'(v)`, a class-agnostic video instance basis plus DINO-only track semantics, seen visual prototypes, a class-level text map, and core open-world attribution should support bag-free open-vocabulary video instance inference better than closed-world weak supervision.

## Core / optional split
Core mainline:
- clip-level protocol `Y'(v)`
- class-agnostic instance basis
- fixed clip-level global track bank
- DINO-only track semantic carrier `z_tau`
- seen visual prototype + class-level text map `A`
- core attribution on `Y'(v) + bg + unk`
- bag-free full-vocab inference

Optional / failure-playbook only:
- prototype EM / momentum refresh
- candidate retrieval
- warm-up BCE
- temporal consistency
- unknown fallback
- one-round quality-aware refinement
