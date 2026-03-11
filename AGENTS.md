# WS-OVVIS Automation AGENTS

This repository runs in **document-driven automation mode**.

Read in this exact order before planning or coding:
1. `docs/mainline/INDEX.md`
2. `docs/mainline/PLAN.md`
3. `docs/mainline/IMPLEMENT.md`
4. `docs/mainline/STATUS.md`
5. `docs/mainline/METRICS_ACCEPTANCE.md`
6. `docs/mainline/EVIDENCE_REQUIREMENTS.md`
7. `docs/mainline/FAILURE_PLAYBOOK.md`
8. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
9. `docs/mainline/CODEBASE_MAP.md`
10. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
11. `docs/runbooks/mainline_phase_gate_runbook.md`

## Mission
The current mainline exists to prove the project's core claim:

> Under clip-level incomplete positive evidence `Y'(v)`, a class-agnostic video instance basis plus clip-level global tracks, DINO-only track semantics, seen visual prototypes, a class-level text map, and core open-world attribution should support bag-free open-vocabulary video instance inference without relying on the observed bag at test time.

Any code change that does not directly strengthen that claim should **not** enter the mainline by default.

## Authority and precedence
Priority order:
1. This `AGENTS.md`
2. `docs/mainline/*`
3. repo skills under `.agents/skills/*`
4. code and tests
5. the current user prompt, only when it does not conflict with 1-4

Historical handoff docs, old Stage-C / Stage-D control packs, milestone memos, and deleted workflow files are **not authoritative** under this control plane.

## Default-on mainline scope
Default-on:
- clip-level weak-label protocol generation for `Y'(v)` under the v9 outline
- class-agnostic Stage A / Stage B basis and local-tracklet export path
- clip-level global-track-bank closure
- DINO-only track semantic carrier `z_tau` and objectness `o_tau`
- seen visual prototypes plus class-level text map `A`
- core open-world attribution on `Y'(v) + bg + unk`
- bag-free full-vocabulary inference and canonical evaluation

## Default-off modules
Do **not** enable these unless the current gate explicitly authorizes them:
- prototype EM / momentum refresh
- candidate retrieval
- warm-up BCE
- temporal consistency
- unknown fallback at inference
- one-round quality-aware refinement
- any multi-round continuation
- scenario/domain-missing protocol as mainline
- prompt ensemble, synonym expansion, or held-out unrestricted text branches

## Operating style
- Make the **smallest valid step** toward the current active gate.
- Do not widen scope to make a failing result look better.
- Prefer the documented fallback path over adding a new mechanism.
- If docs and code disagree, report the conflict before patching.

## Validation model
- Local checks are informative.
- Canonical PASS is determined by `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`.
- A remote PASS only counts if remote `HEAD == intended local commit`.
- A gate PASS counts only if the acceptance contract and the required evidence pack are both complete.
- Environment mismatch, wrapper failure, bootstrap-link failure, push-route failure, or remote-HEAD mismatch must be classified as `BLOCKED` or `INCONCLUSIVE`, not as algorithmic failure.

## Reporting discipline
- Every phase/gate check must write `docs/mainline/reports/phase_gate_latest.txt` plus a timestamped archive copy.
- Every acceptance evaluation must write `docs/mainline/reports/acceptance_latest.txt` plus a timestamped archive copy.
- Every evidence review must write `docs/mainline/reports/evidence_latest.txt` plus a timestamped archive copy.
- Every gate requiring a worked example must also write `docs/mainline/reports/worked_example_verification_latest.md` and `.json` plus timestamped archive copies.
- `docs/mainline/STATUS.md` must be updated whenever the active gate, blockers, acceptance status, evidence status, or canonical evidence materially change.

## When blocked
If current acceptance cannot be reached:
1. identify the blocking acceptance or evidence gap
2. follow `docs/mainline/FAILURE_PLAYBOOK.md`
3. update `docs/mainline/STATUS.md`
4. do not open default-off modules unless the docs explicitly authorize that step

## Terminal rule
The documented mainline ends at `G7`.
Once `G7` has an evidence-backed canonical `PASS`, stop by default and allow only bounded terminal revalidation under the current scope.
