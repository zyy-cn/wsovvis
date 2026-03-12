# WS-OVVIS Automation AGENTS

This repository runs in **document-driven automation mode**.

## Engineering control plane
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

## Scientific overlay authority (v2 / v10-refined)
When a scientific gate or scientific migration task is active, also read in this exact order after the engineering control plane:
1. `docs/scientific/INDEX.md`
2. `docs/scientific/V10_RECONCILIATION_MEMO.md`
3. `docs/scientific/P0_EXPERIMENTAL_CHARTER.md`
4. `docs/scientific/STATUS.md`
5. the currently active scientific gate spec under `docs/scientific/`
6. `docs/scientific/reports/README.md`

Engineering PASS is necessary but not sufficient for Scientific PASS.
Engineering reports under `docs/mainline/reports/` and scientific reports under `docs/scientific/reports/` must be kept separate.

## Mission
The current mainline exists to prove the project's core claim:

> Under clip-level incomplete positive evidence `Y'(v)`, a class-agnostic video instance basis plus clip-level global tracks, DINO-only track semantics, seen visual prototypes, a class-level text map, and core open-world attribution should support bag-free open-vocabulary video instance inference without relying on the observed bag at test time.

Any code change that does not directly strengthen that claim should **not** enter the mainline by default.

## Authority and precedence
Priority order:
1. This `AGENTS.md`
2. `docs/mainline/*` for engineering readiness, bounded execution, and canonical validation
3. `docs/scientific/*` for scientific validation, comparators, metric hierarchies, evidence packs, pass rules, formal-vs-diagnostic mode semantics, and human scientific sign-off when a scientific gate is active
4. repo skills under `.agents/skills/*`
5. code and tests
6. the current user prompt, only when it does not conflict with 1-5

Historical handoff docs, old Stage-C / Stage-D control packs, milestone memos, and deleted workflow files are **not authoritative** under this control plane.

## Scientific overlay v2 note
- The prior strict `S1 Basis Superiority` / `S1R` recovery path is superseded by the v10-refined scientific overlay.
- Existing `S1 FAIL` / `S1R` materials remain historical evidence, not the active charter.
- Formal scientific progression remains ordered, but diagnostic-only probing and recovery analysis may run under explicit bounded gate specs without activating later formal gates.

## Default-on mainline scope
Default-on:
- clip-level weak-label protocol generation for `Y'(v)` under the outline
- class-agnostic Stage A / Stage B basis and local-tracklet export path
- clip-level global-track-bank closure
- DINO-only track semantic carrier `z_tau` and objectness `o_tau`
- seen visual prototypes plus class-level text map `A`
- core open-world attribution on `Y'(v) + bg + unk`
- bag-free full-vocabulary inference and canonical evaluation

## Default-off modules
Do **not** enable these unless the current engineering or scientific gate explicitly authorizes them:
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
- Make the **smallest valid step** toward the current active engineering gate and, when applicable, the current active scientific gate.
- Distinguish formal scientific progression from diagnostic-only probing and recovery-mode work.
- Do not widen scope to make a failing result look better.
- Prefer the documented fallback or recovery path over adding a new mechanism.
- If docs and code disagree, report the conflict before patching.

## Validation model
- Local checks are informative.
- Canonical PASS is determined by `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`.
- A remote PASS only counts if remote `HEAD == intended local commit`.
- An engineering gate PASS counts only if the engineering acceptance contract and required engineering evidence pack are both complete.
- A scientific gate PASS counts only if the scientific comparator, metric, evidence, and sign-off rules in `docs/scientific/*` are all satisfied.
- Diagnostic-only evidence may inform recovery decisions but may not substitute for formal Scientific PASS unless the active charter explicitly says so.
- Environment mismatch, wrapper failure, bootstrap-link failure, push-route failure, or remote-HEAD mismatch must be classified as `BLOCKED` or `INCONCLUSIVE`, not as algorithmic failure.

## Reporting discipline
Engineering artifacts belong under `docs/mainline/reports/`.
Scientific artifacts belong under `docs/scientific/reports/`.
Do not merge the two layers into a single report set.

## When blocked
If current acceptance cannot be reached:
1. identify the blocking acceptance or evidence gap
2. follow the relevant engineering or scientific fallback / recovery path
3. update the appropriate `STATUS.md`
4. do not open default-off modules unless the governing docs explicitly authorize that step

## Terminal rule
The documented engineering mainline ends at `G7`.
Once `G7` has an evidence-backed canonical `PASS`, stop by default and allow only bounded terminal revalidation under the current scope.
